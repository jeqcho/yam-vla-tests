"""Prepare an Ai2 BimanualYAM HF dataset for DreamZero fine-tuning.

The Ai2 release at <https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset-69f81e17b140ec34f430a35e>
ships **LeRobot v3.0**:

  * `data/chunk-CCC/file-FFF.parquet`           — multiple episodes per shard
  * `videos/observation.images.<cam>/chunk-CCC/file-FFF.mp4`  — concatenated frames
  * `meta/episodes/chunk-CCC/file-FFF.parquet`  — per-episode metadata sidecar
  * `meta/info.json` codebase_version: "v3.0"

DreamZero's loader (`groot/vla/data/dataset/lerobot.py:get_parquet_path` and
`get_video_path`) expects **LeRobot v2** with one file per episode and the cam
dir nested inside the chunk:

  * `data/chunk-EEE/episode_EEEEEE.parquet`
  * `videos/chunk-EEE/observation.images.<cam>_camera-images-rgb/episode_EEEEEE.mp4`

So this Modal job does a real v3→v2 conversion:

  1. Downloads the source repo(s) into `dreamzero-yam-data` volume → `/data/raw/`.
  2. For each shard parquet: splits rows by `episode_index` into one parquet
     per episode at the v2 location. Injects `annotation.task` from
     `meta/tasks.parquet` while we're at it.
  3. For each shard mp4: extracts per-episode video clips using ffmpeg
     (re-encode with libx264 ultrafast — original is AV1 which decord on CPU
     can be slow with). Parallelized via multiprocessing.
  4. Rewrites `meta/info.json` to declare v2 paths and `annotation.task`.
  5. Runs `scripts/data/convert_lerobot_to_gear.py` with bimanual `--state-keys`
     / `--action-keys` to compute stats and emit modality.json/embodiment.json
     and relative_stats_dreamzero.json.
  6. Post-fixes modality.json so video keys carry the `_camera-images-rgb`
     suffix the YAM YAML expects.
  7. Writes a deliverable report at `/data/prepared/<tag>_prep_report.md`.

Run with:
    modal run modal/prepare_yam_data.py::prepare --hf-repo allenai/01122025-box-01
    modal run modal/prepare_yam_data.py::prepare \\
        --hf-repos allenai/01122025-box-01,allenai/01122025-box-02 \\
        --tag yam_box_smoke
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import modal

DREAMZERO_DIR = Path(__file__).parent.parent / "dreamzero"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "huggingface_hub[cli]",
        "hf-transfer",
        "pandas>=2.0",
        "pyarrow>=15",
        "numpy>=1.26,<2.0",
        "tqdm",
        "pillow",
        "opencv-python-headless==4.8.0.74",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(DREAMZERO_DIR), "/opt/dreamzero", copy=True)
)

data_volume = modal.Volume.from_name("dreamzero-yam-data", create_if_missing=True)

app = modal.App("dreamzero-yam-data-prep", image=image)


# ---------------------------------------------------------------------------
# v3 → v2 conversion helpers (run inside the container)
# ---------------------------------------------------------------------------

def _load_episode_annotations(src: Path) -> dict[int, str]:
    """Per-episode language annotation. Ai2 ships this in tasks_annotated.parquet
    keyed by `episode_index` (not by task_index, as the LeRobot v3 spec would
    suggest). Returns {episode_index: task_text}."""
    import pandas as pd
    p = src / "meta" / "tasks_annotated.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        # tasks_annotated.parquet has a single column `task` indexed by `episode_index`.
        if "task" in df.columns:
            if df.index.name == "episode_index" or "episode_index" in str(df.index.name or ""):
                return {int(i): str(t) for i, t in zip(df.index, df["task"])}
            if "episode_index" in df.columns:
                return {int(r.episode_index): str(r.task) for r in df.itertuples()}
        # Fallback: just enumerate rows.
        return {int(i): str(v) for i, v in enumerate(df.iloc[:, 0].astype(str))}
    # Final fallback: derive from tasks.jsonl + episodes_metadata if present.
    p = src / "meta" / "tasks.jsonl"
    if p.exists():
        out = {}
        with open(p) as f:
            for line in f:
                e = json.loads(line)
                out[int(e["task_index"])] = str(e["task"])
        return out
    return {}


def _unbox(v):
    """v3 metadata wraps some scalars in single-element lists (`[0.0]`).
    Unbox to a plain scalar if needed."""
    if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
        return _unbox(v[0])
    return v


def _split_one_shard(args: tuple) -> dict:
    """Split a single v3 shard into per-episode parquets + per-episode mp4s.

    Uses `meta/episodes/chunk-CCC/file-FFF.parquet` for the per-camera video
    shard mapping + timestamps. ffmpeg cuts by timestamp (precise, no
    re-encode unnecessary work). Runs in a worker process.
    """
    import pandas as pd

    (shard_pq, src_root, out_root, cam_keys, ep_annotations, ep_offset) = args
    shard_pq = Path(shard_pq)
    src_root = Path(src_root)
    out_root = Path(out_root)

    df = pd.read_parquet(shard_pq)
    if "annotation.task" not in df.columns and ep_annotations and "episode_index" in df.columns:
        df["annotation.task"] = df["episode_index"].map(ep_annotations).astype("string")

    chunk_str = shard_pq.parent.name  # "chunk-000"
    shard_name = shard_pq.stem        # "file-000"

    # Per-episode metadata sidecar — tells us which video file + timestamps.
    ep_meta_path = src_root / "meta" / "episodes" / chunk_str / f"{shard_name}.parquet"
    ep_meta_lookup: dict[int, dict] = {}
    if ep_meta_path.exists():
        em = pd.read_parquet(ep_meta_path)
        for _, row in em.iterrows():
            ep_idx_raw = int(_unbox(row["episode_index"]))
            entry: dict = {"length": int(_unbox(row["length"])), "cams": {}}
            for cam in cam_keys:
                fi_col = f"videos/observation.images.{cam}/file_index"
                ci_col = f"videos/observation.images.{cam}/chunk_index"
                from_col = f"videos/observation.images.{cam}/from_timestamp"
                to_col = f"videos/observation.images.{cam}/to_timestamp"
                if fi_col in row.index:
                    entry["cams"][cam] = {
                        "file_index": int(_unbox(row[fi_col])),
                        "chunk_index": int(_unbox(row[ci_col])),
                        "from_ts": float(_unbox(row[from_col])),
                        "to_ts": float(_unbox(row[to_col])),
                    }
            ep_meta_lookup[ep_idx_raw] = entry
    else:
        print(f"  !! no ep meta at {ep_meta_path}; videos will be skipped for shard {shard_name}",
              flush=True)

    written_episodes = []
    for raw_ep_idx, ep_df in df.groupby("episode_index", sort=True):
        raw_ep_idx = int(raw_ep_idx)
        ep_idx = raw_ep_idx + ep_offset
        chunk_idx = ep_idx // 1000

        # Per-episode parquet.
        out_chunk = out_root / "data" / f"chunk-{chunk_idx:03d}"
        out_chunk.mkdir(parents=True, exist_ok=True)
        ep_df = ep_df.copy()
        ep_df["episode_index"] = ep_idx
        out_pq = out_chunk / f"episode_{ep_idx:06d}.parquet"
        ep_df.to_parquet(out_pq)

        meta = ep_meta_lookup.get(raw_ep_idx, {})
        ep_len = meta.get("length", len(ep_df))

        # Per-camera video cut, driven by v3 episode metadata.
        for cam in cam_keys:
            cam_meta = meta.get("cams", {}).get(cam)
            if cam_meta is None:
                continue
            src_mp4 = (src_root / "videos" / f"observation.images.{cam}"
                       / f"chunk-{cam_meta['chunk_index']:03d}"
                       / f"file-{cam_meta['file_index']:03d}.mp4")
            if not src_mp4.exists():
                print(f"  !! missing video {src_mp4}", flush=True)
                continue
            out_cam_dir = (out_root / "videos" / f"chunk-{chunk_idx:03d}"
                           / f"observation.images.{cam}_camera-images-rgb")
            out_cam_dir.mkdir(parents=True, exist_ok=True)
            out_mp4 = out_cam_dir / f"episode_{ep_idx:06d}.mp4"
            if out_mp4.exists():
                continue
            # Re-encode with libx264 ultrafast — small clip, decord handles
            # h264 well, and stream-copying AV1 wouldn't be keyframe-aligned.
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
                "-ss", f"{cam_meta['from_ts']:.6f}",
                "-to", f"{cam_meta['to_ts']:.6f}",
                "-i", str(src_mp4),
                "-an",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(out_mp4),
            ]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                print(f"  ffmpeg rc={r.returncode} for ep {ep_idx} cam {cam}: "
                      f"{r.stderr.decode()[:200]}", flush=True)

        written_episodes.append({
            "episode_index": ep_idx, "length": int(ep_len),
            "tasks": list(ep_df["annotation.task"].dropna().unique())
                     if "annotation.task" in ep_df.columns else [""],
        })

    return {"shard": str(shard_pq), "episodes": written_episodes}


# numpy injected at module scope so the worker can see it (no extra import).
import numpy as np  # noqa: E402


def _convert_v3_to_v2(src_root: Path, out_root: Path, ep_offset: int,
                      max_workers: int = 8) -> int:
    """Convert one Ai2 v3 dataset into DreamZero v2 layout. Returns episodes written."""
    if out_root.exists():
        # Don't blow away on every call — keep what's there for resume.
        pass
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "data").mkdir(exist_ok=True)
    (out_root / "videos").mkdir(exist_ok=True)
    (out_root / "meta").mkdir(exist_ok=True)

    # Discover layout.
    with open(src_root / "meta" / "info.json") as f:
        info = json.load(f)
    cam_keys = []
    for fname, fmeta in info.get("features", {}).items():
        if fmeta.get("dtype") == "video" and fname.startswith("observation.images."):
            cam_keys.append(fname.replace("observation.images.", ""))
    print(f"  cameras: {cam_keys}", flush=True)

    ep_annotations = _load_episode_annotations(src_root)
    print(f"  per-episode annotations: {len(ep_annotations)} entries "
          f"(sample: {next(iter(ep_annotations.items()), None)})", flush=True)

    shard_parquets = sorted((src_root / "data").rglob("file-*.parquet"))
    print(f"  v3 shards: {len(shard_parquets)} parquet files", flush=True)

    work = [(str(p), str(src_root), str(out_root), cam_keys, ep_annotations, ep_offset)
            for p in shard_parquets]

    total_eps = 0
    all_episodes: list[dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_split_one_shard, w) for w in work]
        for fut in as_completed(futures):
            res = fut.result()
            total_eps += len(res["episodes"])
            all_episodes.extend(res["episodes"])
            print(f"    + {len(res['episodes']):3d} eps from {Path(res['shard']).name}",
                  flush=True)

    return total_eps, all_episodes, info, cam_keys, ep_annotations


def _write_v2_meta(out_root: Path, info_v3: dict, episodes: list[dict],
                   ep_annotations: dict, cam_keys: list[str]) -> None:
    """Rewrite meta/info.json + tasks.jsonl + episodes.jsonl to DreamZero's v2 expectations."""
    info_v2 = dict(info_v3)
    info_v2["codebase_version"] = "v2.1"
    info_v2["total_episodes"] = len(episodes)
    info_v2["total_frames"] = sum(e["length"] for e in episodes)
    info_v2["total_chunks"] = max(1, (len(episodes) // 1000) + 1)
    info_v2["chunks_size"] = 1000
    info_v2["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info_v2["video_path"] = (
        "videos/chunk-{episode_chunk:03d}/observation.images.{video_key}/episode_{episode_index:06d}.mp4"
    )

    # Rename feature keys so observation.images.<cam> becomes
    # observation.images.<cam>_camera-images-rgb (matches the renamed dirs).
    new_features = {}
    for k, v in info_v2.get("features", {}).items():
        if v.get("dtype") == "video" and k.startswith("observation.images."):
            cam = k.replace("observation.images.", "")
            new_features[f"observation.images.{cam}_camera-images-rgb"] = v
        else:
            new_features[k] = v
    # Add annotation.task as a feature so convert_lerobot_to_gear.py finds it.
    new_features["annotation.task"] = {"dtype": "string"}
    info_v2["features"] = new_features

    with open(out_root / "meta" / "info.json", "w") as f:
        json.dump(info_v2, f, indent=4)

    # tasks.jsonl: assign a contiguous task_index per UNIQUE task string seen.
    unique_tasks: dict[str, int] = {}
    for text in ep_annotations.values():
        t = str(text)
        if t and t not in unique_tasks:
            unique_tasks[t] = len(unique_tasks)
    if not unique_tasks:
        unique_tasks[""] = 0
    with open(out_root / "meta" / "tasks.jsonl", "w") as f:
        for text, idx in sorted(unique_tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": int(idx), "task": str(text)}) + "\n")

    with open(out_root / "meta" / "episodes.jsonl", "w") as f:
        for ep in sorted(episodes, key=lambda e: e["episode_index"]):
            f.write(json.dumps(ep) + "\n")


def _write_prep_report(dataset_dir: Path, report_path: Path) -> None:
    import pandas as pd
    import cv2

    with open(dataset_dir / "meta" / "info.json") as f:
        info = json.load(f)
    modality = {}
    if (dataset_dir / "meta" / "modality.json").exists():
        with open(dataset_dir / "meta" / "modality.json") as f:
            modality = json.load(f)
    embodiment = {}
    if (dataset_dir / "meta" / "embodiment.json").exists():
        with open(dataset_dir / "meta" / "embodiment.json") as f:
            embodiment = json.load(f)

    parquets = sorted((dataset_dir / "data").rglob("episode_*.parquet"))
    ep_lens = []
    sample_tasks = []
    sample_states = []
    for pq in parquets[:5]:
        df = pd.read_parquet(pq)
        ep_lens.append(len(df))
        if "annotation.task" in df.columns:
            sample_tasks.extend(df["annotation.task"].dropna().unique().tolist()[:2])
        if "observation.state" in df.columns and len(df):
            sample_states.append(df["observation.state"].iloc[0])

    sample_dir = report_path.parent / "sample_frames"
    sample_dir.mkdir(exist_ok=True)
    first_ep_videos = sorted((dataset_dir / "videos").rglob("episode_000000.mp4"))
    sampled = []
    for mp4 in first_ep_videos:
        cam = mp4.parent.name
        cap = cv2.VideoCapture(str(mp4))
        ok, frame = cap.read()
        cap.release()
        if ok:
            out_png = sample_dir / f"{cam}_first_frame.png"
            cv2.imwrite(str(out_png), frame)
            sampled.append((cam, out_png.name, frame.shape))

    stats = {}
    stats_path = dataset_dir / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            raw_stats = json.load(f)
        for k, v in raw_stats.items():
            mean = v.get("mean", [])
            stats[k] = {
                "dim": len(mean) if isinstance(mean, list) else 1,
                "mean_first3": mean[:3] if isinstance(mean, list) else mean,
                "q01_first3": v.get("q01", [])[:3] if isinstance(v.get("q01"), list) else None,
                "q99_first3": v.get("q99", [])[:3] if isinstance(v.get("q99"), list) else None,
            }
    rel_stats_present = (dataset_dir / "meta" / "relative_stats_dreamzero.json").exists()

    lines = []
    lines.append(f"# YAM prep report — `{dataset_dir.name}`\n")
    lines.append(f"**Path on Modal volume:** `{dataset_dir}`\n")
    lines.append("## Summary\n")
    lines.append(f"- Episodes: **{info.get('total_episodes', '?')}**")
    lines.append(f"- Frames:   **{info.get('total_frames', '?')}**  "
                 f"(≈ {info.get('total_frames', 0)/30/60:.1f} min at 30 fps)")
    lines.append(f"- FPS:      {info.get('fps', '?')}")
    lines.append(f"- First-5-episode lengths: {ep_lens}")
    lines.append(f"- Embodiment: `{embodiment}`")
    lines.append(f"- Relative-action stats present: **{rel_stats_present}**\n")

    lines.append("## Modality (sliced state/action keys)\n")
    if modality:
        for kind in ["state", "action", "video", "annotation"]:
            lines.append(f"**{kind}:**")
            for name, spec in modality.get(kind, {}).items():
                lines.append(f"  - `{kind}.{name}` ← {spec}")
            lines.append("")
    else:
        lines.append("_(modality.json not present yet — convert_lerobot_to_gear failed?)_\n")

    lines.append("## Sample task strings\n")
    if sample_tasks:
        for t in dict.fromkeys(sample_tasks):
            lines.append(f"  - {t!r}")
    else:
        lines.append("  _(no annotation.task column found in first 5 episodes!)_")
    lines.append("")

    lines.append("## Sample state[0] from first 5 episodes (14-D)\n")
    for i, st in enumerate(sample_states):
        lines.append(f"  ep{i}: {list(map(float, st))}")
    lines.append("")

    lines.append("## Stats summary (state + action keys)\n")
    for k, s in stats.items():
        lines.append(f"  - `{k}` dim={s['dim']}, mean[:3]={s['mean_first3']}, "
                     f"q01[:3]={s['q01_first3']}, q99[:3]={s['q99_first3']}")
    lines.append("")

    lines.append("## Sample camera frames\n")
    for cam, fname, shape in sampled:
        lines.append(f"  - `{cam}` first frame `{fname}`, shape={shape}")
    lines.append("")

    lines.append("## Pre-fine-tune checklist\n")
    has_state = modality.get("state", {})
    has_video = modality.get("video", {})
    checklist = [
        ("episodes > 0", info.get("total_episodes", 0) > 0),
        ("modality.state has left_joint_pos", "left_joint_pos" in has_state),
        ("modality.state has left_gripper_pos", "left_gripper_pos" in has_state),
        ("modality.state has right_joint_pos", "right_joint_pos" in has_state),
        ("modality.state has right_gripper_pos", "right_gripper_pos" in has_state),
        ("modality.action mirrors state keys",
         set(has_state.keys()) == set(modality.get("action", {}).keys())),
        ("video keys end in _camera-images-rgb",
         all(k.endswith("_camera-images-rgb") for k in has_video)),
        ("annotation.task in modality", "task" in modality.get("annotation", {})),
        ("embodiment_tag == 'yam'", embodiment.get("embodiment_tag") == "yam"),
        ("relative_stats_dreamzero.json exists", rel_stats_present),
        ("sample frame per camera saved", len(sampled) >= 3),
    ]
    for desc, ok in checklist:
        lines.append(f"  - [{'x' if ok else ' '}] {desc}")

    report_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Modal function
# ---------------------------------------------------------------------------

@app.function(
    timeout=60 * 60 * 6,
    cpu=8.0,
    memory=32 * 1024,
    volumes={"/data": data_volume},
)
def prepare_impl(
    hf_repos: List[str],
    tag: str,
    force: bool = False,
) -> str:
    """Convert v3 Ai2 datasets to v2 DreamZero layout and emit a prep report."""
    raw_root = Path("/data/raw")
    raw_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage 1/5: download {len(hf_repos)} dataset(s) ===", flush=True)
    src_roots: list[Path] = []
    for repo in hf_repos:
        local = raw_root / repo.replace("/", "__")
        if local.exists() and any(local.iterdir()) and not force:
            print(f"  (cached) {repo}", flush=True)
        else:
            print(f"  → {repo}", flush=True)
            subprocess.check_call([
                "hf", "download", repo, "--repo-type", "dataset",
                "--local-dir", str(local),
            ])
        src_roots.append(local)

    out_root = Path(f"/data/prepared/{tag}")
    if force and out_root.exists():
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage 2/5: v3 → v2 split (parquets + ffmpeg video cuts) ===", flush=True)
    ep_offset = 0
    all_episodes: list[dict] = []
    merged_ep_annotations: dict[int, str] = {}
    info_v3 = None
    cam_keys: list[str] = []
    for src in src_roots:
        print(f"\n  converting {src.name} (offset={ep_offset}) ...", flush=True)
        n, episodes, info_v3_this, cam_keys, ep_annotations = _convert_v3_to_v2(
            src, out_root, ep_offset=ep_offset, max_workers=8,
        )
        # Re-key annotations to GLOBAL episode_index (with offset applied).
        for raw_idx, text in ep_annotations.items():
            merged_ep_annotations[int(raw_idx) + ep_offset] = text
        all_episodes.extend(episodes)
        ep_offset += info_v3_this.get("total_episodes", n)
        if info_v3 is None:
            info_v3 = info_v3_this
        print(f"  → {n} episodes written.", flush=True)
    print(f"\n  total episodes: {len(all_episodes)}", flush=True)

    print(f"\n=== Stage 3/5: write v2 meta files ===", flush=True)
    _write_v2_meta(out_root, info_v3, all_episodes, merged_ep_annotations, cam_keys)
    print(f"  meta/{{info.json,tasks.jsonl,episodes.jsonl}} written", flush=True)

    print(f"\n=== Stage 4/5: convert_lerobot_to_gear.py (stats + modality + relative_stats) ===",
          flush=True)
    cmd = [
        "python", "/opt/dreamzero/scripts/data/convert_lerobot_to_gear.py",
        "--dataset-path", str(out_root),
        "--embodiment-tag", "yam",
        "--state-keys",
        '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],'
        '"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}',
        "--action-keys",
        '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],'
        '"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}',
        "--relative-action-keys", "left_joint_pos", "left_gripper_pos",
                                  "right_joint_pos", "right_gripper_pos",
        "--task-key", "annotation.task",
        "--action-horizon", "24",
        "--force",
    ]
    subprocess.check_call(cmd)

    # Patch modality.json to use the _camera-images-rgb-suffixed video keys.
    print(f"  fixing up modality.json video keys ...", flush=True)
    mp = out_root / "meta" / "modality.json"
    with open(mp) as f:
        mj = json.load(f)
    new_video = {}
    for short, spec in mj.get("video", {}).items():
        if short.endswith("_camera-images-rgb"):
            new_video[short] = spec
        else:
            new_video[f"{short}_camera-images-rgb"] = spec
    mj["video"] = new_video
    with open(mp, "w") as f:
        json.dump(mj, f, indent=4)

    print(f"\n=== Stage 5/5: write prep report ===", flush=True)
    report = out_root.parent / f"{tag}_prep_report.md"
    _write_prep_report(out_root, report)
    print(f"  wrote {report}", flush=True)
    data_volume.commit()
    return str(out_root)


@app.local_entrypoint()
def prepare(
    hf_repo: str = "",
    hf_repos: str = "",
    tag: str = "yam_box_smoke",
    force: bool = False,
):
    if hf_repos:
        repos = [r.strip() for r in hf_repos.split(",") if r.strip()]
    elif hf_repo:
        repos = [hf_repo]
    else:
        sys.exit(
            "Pass --hf-repo allenai/01122025-box-01  (single)\n"
            "or    --hf-repos allenai/01122025-box-01,allenai/01122025-box-02"
        )
    if not DREAMZERO_DIR.is_dir():
        sys.exit(f"DreamZero repo missing at {DREAMZERO_DIR}; clone it first.")
    out = prepare_impl.remote(repos, tag, force)
    print(f"\nPrep complete. Volume path: {out}")
    print(f"Pull artifacts:")
    print(f"  ./scripts/inspect_prep.sh {tag}")
