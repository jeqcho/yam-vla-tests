"""Auto-populate report/images/ with IKEA manual screenshots + Rerun frames.

Two sources:

  1. IKEA manual PDFs from short-list.md -> page 1 -> PNG via pdftoppm
     (saved as images/<slug>/manual.png)

  2. Rerun .rrd recordings under eval-yam/logs/rrd/ -> one cam/top frame
     per task, matched to the journal entry's wall-clock timestamp
     (saved as images/<slug>/rerun.png)

After running this, re-run build.py to bake the populated images into
report.html.

Run via the i2rt venv (needs rerun-sdk + PIL):
    /home/andon/yam-tests/i2rt/.venv/bin/python extract_images.py
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SHORTLIST = _REPO / "reference" / "robotics-task-horizon" / \
    "experiments" / "7-ikea-full-catalog" / "reports" / "short-list.md"
_JOURNAL = _REPO.parent / "molmoact2-setup" / "journal.md"
_RRD_DIR = _REPO.parent / "eval-yam" / "logs" / "rrd"
_IMAGES = _HERE / "images"


def _slug(swedish: str) -> str:
    # Drop secondary product names (e.g. "GREJIG / BAGGMUCK" -> "GREJIG")
    # so we stay aligned with build.py's TASKS-driven slug convention.
    primary = swedish.split(" / ")[0].strip()
    n = unicodedata.normalize("NFKD", primary)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[/\s]+", "_", n)
    n = re.sub(r"[^a-z0-9_]", "", n)
    return n


# ---------------------------------------------------------------------------
# Manual PDF -> PNG
# ---------------------------------------------------------------------------
def parse_shortlist_urls() -> dict[str, dict[str, str]]:
    """Return {swedish_name: {"pdf": url, "page": url}} from short-list.md.

    Each row has both a [page] link (IKEA product page HTML) and a [pdf]
    link (assembly instructions). We need both: PDF for the manual image,
    page for scraping the product photo via <meta og:image>.
    """
    text = _SHORTLIST.read_text(encoding="utf-8")
    out: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("|") or "[pdf]" not in line:
            continue
        m_pdf = re.search(r"\[pdf\]\((https://[^)]+\.pdf)\)", line)
        m_page = re.search(r"\[page\]\((https://[^)]+)\)", line)
        if not m_pdf or not m_page: continue
        # Swedish name = all-caps word(s) at the start of the Product column.
        m_name = re.search(r"\| ([A-ZÅÄÖ /]+?)(?: [A-Z]?[a-z])", line)
        if not m_name: continue
        out[m_name.group(1).strip()] = {
            "pdf": m_pdf.group(1),
            "page": m_page.group(1),
        }
    return out


# ---------------------------------------------------------------------------
# Product photo (og:image scraped from IKEA's product page)
# ---------------------------------------------------------------------------
def download_product_photo(swedish: str, page_url: str, slug: str) -> bool:
    """Fetch the IKEA product page, extract <meta property='og:image'>,
    download and save to images/<slug>/product.png."""
    out_dir = _IMAGES / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "product.png"
    if out.exists():
        print(f"  [product] {slug}: already exists, skipping")
        return True
    try:
        req = urllib.request.Request(
            page_url,
            headers={"User-Agent": "Mozilla/5.0 (ikea-10-report)"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [product] {slug}: page fetch FAILED: {e}")
        return False
    # IKEA pages: <meta property="og:image" content="https://....jpg"/>
    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html)
    if not m:
        m = re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']', html)
    if not m:
        print(f"  [product] {slug}: no og:image found")
        return False
    img_url = m.group(1)
    try:
        req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
    except Exception as e:
        print(f"  [product] {slug}: img fetch FAILED ({img_url}): {e}")
        return False
    out.write_bytes(data)
    print(f"  [product] {slug}: wrote {out} ({len(data)//1024} KB from {img_url})")
    return True


# Per-furniture autocrop tuning. Default works for most IKEA manuals;
# override here if a specific PDF needs a tighter / looser threshold or
# a forced fractional crop. Keys are slugs.
#   threshold: brightness (0-255) below which a pixel counts as "content"
#   margin:    extra padding (px) added around the detected bbox
#   force_crop: (left, top, right, bottom) in 0..1 fractions, applied
#               AFTER autocrop. None = no extra crop.
_MANUAL_CROP_TUNING: dict[str, dict] = {
    # IKEA's diagram pages are mostly black/grey line art on white;
    # threshold 240 picks up light grey shading too.
    "_default":   {"threshold": 240, "margin": 18, "force_crop": None},
    # FISKBO page has a wide IKEA logo banner at top -- standard crop ok
    "fiskbo":     {"threshold": 240, "margin": 12, "force_crop": None},
    # KROKFJORDEN PDF is unusually tall (0.36 aspect) -- crop top/bottom
    # white aggressively so the steps fill more of the cell
    "krokfjorden":{"threshold": 235, "margin": 8,  "force_crop": None},
    # SKOGSRÖR is super-wide (4.11) with thin diagram strip; tight crop
    "skogsror":   {"threshold": 245, "margin": 4,  "force_crop": None},
    # GREJIG render has dotted/grey overlays we don't want to clip
    "grejig":     {"threshold": 230, "margin": 14, "force_crop": None},
}


def _autocrop_whitespace(png_path: Path, slug: str) -> bool:
    """Crop the white margins off a manual.png so it fills the report
    cell tightly. Per-furniture overrides in _MANUAL_CROP_TUNING."""
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return False
    cfg = {**_MANUAL_CROP_TUNING["_default"],
           **_MANUAL_CROP_TUNING.get(slug, {})}
    threshold = cfg["threshold"]
    margin = cfg["margin"]
    force = cfg["force_crop"]

    img = Image.open(png_path).convert("RGB")
    W0, H0 = img.size
    # Convert to grayscale, then invert so content is bright and bg is dark.
    # getbbox() on the inverted thresholded image returns the content bbox.
    gray = img.convert("L")
    # Pixels brighter than threshold count as "white" -> mask them out.
    mask = gray.point(lambda v: 0 if v >= threshold else 255)
    bbox = mask.getbbox()
    if bbox is None:
        print(f"  [crop] {slug}: nothing to crop (image all-white?)")
        return False
    l, t, r, b = bbox
    l = max(0, l - margin); t = max(0, t - margin)
    r = min(W0, r + margin); b = min(H0, b + margin)
    cropped = img.crop((l, t, r, b))
    if force is not None:
        fW, fH = cropped.size
        fl = int(fW * force[0]); ft = int(fH * force[1])
        fr = int(fW * (1 - force[2])); fb = int(fH * (1 - force[3]))
        cropped = cropped.crop((fl, ft, fr, fb))
    cropped.save(png_path)
    pct_w = 100 * cropped.size[0] / W0
    pct_h = 100 * cropped.size[1] / H0
    print(f"  [crop] {slug}: {W0}×{H0} → {cropped.size[0]}×{cropped.size[1]}  "
          f"({pct_w:.0f}% × {pct_h:.0f}%)")
    return True


def download_manual(swedish: str, pdf_url: str, slug: str,
                    overwrite: bool = False) -> bool:
    """Download PDF, render page 1 to manual_raw.png, then crop into
    manual.png per per-furniture _MANUAL_CROP_TUNING.

    The raw render is kept so tuning can be iterated without re-
    downloading. After editing _MANUAL_CROP_TUNING values, run with
    --recrop (no PDF download) to re-apply.
    """
    out_dir = _IMAGES / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_png = out_dir / "manual_raw.png"
    out_png = out_dir / "manual.png"
    have_raw = raw_png.exists()
    if have_raw and out_png.exists() and not overwrite:
        print(f"  [manual] {slug}: already exists, skipping")
        return True
    if not have_raw or overwrite:
        with tempfile.TemporaryDirectory() as td:
            pdf_path = Path(td) / "doc.pdf"
            try:
                print(f"  [manual] {slug}: downloading {pdf_url} ...", end=" ", flush=True)
                urllib.request.urlretrieve(pdf_url, pdf_path)
                print("ok")
            except Exception as e:
                print(f"FAIL: {e}")
                return False
            prefix = Path(td) / "out"
            try:
                subprocess.run(
                    ["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "1",
                     str(pdf_path), str(prefix)],
                    check=True, capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"  [manual] {slug}: pdftoppm FAILED: {e.stderr.decode()[:200]}")
                return False
            rendered = Path(td) / "out-1.png"
            if not rendered.exists():
                candidates = list(Path(td).glob("out-*.png"))
                if not candidates:
                    print(f"  [manual] {slug}: pdftoppm produced no PNG")
                    return False
                rendered = candidates[0]
            shutil.copy(rendered, raw_png)
            print(f"  [manual] {slug}: raw -> {raw_png}")
    # Always crop from the cached raw render so re-runs reflect tuning.
    shutil.copy(raw_png, out_png)
    _autocrop_whitespace(out_png, slug)
    return True


def recrop_existing_manuals() -> None:
    """Re-apply autocrop tuning to every furniture with a cached raw
    render. Useful after editing _MANUAL_CROP_TUNING to preview new
    margins without re-downloading PDFs.

    Recovers manual.png from manual_raw.png; falls back to in-place
    re-crop of manual.png (idempotent, no-op if margins already gone)
    for furnitures that don't have a cached raw.
    """
    for d in _IMAGES.iterdir():
        if not d.is_dir(): continue
        slug = d.name
        raw = d / "manual_raw.png"
        cur = d / "manual.png"
        if raw.exists():
            shutil.copy(raw, cur)
            _autocrop_whitespace(cur, slug)
        elif cur.exists():
            print(f"  [crop] {slug}: no manual_raw.png cached -- "
                  f"re-cropping in place (may be idempotent)")
            _autocrop_whitespace(cur, slug)


# ---------------------------------------------------------------------------
# Rerun .rrd -> one cam/top PNG per task, matched by journal timestamp
# ---------------------------------------------------------------------------
JOURNAL_TASK_RE = re.compile(
    r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+--\s+\w+\s+\(repl attempt #\d+\)"
    r"(?:.*?\n)*?"
    r"\*\*Notes\*\*:\s+\[ikea-10 task \d+/\d+: ([^\]]+)\]",
    re.M,
)


def parse_journal_task_times() -> dict[str, list[dt.datetime]]:
    """Return {swedish: [datetime, ...]} for ikea-10 attempts in the journal.

    The Notes string looks like:
      [ikea-10 task 5/10: LACK / side table] [prompt=...] [policy=molmoact2] ...
    We extract the SWEDISH portion (text before ' / ' in the task tag).
    """
    text = _JOURNAL.read_text(encoding="utf-8")
    out: dict[str, list[dt.datetime]] = {}
    for ts_str, tag in JOURNAL_TASK_RE.findall(text):
        # tag looks like "LACK / side table" or "GREJIG / BAGGMUCK / shoe rack"
        # Swedish part can itself contain ' / ' (e.g. GREJIG / BAGGMUCK), so
        # we keep the first chunk that's all-caps.
        parts = [p.strip() for p in tag.split(" / ")]
        if not parts: continue
        swedish_parts = []
        for p in parts:
            if p == p.upper() and any(ch.isalpha() for ch in p):
                swedish_parts.append(p)
            else:
                break
        swedish = " / ".join(swedish_parts)
        if not swedish: continue
        try:
            ts = dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        out.setdefault(swedish, []).append(ts)
    return out


# CSV results from eval_ikea_tasks.py contain the canonical attempt list
# for tasks that were attempted outside of the REPL flow (which is the
# only source the journal sees). Merging both sources gives us full
# coverage of all 10 ikea tasks.
_CSV_RESULTS = _REPO / "eval" / "results"


def parse_csv_task_times() -> dict[str, list[dt.datetime]]:
    """Return {slug: [datetime, ...]} for ikea-10 attempts logged via
    eval_ikea_tasks.py CSVs. Walks every results/<policy>/*.csv file.

    Note: CSV swedish column may be the full name as typed in TASKS
    (e.g. "GREJIG / BAGGMUCK"). We key by *slug* here, not raw swedish,
    so it pairs cleanly with the journal parser whose output we
    normalize on the consumer side.
    """
    import csv as _csv
    out: dict[str, list[dt.datetime]] = {}
    if not _CSV_RESULTS.exists():
        return out
    for csv_path in _CSV_RESULTS.rglob("*.csv"):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in _csv.DictReader(f):
                    try:
                        ts = dt.datetime.strptime(
                            row["timestamp"], "%Y-%m-%d %H:%M:%S"
                        )
                    except (KeyError, ValueError):
                        continue
                    swedish = row.get("swedish", "").strip()
                    if not swedish: continue
                    out.setdefault(_slug(swedish), []).append(ts)
        except Exception as e:
            print(f"  [csv] {csv_path.name}: skip ({e})")
    return out


def discover_rrds() -> list[tuple[Path, dt.datetime, dt.datetime]]:
    """Return [(path, start_wall, end_wall)] for all relevant .rrds.

    Time range is read from the rrd's *internal* cam/top log_time
    (first and last chunk). This is more reliable than filename or
    mtime — newly saved rrds from the viewer have a recording-time
    prefix in the filename but mtime reflects when you hit Save, not
    when the data was logged.

    Accepts both 'ikea10_eval_*' (eval_ikea_tasks.py with --rerun) and
    'yam_repl_*' (repl_yam.py with --rerun-save).
    """
    try:
        from rerun_sdk.rerun import bindings  # type: ignore
    except ImportError:
        return []
    out: list[tuple[Path, dt.datetime, dt.datetime]] = []
    for p in _RRD_DIR.glob("*.rrd"):
        try:
            rec = bindings.load_recording(str(p))
            app_id = rec.application_id()
        except Exception:
            continue
        if not (app_id.startswith("ikea10_eval_")
                or app_id.startswith("yam_repl_")):
            continue
        top_chunks = [c for c in rec.chunks() if c.entity_path == "/cam/top"]
        if not top_chunks:
            continue
        try:
            ft_ns = top_chunks[0].to_record_batch().column("log_time")[0].value
            lt_ns = top_chunks[-1].to_record_batch().column("log_time")[-1].value
            start = dt.datetime.fromtimestamp(ft_ns / 1e9)
            end = dt.datetime.fromtimestamp(lt_ns / 1e9)
        except Exception:
            continue
        out.append((p, start, end))
    out.sort(key=lambda x: x[1])
    return out


def extract_cam_top(rrd_path: Path, target_ts: dt.datetime,
                    rrd_start: dt.datetime, rrd_end: dt.datetime,
                    out_png: Path) -> bool:
    """Extract the cam/top frame nearest to target_ts wall-clock time.

    Since rrd-internal log_time has corruption issues on some entries,
    we map the target timestamp to a proportional position within the
    rrd's chunk sequence: offset_frac = (target - start) / (end - start).

    The journal timestamp marks the START of an attempt; we add a
    small bias (+15s) so we sample mid-attempt rather than at the
    static "starting pose" frame.
    """
    try:
        from rerun_sdk.rerun import bindings  # type: ignore
    except ImportError:
        print(f"  [rerun] FAIL: rerun-sdk not installed")
        return False
    try:
        import numpy as np
        from PIL import Image
    except ImportError as e:
        print(f"  [rerun] FAIL: {e}")
        return False

    rec = bindings.load_recording(str(rrd_path))
    top_chunks = [c for c in rec.chunks() if c.entity_path == "/cam/top"]
    if not top_chunks:
        print(f"  [rerun] no cam/top chunks in {rrd_path.name}")
        return False

    # +15s bias → mid-attempt frame, not the static start pose.
    biased_ts = target_ts + dt.timedelta(seconds=15)
    total_sec = max(1.0, (rrd_end - rrd_start).total_seconds())
    offset_sec = max(0.0, (biased_ts - rrd_start).total_seconds())
    frac = min(1.0, offset_sec / total_sec)
    chunk_idx = min(len(top_chunks) - 1, int(frac * len(top_chunks)))

    rb = top_chunks[chunk_idx].to_record_batch()
    row_i = 0
    buf_outer = rb.column("Image:buffer")[row_i].as_py()
    if not buf_outer:
        print(f"  [rerun] empty buffer in {rrd_path.name}")
        return False
    buf_bytes = bytes(buf_outer[0]) if isinstance(buf_outer[0], (list, bytes)) else bytes(buf_outer)
    fmt_outer = rb.column("Image:format")[row_i].as_py()
    fmt = fmt_outer[0] if isinstance(fmt_outer, list) else fmt_outer
    W = fmt["width"]
    H = fmt["height"]
    expected = W * H * 3
    if len(buf_bytes) != expected:
        print(f"  [rerun] size mismatch: got {len(buf_bytes)} expected {expected}")
        return False
    img = np.frombuffer(buf_bytes, dtype=np.uint8).reshape(H, W, 3)
    Image.fromarray(img, mode="RGB").save(out_png)
    print(f"  [rerun] wrote {out_png} "
          f"(chunk {chunk_idx+1}/{len(top_chunks)} at "
          f"t+{offset_sec:.0f}s of {rrd_path.name})")
    return True


def find_rrd_for_task(target_ts: dt.datetime,
                      rrds: list[tuple[Path, dt.datetime, dt.datetime]],
                      tolerance_s: int = 30,
                      ) -> tuple[Path, dt.datetime, dt.datetime] | None:
    """Pick the .rrd whose wall-clock recording range covers target_ts.

    A `tolerance_s` window on both sides handles the case where the
    CSV timestamp logs the *end* of an attempt and the rrd's last
    frame stops a few seconds earlier (or vice versa). Without this,
    PATRULL/SKOGSRÖR/VÅRSYREN attempts (whose CSV ts is 1-8s past
    rrd end) get no match.

    Prefer rrds that strictly contain the timestamp; fall back to the
    nearest tolerant match.
    """
    strict = [(p, s, e) for (p, s, e) in rrds if s <= target_ts <= e]
    if strict:
        return strict[0]
    tol = dt.timedelta(seconds=tolerance_s)
    near = [(p, s, e) for (p, s, e) in rrds if s - tol <= target_ts <= e + tol]
    if not near:
        return None
    # closest by distance from rrd center
    return min(near, key=lambda x: abs((target_ts - (x[1] + (x[2] - x[1])/2)).total_seconds()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # --recrop: re-apply per-furniture _MANUAL_CROP_TUNING to cached
    # manual_raw.png files without re-downloading PDFs. Fast iteration
    # for tuning each furniture's margins.
    if "--recrop" in sys.argv[1:]:
        print("RECROP mode: re-applying _MANUAL_CROP_TUNING to cached raws\n")
        recrop_existing_manuals()
        return

    if not _SHORTLIST.exists():
        print(f"missing: {_SHORTLIST}", file=sys.stderr); sys.exit(2)
    if not _JOURNAL.exists():
        print(f"missing: {_JOURNAL}", file=sys.stderr); sys.exit(2)

    pdf_urls = parse_shortlist_urls()
    journal_times = parse_journal_task_times()
    csv_times = parse_csv_task_times()
    rrds = discover_rrds()

    # Merge journal + csv attempt timestamps keyed by slug.
    by_slug: dict[str, list[dt.datetime]] = {}
    for swedish, times in journal_times.items():
        by_slug.setdefault(_slug(swedish), []).extend(times)
    for s, times in csv_times.items():
        by_slug.setdefault(s, []).extend(times)

    print(f"shortlist URLs:     {len(pdf_urls)}")
    print(f"journal attempts:   {sum(len(v) for v in journal_times.values())}")
    print(f"csv attempts:       {sum(len(v) for v in csv_times.values())}")
    print(f"merged tasks:       {len(by_slug)} "
          f"({sum(len(v) for v in by_slug.values())} attempts)")
    print(f"rrd recordings:     {len(rrds)}")
    print()

    for swedish, urls in pdf_urls.items():
        s = _slug(swedish)
        print(f"=== {swedish} ({s}) ===")
        # Manual PDF → page-1 PNG
        download_manual(swedish, urls["pdf"], s)
        # IKEA product photo via og:image
        download_product_photo(swedish, urls["page"], s)
        # Rerun: try each timestamp (journal + csv merged), latest first.
        # Later attempts usually capture the arms in motion vs. start pose.
        timestamps = sorted(set(by_slug.get(s, [])), reverse=True)
        if not timestamps:
            print(f"  [rerun] no journal/csv entries for {swedish}")
            continue
        out_png = _IMAGES / s / "rerun.png"
        if out_png.exists():
            print(f"  [rerun] {s}: already exists, skipping")
            continue
        matched = False
        for target in timestamps:
            hit = find_rrd_for_task(target, rrds)
            if hit is None:
                continue
            rrd, start, end = hit
            if extract_cam_top(rrd, target, start, end, out_png):
                matched = True
                break
        if not matched:
            print(f"  [rerun] no rrd covers {swedish} "
                  f"({len(timestamps)} attempt(s) tried) -- placeholder")


if __name__ == "__main__":
    main()
