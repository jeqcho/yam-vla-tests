# bimanual_easy_bench_4 — corpus-aligned 4-task bimanual suite

A short, diagnostic eval where **every task is derived bottom-up from
the BimanualYAM training-corpus clusters**. Designed to produce
*pass rates the policy can actually achieve*, by matching the
phrasings and primitives the policy was most heavily trained on.

This is the third major revision of the suite. The first two
(`bench_8`, then `bench_5/4` after pruning) were designed top-down
from ALOHA-canonical primitives — fold, pour, pull-apart, etc. — and
produced near-zero pass rates because those primitives don't appear
in molmoact2-BimanualYAM's training mix. This revision flips the
methodology: we clustered the actual training annotations and chose
one task per top training cluster.

## Quickstart

```bash
# 1. Bring up your inference server (Terminal A)
./scripts/run_server.sh molmoact2     # or pi05 / gr00t-n17

# 2. Run the eval (Terminal B). Samples = attempts per task.
uv run scripts/run_eval.py --policy molmoact2 --eval bimanual_easy_bench_4 \
    --samples 5
```

## Operator flow (one attempt)

Per iteration, the harness drives you through five stages. **→ (right
arrow) is the universal "advance" key.** Enter works the same.

```
1. BANNER       shows TASK, iteration N of X, the exact prompt
2. ROLLOUT      arms execute the prompt; → ends rollout early
3. ARM RESET    arms ramp back to canonical training-mean ready pose
4. SCORE        s=success  f=failure  u=unclear  r=redo  Enter=skip
5. RESET WINDOW countdown (default 30 s); → advances early
```

## The four tasks

| # | Task | Training cluster | Instruction |
|---|---|---|---|
| 1 | arrange_cubes_line | Block arrangement (36%) | *"Pick up the four orange cubes and arrange them in a horizontal line."* |
| 2 | marker_apple_into_box | Stationery box-loading (17%) | *"Place the red marker and the apple into the gray box."* |
| 3 | cube_into_gray_box | Held-receptacle variant (3+4) | *"Pick up the gray box with your left arm and place the orange cube into the box with your right arm."* |
| 4 | plug_socket_into_outlet | Charging workflow (9%) | *"Grasp the black socket, then plug it into the white outlet."* |

## How the tasks were chosen

We embedded all 1,387 unique annotations from 90 BimanualYAM
training datasets (`allenai/<date>-<category>-<NN>` on HuggingFace)
using `sentence-transformers/all-MiniLM-L6-v2` and k-means clustered
to k=5. The clusters that emerged:

| Cluster | % | Theme | Centroid example |
|---|---|---|---|
| 0 | 9%  | Charging workflow | *"Grasp the charger, plug it into the phone, then switch on the socket."* |
| 1 | 36% | Block arrangement | *"Pick up blocks and arrange them in a horizontal line."* |
| 2 | 17% | Block spelling    | *"Move blocks to spell 'AI2' by picking up and placing them in order."* |
| 3 | 17% | Stationery box-loading | *"Place the marker, utility knife, and tape into the cardboard box, then close the box."* |
| 4 | 22% | Snack box-loading | *"Place yellow, red, green, and blue snack packets into the black box, then close the lid."* |

Cluster 2 (block spelling) requires letter blocks not in our
inventory, so it's omitted. The remaining four tasks each emulate
one of the other four clusters.

## Expected pass rates

These are *predictions* based on corpus alignment, not measurements:

| Task | Expected pass rate (10 attempts) |
|---|---|
| arrange_cubes_line | 50–80% |
| marker_apple_into_box | 60–80% |
| cube_into_gray_box | 50–75% |
| plug_socket_into_outlet | 40–70% |

If actual numbers come in much lower, the most likely explanation
is *physical setup mismatch* (e.g. objects too far apart, wrong
camera angles, gripper geometry doesn't match what it grasps in
training), not language-OOD-ness. If actual numbers are much
higher, the model's generalization is stronger than the cluster
analysis suggests.

## Suite history

This is the third revision of the suite. The full evolution:

**v1 — `bimanual_easy_bench_8`** (2026-05-26): 8 ALOHA-canonical
primitives (fold, pull-apart, hand-off, lift, pour, tap, uncap,
rotate). Goal was primitive diversity. Result: bag_fold = 0/10
on initial run; remaining tasks unverified but likely similar.

**v2 — `bimanual_easy_bench_5` then `_4`** (this same file's prior
revision): dropped bag_fold (verb 0/591), cube_handoff (pass 0/591),
allen_wrench_turn (rotation absent from corpus), pour_cubes (pour
0/591). Replaced velcro_pull with cube_into_gray_box (a held-
receptacle variant that matches the dominant Place-into template).
Result: cube_into_gray_box predicted to work, remaining 3 still
OOD on verbs (lift, tap, pull) and absent from corpus structure.

**v3 — this file** (current): scrapped the OOD-heavy approach.
K-means clustered the corpus into 5 training task-families.
Picked 4 tasks each derived from a top training cluster.

The history is preserved in git: any commit before this rewrite
shows the prior task set.

## Sources

- AllenAI BimanualYAM training corpus annotations, 90 datasets,
  2815 raw / 1387 unique annotations, fetched from
  `huggingface.co/datasets/allenai/<slug>/resolve/main/meta/
  tasks_annotated.parquet`.
- Sentence embeddings: `sentence-transformers/all-MiniLM-L6-v2`.
- Clustering: scikit-learn `KMeans(n_clusters=5, random_state=42)`.
- Cluster analysis cached at `/tmp/yam_corpus/annotations.json`.
