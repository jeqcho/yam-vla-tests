"""Generate report.html for the IKEA-10 VLA results.

Pulls each task's swedish/english/instruction from
``../eval/eval_ikea_tasks.py``'s TASKS list so the report stays in
sync with the harness config -- just re-run this script after any
TASKS edit.

Per-task page contains:
  - Swedish SKU as title
  - English description as subtitle
  - Full model-facing instruction (italic, quote-bar)
  - Two image slots side-by-side: IKEA manual screenshot + Rerun screengrab
  - Bar chart: success rate for MolmoAct2 / GR00T-N1.7 / Pi-0.5+YAM

Image slots expect files at:
  images/<slug>/manual.png   (IKEA assembly PDF screenshot)
  images/<slug>/rerun.png    (Rerun viewer screengrab)
where <slug> is the lowercase ASCII swedish name (diacritics stripped,
spaces and slashes -> underscore). build.py creates the dirs empty;
drop the PNGs in, then open report.html in Chrome/Safari/Firefox and
File -> Print -> Save as PDF.

Today this hardcodes a 0/3 result for every (policy, task) because
all 3 VLAs failed everything. To pivot to per-policy CSV-driven later,
edit ``_zero_results`` to load eval/results/<policy>/results_*.csv.

Run:
    python3 build.py
"""
from __future__ import annotations

import ast
import os
import re
import unicodedata

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASKS_FILE = os.path.normpath(os.path.join(_HERE, "..", "eval", "eval_ikea_tasks.py"))


# ---------------------------------------------------------------------------
# Static AST walker: pull the TASKS list literal out of the source without
# importing the module (which would drag in i2rt + openpi etc). Handles the
# narrow subset of Python literals we actually use: dict, list, str (incl.
# implicit-concat strings, int).
# ---------------------------------------------------------------------------

def _walk_value(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_walk_value(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_walk_value(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_walk_value(k): _walk_value(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # adjacent-string concatenation across lines, e.g.:
        #   "foo " "bar"   -> parser folds to one Constant
        #   "foo " + "bar" -> ast.BinOp(Add)
        return _walk_value(node.left) + _walk_value(node.right)
    raise ValueError(f"unsupported value node: {ast.dump(node)[:200]}")


def _load_tasks() -> list[dict]:
    with open(_TASKS_FILE, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src)
    for node in ast.iter_child_nodes(tree):
        targets = []
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                targets = [node.target]
            value_node = node.value
        elif isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
            value_node = node.value
        else:
            continue
        for tgt in targets:
            if tgt.id == "TASKS" and value_node is not None:
                return _walk_value(value_node)
    raise RuntimeError(f"TASKS not found in {_TASKS_FILE}")


def slug(swedish: str) -> str:
    """ASCII-safe lowercase slug for filesystem paths."""
    n = unicodedata.normalize("NFKD", swedish)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[/\s]+", "_", n)
    n = re.sub(r"[^a-z0-9_]", "", n)
    return n


# ---------------------------------------------------------------------------
# Per-policy per-task results. Today: hardcoded all-zeros because the actual
# eval run had MolmoAct2 fail every atomic. Replace with a CSV loader once
# you have GR00T and Pi-0.5 runs too.
# ---------------------------------------------------------------------------
def _zero_results(task: dict) -> list[tuple[str, int, int]]:
    return [
        ("MolmoAct2",     0, 3),
        ("GR00T-N1.7",    0, 3),
        ("Pi-0.5 + YAM",  0, 3),
    ]


PAGE_TMPL = """\
  <section class="page" id="{slug}">
    <div class="masthead">
      <span class="masthead-page">{page_num} / 10</span>
    </div>
    <div class="page-body">
      <div class="grid">
        <div class="cell cell-header">
          <h1 class="swedish">{swedish}</h1>
          <p class="english">{english}</p>
        </div>
        <figure class="cell cell-product">
          <div class="cell-eyebrow">Product</div>
          <div class="cell-frame">{product_slot}</div>
        </figure>
        <figure class="cell cell-manual">
          <div class="cell-eyebrow">Manual</div>
          <div class="cell-frame">{manual_slot}</div>
        </figure>
        <div class="cell cell-atomic">
          <div class="cell-eyebrow">Tasks</div>
          <ol class="atomic-list">
{atomic_items}
          </ol>
        </div>
        <figure class="cell cell-rerun">
          <div class="cell-eyebrow">Real-world rollouts</div>
          <div class="cell-frame">{rerun_slot}</div>
        </figure>
        <div class="cell cell-bars">
          <div class="cell-eyebrow">Success rates&nbsp;·&nbsp;N=3</div>
          <div class="chart">
{bars}
          </div>
        </div>
      </div>
    </div>
  </section>
"""

ATOM_TMPL = """\
            <li class="atom">
              <span class="atom-mark"></span>
              <span class="atom-num">{n:02d}</span>
              <span class="atom-text">{text}</span>
            </li>
"""


def _image_slot(slug: str, kind: str, alt: str) -> str:
    """Return either an <img> if the file exists on disk OR a styled
    placeholder div if not. Re-run build.py after adding/removing images
    to swap between the two."""
    rel = f"images/{slug}/{kind}.png"
    abs_path = os.path.join(_HERE, rel)
    if os.path.exists(abs_path):
        return f'<img src="{rel}" alt="{alt}">'
    return (
        f'<div class="image-placeholder">'
        f'<span class="ph-icon">▢</span>'
        f'<span class="ph-text">{rel}</span>'
        f'</div>'
    )

BAR_TMPL = """\
            <div class="bar-row{fail_cls}">
              <div class="bar-col">
                <div class="bar-label-row">
                  <span class="bar-name">{name}</span>
                  <span class="bar-meta">{meta}</span>
                </div>
                <div class="bar-track">
                  <div class="bar-fill" style="width: {pct}%"></div>
                </div>
              </div>
              <div class="bar-value">
                <span class="bar-pct">{pct}<span class="pct-sign">%</span></span>
              </div>
            </div>
"""


HTML_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IKEA-10 · VLA Eval Report</title>
<!-- Fonts self-hosted under fonts/ to avoid CDN latency on flaky wifi.
     Regenerate with: see fonts/_google.css and the helper script. -->
<link href="fonts/fonts.css" rel="stylesheet">
<style>
  @page {{
    size: letter;
    margin: 0;
  }}

  :root {{
    --paper:        #ffffff;          /* IKEA-manual-white */
    --paper-soft:   #f5f5f5;          /* image card background */
    --ink:          #0d1117;
    --ink-muted:    #3a3a3a;
    --ink-soft:     #6a6a6a;
    --line:         #d6d6d6;
    --line-soft:    #e8e8e8;
    --rust:         #a23b2f;
    --rust-deep:    #7d2b22;
    --rust-soft:    #e8d4d1;
  }}

  * {{ box-sizing: border-box; }}

  html {{ background: #2a2723; }}  /* dark surround like a real PDF reader */
  body {{
    margin: 0;
    padding: 2rem 0;
    min-height: 100vh;
    background: #2a2723;
    color: var(--ink);
    font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif;
    font-weight: 400;
    font-feature-settings: "ss02", "cv11", "tnum";
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  .page {{
    width: 8.5in;
    height: 11in;
    padding: 0.45in 0.5in 0.4in;
    background: var(--paper);
    position: relative;
    display: none;            /* SCREEN: hide all by default; .active is shown */
    flex-direction: column;
    overflow: hidden;
    page-break-after: always;
    margin: 0 auto;
    box-shadow:
      0 30px 60px -20px rgba(0, 0, 0, 0.5),
      0 18px 36px -18px rgba(0, 0, 0, 0.4),
      0 0 0 0.5px rgba(0, 0, 0, 0.15);
    border-radius: 1px;
  }}
  .page.active {{ display: flex; }}   /* show the one selected by nav */
  .page:last-of-type {{ page-break-after: auto; }}

  /* ============================================================ */
  /* MASTHEAD  -- thin publication band at top of every page      */
  /* ============================================================ */
  .masthead {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 9px;
    font-weight: 400;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-muted);
    border-bottom: 0.5px solid var(--line);
    padding-bottom: 0.35rem;
    margin-bottom: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }}
  .masthead-page {{
    font-variant-numeric: tabular-nums;
    color: var(--ink-soft);
  }}

  .page-body {{
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }}

  /* ============================================================ */
  /* GRID  -- 2 columns × 3 rows, equal cells                     */
  /* ============================================================ */
  .grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    /* Header is text-only (title + english), so row 1 is compact;
       remaining height goes to product (spans 1-2), manual (spans
       2-3), and bottom row with rerun + bars. Slight asymmetry
       between product (1.6fr) and manual (2fr) reads as editorial
       stagger rather than imbalance. */
    grid-template-rows: 0.6fr 1fr 1fr 1fr;
    grid-template-areas:
      "header  product"
      "manual  product"
      "manual  atomic"
      "rerun   bars";
    gap: 0.45rem 0.7rem;
    flex: 1;
    min-height: 0;
  }}
  .cell-header  {{ grid-area: header; }}
  .cell-product {{ grid-area: product; }}
  .cell-manual  {{ grid-area: manual; }}
  .cell-atomic  {{ grid-area: atomic; }}
  .cell-rerun   {{ grid-area: rerun; }}
  .cell-bars    {{ grid-area: bars; }}
  .cell {{
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }}

  /* Eyebrow label that sits above each cell (mono small caps).
     Gives the report a strong editorial / catalog rhythm. */
  .cell-eyebrow {{
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 0.5px solid var(--line);
    margin-bottom: 0.5rem;
    font-family: "IBM Plex Mono", monospace;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 8.5px;
    line-height: 1;
  }}
  .eyebrow-num {{ color: var(--rust); font-weight: 500; }}
  .eyebrow-label {{ color: var(--ink-soft); }}

  /* Image cells: borderless, transparent. Image fills available space
     (object-fit:contain) — letterboxing matches paper white so it
     reads as natural margin rather than a framed empty cell. */
  .cell-frame {{
    flex: 1;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }}
  .cell-frame img,
  .cell-frame .image-placeholder {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }}
  .cell-frame .image-placeholder {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.45rem;
    color: var(--ink-soft);
    font-family: "IBM Plex Mono", monospace;
    background:
      repeating-linear-gradient(
        45deg,
        var(--paper), var(--paper) 8px,
        var(--paper-soft) 8px, var(--paper-soft) 16px
      );
  }}
  .cell-frame .image-placeholder .ph-icon {{
    font-size: 1.4rem;
    line-height: 1;
    color: var(--line);
  }}
  .cell-frame .image-placeholder .ph-text {{
    font-size: 8.5px;
    color: var(--ink-soft);
    letter-spacing: 0.04em;
    opacity: 0.7;
  }}

  /* ============================================================ */
  /* HEADER CELL  -- Swedish name + English + pull-quote          */
  /* ============================================================ */
  .cell-header {{
    /* Title + subtitle fill the compact header row cleanly. */
    justify-content: center;
    padding: 0.15rem 0 0.1rem;
  }}
  .swedish {{
    font-family: "Fraunces", "Cormorant Garamond", Georgia, serif;
    font-optical-sizing: auto;
    font-variation-settings: "SOFT" 100, "WONK" 0, "opsz" 144;
    font-size: 52px;
    font-weight: 400;
    letter-spacing: -0.022em;
    line-height: 0.9;
    margin: 0;
    color: var(--ink);
  }}
  .english {{
    font-family: "IBM Plex Sans", sans-serif;
    font-size: 11.5px;
    font-weight: 400;
    font-style: italic;
    color: var(--ink-soft);
    margin: 0.3rem 0 0 0;
    letter-spacing: 0.005em;
  }}

  /* ============================================================ */
  /* ATOMIC LIST cell                                              */
  /* ============================================================ */
  .atomic-list {{
    list-style: none;
    margin: 0;
    padding: 0;
    counter-reset: none;
    flex: 1;
    overflow: hidden;
  }}
  .atom {{
    display: grid;
    grid-template-columns: 1.2rem 1.6rem 1fr;
    align-items: baseline;
    gap: 0.45rem;
    padding: 0.42rem 0;
    border-bottom: 0.5px dotted var(--line-soft);
  }}
  .atom:last-child {{ border-bottom: none; }}
  .atom-mark {{
    width: 9px;
    height: 9px;
    border: 1px solid var(--ink-soft);
    border-radius: 1px;
    display: inline-block;
    justify-self: center;
    align-self: center;
  }}
  .atom-num {{
    font-family: "IBM Plex Mono", monospace;
    font-size: 9px;
    color: var(--rust);
    letter-spacing: 0.05em;
    font-weight: 500;
  }}
  .atom-text {{
    font-family: "Fraunces", Georgia, serif;
    font-variation-settings: "opsz" 18;
    font-size: 12px;
    font-weight: 400;
    line-height: 1.4;
    color: var(--ink);
  }}

  /* ============================================================ */
  /* OUTCOME (chart)                                               */
  /* ============================================================ */
  .chart {{
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    flex: 1;
    padding: 0.25rem 0;
  }}
  .bar-row {{
    display: grid;
    grid-template-columns: 1fr 4.8rem;
    align-items: center;
    gap: 0.7rem;
    padding: 0.4rem 0;
  }}
  .bar-label-row {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 0.2rem;
  }}
  .bar-name {{
    font-family: "Fraunces", Georgia, serif;
    font-variation-settings: "opsz" 24;
    font-size: 13px;
    font-weight: 500;
    color: var(--ink);
    line-height: 1;
  }}
  .bar-meta {{
    font-family: "IBM Plex Mono", monospace;
    font-size: 8px;
    color: var(--ink-soft);
    letter-spacing: 0.05em;
  }}

  .bar-track {{
    position: relative;
    height: 18px;
    background: transparent;
    border-top: 0.5px solid var(--line);
    border-bottom: 0.5px solid var(--line);
    overflow: visible;
  }}
  .bar-fill {{
    position: absolute;
    top: 0;
    left: 0;
    bottom: 0;
    width: 0%;
    background: var(--ink);
  }}
  .bar-track::before {{
    content: "";
    position: absolute;
    left: 0;
    top: -3px;
    bottom: -3px;
    width: 2px;
    background: var(--rust);
  }}
  .bar-zero,
  .bar-end {{
    position: absolute;
    bottom: -11px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 7px;
    color: var(--ink-soft);
    letter-spacing: 0.05em;
  }}
  .bar-zero {{ left: 0; }}
  .bar-end {{ right: 0; }}
  .bar-row.fail .bar-track {{
    background: repeating-linear-gradient(
      90deg,
      transparent, transparent 6px,
      var(--rust-soft) 6px, var(--rust-soft) 7px
    );
    opacity: 0.6;
  }}

  .bar-value {{
    text-align: right;
    line-height: 1;
  }}
  .bar-pct {{
    display: block;
    font-family: "Fraunces", Georgia, serif;
    font-variation-settings: "opsz" 144;
    font-size: 22px;
    font-weight: 400;
    color: var(--rust);
    line-height: 1;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
  }}
  .pct-sign {{
    font-size: 0.55em;
    color: var(--rust);
    margin-left: 0.05em;
    vertical-align: 0.3em;
  }}
  .bar-status {{
    display: block;
    font-family: "IBM Plex Mono", monospace;
    font-size: 7.5px;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: var(--rust);
    margin-top: 0.18rem;
    font-weight: 500;
  }}
  .bar-frac {{
    display: block;
    font-family: "IBM Plex Mono", monospace;
    font-size: 8px;
    color: var(--ink-soft);
    letter-spacing: 0.04em;
    margin-top: 0.1rem;
  }}

  /* ============================================================ */
  /* PAGE NAV (browser-only)                                       */
  /* ============================================================ */
  .page-nav {{
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    background: var(--ink);
    color: var(--paper);
    padding: 0.45rem 0.55rem;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    font-family: "IBM Plex Mono", monospace;
    font-size: 11px;
    letter-spacing: 0.08em;
    box-shadow: 0 8px 28px rgba(13, 17, 23, 0.18),
                0 2px 6px rgba(13, 17, 23, 0.12);
    z-index: 1000;
    user-select: none;
  }}
  .page-nav button {{
    background: none;
    border: none;
    color: var(--paper);
    cursor: pointer;
    font-size: 16px;
    padding: 0.25rem 0.55rem;
    font-family: inherit;
    border-radius: 999px;
    transition: background 120ms ease, color 120ms ease;
  }}
  .page-nav button:hover {{ background: rgba(255,255,255,0.08); }}
  .page-nav button:disabled {{ opacity: 0.3; cursor: not-allowed; }}
  .page-nav button:disabled:hover {{ background: none; }}
  .page-nav .nav-pos {{
    font-variant-numeric: tabular-nums;
    padding: 0 0.35rem;
    color: rgba(247, 244, 236, 0.7);
  }}
  .page-nav .nav-pos .nav-cur {{ color: var(--paper); }}
  .page-nav .nav-hint {{
    margin-left: 0.5rem;
    padding-left: 0.6rem;
    border-left: 0.5px solid rgba(255,255,255,0.18);
    color: rgba(247, 244, 236, 0.45);
    font-size: 9px;
    letter-spacing: 0.1em;
  }}

  /* ============================================================ */
  /* PRINT                                                         */
  /* ============================================================ */
  @media print {{
    html, body {{
      background: var(--paper);
      padding: 0;
      display: block;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .page {{
      display: flex !important;       /* show ALL pages for print */
      box-shadow: none;
      border-radius: 0;
      margin: 0;
      /* Mirror screen dimensions exactly so content that fits the
         on-screen view also fits the printed page. height: auto would
         let cells overflow into a second physical page when print
         padding > screen padding. */
      width: 8.5in;
      height: 11in;
      padding: 0.45in 0.5in 0.4in;
      overflow: hidden;
    }}
    .page-nav {{ display: none !important; }}
  }}
</style>
</head>
<body>
{pages}
<nav class="page-nav" aria-label="Page navigation">
  <button id="nav-prev" aria-label="Previous page">&larr;</button>
  <span class="nav-pos"><span class="nav-cur" id="nav-cur">01</span> / <span id="nav-total">10</span></span>
  <button id="nav-next" aria-label="Next page">&rarr;</button>
  <span class="nav-hint">&larr; &rarr; / j k</span>
</nav>
<script>
(function() {{
  var pages = Array.prototype.slice.call(document.querySelectorAll('.page'));
  if (!pages.length) return;
  var prev = document.getElementById('nav-prev');
  var next = document.getElementById('nav-next');
  var cur = document.getElementById('nav-cur');
  var total = document.getElementById('nav-total');
  total.textContent = String(pages.length).padStart(2, '0');

  // One-page-at-a-time viewer: only .page.active is rendered.
  // URL hash (#slug) keeps deep links + browser back/forward working.
  var idx = 0;
  function indexFromHash() {{
    var h = (window.location.hash || '').replace(/^#/, '');
    if (!h) return 0;
    for (var i = 0; i < pages.length; i++) {{
      if (pages[i].id === h) return i;
    }}
    return 0;
  }}
  function render() {{
    for (var i = 0; i < pages.length; i++) {{
      pages[i].classList.toggle('active', i === idx);
    }}
    cur.textContent = String(idx + 1).padStart(2, '0');
    prev.disabled = (idx === 0);
    next.disabled = (idx === pages.length - 1);
    var newHash = '#' + pages[idx].id;
    if (window.location.hash !== newHash) {{
      // replaceState avoids polluting browser back-button history with
      // every page flip; user's browser back still leaves the report.
      history.replaceState(null, '', newHash);
    }}
    window.scrollTo({{top: 0, behavior: 'instant'}});
  }}
  function go(i) {{
    idx = Math.max(0, Math.min(pages.length - 1, i));
    render();
  }}

  prev.addEventListener('click', function() {{ go(idx - 1); }});
  next.addEventListener('click', function() {{ go(idx + 1); }});

  document.addEventListener('keydown', function(e) {{
    var t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === 'j' || e.key === ' ') {{
      e.preventDefault(); go(idx + 1);
    }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp' || e.key === 'k') {{
      e.preventDefault(); go(idx - 1);
    }} else if (e.key === 'Home') {{
      e.preventDefault(); go(0);
    }} else if (e.key === 'End') {{
      e.preventDefault(); go(pages.length - 1);
    }}
  }});

  // Initial state from URL hash (so refresh keeps you on the same page).
  idx = indexFromHash();
  render();
}})();
</script>
</body>
</html>
"""


_POLICY_META = {
    "MolmoAct2":    "",
    "GR00T-N1.7":   "finetuned on MolmoAct2 YAM dataset",
    "Pi-0.5 + YAM": "finetuned on MolmoAct2 YAM dataset",
}


def build() -> None:
    tasks = _load_tasks()
    # Move any task missing its rerun.png to the back of the report
    # (stable sort preserves alphabetical order within each group).
    # If a missing rerun.png is later populated, the task automatically
    # slides back into its canonical position on the next rebuild.
    def has_rerun(t):
        return os.path.exists(
            os.path.join(_HERE, "images", slug(t["swedish"]), "rerun.png")
        )
    tasks = sorted(tasks, key=lambda t: 0 if has_rerun(t) else 1)
    pages = []
    for i, t in enumerate(tasks, 1):
        s = slug(t["swedish"])
        os.makedirs(os.path.join(_HERE, "images", s), exist_ok=True)
        gk = os.path.join(_HERE, "images", s, ".gitkeep")
        if not os.path.exists(gk):
            open(gk, "w").close()

        results = _zero_results(t)
        bars = "".join(
            BAR_TMPL.format(
                name=name,
                meta=_POLICY_META.get(name, ""),
                pct=int(100 * succ / n) if n else 0,
                s=succ,
                n=n,
                status="FAILED" if succ == 0 else ("PARTIAL" if succ < n else "PASSED"),
                fail_cls=" fail" if succ == 0 else "",
            )
            for (name, succ, n) in results
        )
        atoms = t.get("atomic_actions") or []
        atomic_items = "".join(
            ATOM_TMPL.format(n=ai, text=text)
            for ai, text in enumerate(atoms, 1)
        )
        if not atomic_items:
            atomic_items = '            <li class="atom">' \
                           '<span class="atom-mark"></span>' \
                           '<span class="atom-num">--</span>' \
                           '<span class="atom-text"><em>(no atomic actions defined)</em></span>' \
                           '</li>'

        pages.append(PAGE_TMPL.format(
            slug=s,
            page_num=f"{i:02d}",
            swedish=t["swedish"],
            english=t["english"],
            atomic_items=atomic_items,
            product_slot=_image_slot(s, "product", f"{t['swedish']} IKEA product photo"),
            manual_slot=_image_slot(s, "manual", f"{t['swedish']} IKEA assembly manual"),
            rerun_slot=_image_slot(s, "rerun", f"{t['swedish']} robot execution"),
            bars=bars,
        ))

    html = HTML_TMPL.format(pages="\n".join(pages))
    out = os.path.join(_HERE, "report.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}")
    print(f"  {len(tasks)} pages")
    print(f"  drop images at: images/<slug>/manual.png + images/<slug>/rerun.png")
    print(f"  preview: open '{out}' in a browser, then File -> Print -> Save as PDF")

    # Also publish to ../docs/ so GitHub Pages serves the latest build
    # without a separate copy step. docs/ mirrors report/ but renames
    # report.html -> index.html so the URL has a clean trailing slash.
    docs_dir = os.path.normpath(os.path.join(_HERE, "..", "docs"))
    if os.path.isdir(docs_dir):
        docs_html = os.path.join(docs_dir, "index.html")
        with open(docs_html, "w") as f:
            f.write(html)
        print(f"  published -> {docs_html}")


if __name__ == "__main__":
    build()
