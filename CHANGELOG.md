# Changelog

All notable changes to AI Vector Cleanroom. Versions before 1.0 are alpha /
technical previews: interfaces and output may change.

## v0.3.0-alpha - 2026-07-14

Structured editing and global recolor. (Internal lineage: Codex Beta.3.2,
built on the v0.2 engine; reviewed adversarially before release.)

Local workbench:

- Multi-image upload queue fix: jobs now carry a stable id, a second upload
  stays visibly "queued" while the first is converting, and the browser keeps
  polling until every queued job reaches a terminal state (a later waiting
  file no longer looks like it vanished).
- Live result table: each finished image appears immediately, even while later
  files are still queued, instead of only after the whole batch completes.

Native geometry and structure:

- Conservative annulus detector: co-circular, same-color, same-width open
  stroke fragments become one native `<circle>` (with `stroke-dasharray`)
  only after a bidirectional 1 px raster gate; independent rollback per stage.
- Pixel-proven line/polyline nativeization: unreferenced, style/transform-free
  open `M/L/H/V` stroke paths become `<line>`/`<polyline>` only when an
  external renderer proves the RGBA pixels are identical; fails closed when no
  renderer is available. Runs after the scene-graph stage so inherited
  presentation styles are materialized first.
- Safe compound-path splitting: large paths split into independently
  selectable parts only when hole/island topology is provably preserved; a
  cubic is rewritten to a line only when control points are provably collinear
  and monotonic via exact rationals.
- Scene Graph post-process: cross-color parts become real `<g>` groups only
  when stack order and pixels are unchanged; unsafe candidates stay
  manifest-only. Every visible element gets a stable, unique ID.

Recolor and editability:

- Paint-role manifest + offline `色彩調整.html`: global recolor of fills,
  strokes and gradient stops, OKLCH-preserving, exporting explicit SVG colors
  (no tool-specific CSS variables); active-content / external-URL / injection
  guards.
- Editability audit split into three axes (`automation_readiness`,
  `redraw_complexity`, `workflow_friction`); a 5/5 structural handle count is
  never rewritten as a 5/5 human task result. Human-timing fields default to
  `not_performed`.
- Workbench Stage 2 editing-time page records SVG-handoff vs redraw seconds;
  estimated times never count toward a saving claim, and a single session
  never promotes itself to a product "80%" claim.

Quality and safety:

- `report.json` counts the final SVG DOM and keeps per-stage / recolor-role /
  actual-vs-manifest-group evidence; withdrawn stages report only committed
  counts.
- Key writes (SVG, paint-role manifest, recolor page) use atomic replace; a
  disk-write failure never leaves half an XML file.
- Opacity output (`fill-opacity` / `stroke-opacity`) and low-contrast line
  protection (`#dddddd` on white survives); original-resolution color sampling
  keeps thin lines from turning gray or fat.

## v0.2.0-alpha - 2026-07-13

- Monoline stroke reconstruction (center line + `stroke-width`), native
  `<circle>` and stroked rings, banded-ramp `<linearGradient>` reconstruction.
- Ink-ROI foreground score with bidirectional 1 px tolerance; candidate
  comparison with automatic fallback and a hard-fail floor.
- Review workbench (zoom / object list / hotspots) and a local drag-and-drop
  workbench server; Stage 1 blind-test page.
- Third-review P0 fixes: corner detection, pixel-center offset, multicolor
  line splitting, stroke-mask guard, palette merge threshold, honest
  degradation for semi-transparent sources.

## v0.1.0-alpha - 2026-07-03

- First public release. Batch PNG/JPG/WebP/BMP → grouped editable SVG with
  palette flattening and geometry regularization; honest self-check scores;
  synthetic test suite and CI.
