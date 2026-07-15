# Changelog

All notable changes to AI Vector Cleanroom. Versions before 1.0 are alpha /
technical previews: interfaces and output may change.

## v0.5.0-alpha - 2026-07-15

Isolated-component repair, light-color fidelity, and negative-space
guardrails. (Internal lineage: Codex Beta.5, built on the v0.3 engine; the
unreleased Beta.4 light-color and negative-space work is folded into this
release. Reviewed adversarially before release.)

Component topology and repair:

- Component topology schema v1: the detail diagnostic reports per-component
  coverage, fragment counts, and complete failed-component evidence
  (measurement size, viewBox, source-component labels).
- Conservative local re-trace of completely missing, isolated, opaque,
  single-color components: a separate append-only proposal SVG is rendered and
  must pass the visual gate, per-metric non-regression checks, and an exact
  outside-bbox render guard before an atomic commit. Anything ambiguous --
  multicolor, translucent, connected, partially present, edge-touching, or
  oversized -- is reported as skipped, and any failing check rolls back to the
  original bytes.
- `report.json` carries the full component-repair audit (schema
  `ai-vector-cleanroom.component-repair/v1`): status, proposal, transaction
  verdict, and per-component reasons.

Appearance and fidelity (folds in the unreleased Beta.4 work):

- Light solid objects are recovered after tracing with a conservative overlay
  and verified by a dedicated light-core coverage gate, so white text and
  white highlights are no longer hidden behind a high overall similarity score.
- Stricter structural-core threshold on low-noise sources: faint near-white
  modeling bands (RGB ~234-247 inside white glyphs) are no longer mistaken for
  independent dark structure and falsely rejected; those low-contrast pixels
  stay covered by the color and local-detail gates.
- Multi-metric visual gate: appearance is checked on overall foreground, color,
  local-detail P10, topology, and (when applicable) light-core coverage; any
  failing applicable metric is reported explicitly instead of being averaged
  away.

Negative space and geometry:

- Negative-space guardrails on circular and rectangular frames: if an inner
  hole is filled in after a geometry conversion, that conversion is rolled
  back. Candidate, metric, and final `source_reference` share one
  stroke-proven hole mask.
- Grouped-glyph counters (the enclosed holes in letters) are kept transparent
  with a conservative counter mask, so they are not painted in as light solids.

Candidates and reporting:

- Structural-risk results expand into multiple candidates; the selection
  policy is `visual_gate_tier_then_safe_dominance_then_preserve_features`.
- Paint-resource accounting fix: a reused gradient is no longer double-counted
  as a separate solid fill.

Tests: 222 (private real-logo fixtures excluded from the public suite).

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
