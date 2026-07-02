# Changelog

## v0.1.0-alpha - 2026-07-03

Initial public release (technical preview).

Core:

- Convert PNG/JPG/WebP/BMP images into editable SVG vector drafts.
- Simplify noisy antialiasing and gradients into a cleaner limited palette.
- Preserve visual stack order while grouping paths by color/layer.
- Regularize near-circles, circular arcs, concentric rings, and rivet-like dots.
- Generate preview PNG, source reference PNG, review HTML, `report.json`,
  output notes, and a zip package per image.

Reliability and honesty fixes (pre-release review):

- SVG `<title>` and group names are XML-escaped; filenames such as `a&b.png`
  now produce valid SVG.
- New `--background auto|keep|transparent`; `keep` prevents auto removal from
  deleting light design elements that touch the image border.
- Palette detection rewritten: weighted k-means++ over unique colors with
  empty-cluster reseeding and antialiasing-blend pruning; auto mode detects up
  to 8 colors (was 6) and `--colors N` reliably yields N clusters; tiny images
  (fewer pixels than clusters) no longer crash.
- Batch runs exit with code 1 when any file fails, print a failure summary,
  and report "No successful output" when everything fails.
- Same-stem inputs (`same.png` + `same.jpg`) get distinct output names instead
  of overwriting each other.
- Self-check now reports two scores: `flat match` (fidelity to the flattened
  tracing input) and `source match` (similarity to the cleaned source — the
  honest quality number). Low source match prints a warning.
- Preview fallback (when SVG render packages are missing) is clearly labeled
  in the console, `report.json`, and `OUTPUT_README.txt` instead of silently
  posing as an SVG render.
- New `--max-size` (default 2048) bounds tracing cost on very large inputs
  while keeping the original display size in the SVG.
- New `--geometry conservative|normal|off` (default conservative; `normal`
  adds ring/band edge arc straightening; `--no-geometry` kept as a deprecated
  alias for `off`).
- New `--input` / `--output` folder options.
- vtracer output is parsed with an XML parser (regex only as fallback), and
  dependency versions are pinned in `requirements.txt` / `pyproject.toml`.
- Synthetic pytest suite (16 tests) and GitHub Actions CI on Ubuntu + Windows.
