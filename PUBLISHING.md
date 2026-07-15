# Publishing Guide

How to replace the published version of AI Vector Cleanroom on GitHub with a
new one, while keeping the repository, its URL, stars and issue history.

## 0. Before you push

```powershell
python preflight_check.py                 # must print "Preflight OK"
python -m unittest discover -s tests -v   # must end with "OK"
```

`preflight_check.py` fails the build if any binary asset or private string
would be published. Never bypass it.

## 1. Replace the contents of the existing repo (preserves history)

From a clone of the repo, with the new tree prepared separately:

```powershell
git clone https://github.com/NewYAWARA/ai-vector-cleanroom.git
cd ai-vector-cleanroom

# remove everything that git tracks (keeps .git), then copy the new tree in
git rm -r --quiet .
robocopy "PATH\TO\new-tree" . /E /XD .git    # Windows; use rsync on macOS/Linux

git add -A
git commit -m "Release v0.5.0-alpha: component-repair, light-color fidelity, negative-space guardrails"
git push origin main
```

This makes HEAD the new version; the old code stays reachable in history.

## 2. Tag a pre-release

```powershell
git tag v0.5.0-alpha
git push origin v0.5.0-alpha
```

On GitHub, create a Release from that tag and **check "This is a
pre-release"**. Suggested notes (accurate wording only):

```text
v0.5.0-alpha (technical preview). Flat bitmap logos/icons -> editable SVG:
real strokes, native circle/line/polyline, linear gradients, safe compound
splitting, scene grouping, offline global recolor. New since v0.3: light-color
fidelity (conservative overlay + light-core coverage gate), negative-space
guardrails on frames and grouped-glyph counters, a multi-metric visual gate,
and conservative render-validated re-trace of completely-missing components.
Every post-process stage is validated pixel-exact by an external renderer.
Output still needs human review; the "80% time saving" claim is not yet
verified by designer editing timings.
```

## 3. Wording rules

Use accurate wording:

```text
Turns flat bitmap logos and icons into clean, editable, structured SVG
drafts. Alpha / technical preview: output needs human review, especially for
gradients, shadows, transparency, text, and complex multi-color artwork.
```

Do not claim:

```text
lossless vector recovery
one-click 80%/90% vectorization of any image
production ready / designer ready without review
```

## 4. Do not publish restricted assets

Private, client-owned, trademarked, or unclear-license images — and any SVG
generated from them — must never be committed. `input/`, `output/`,
`tests/fixtures/`, and the portable `python/` interpreter are gitignored;
keep them that way.
