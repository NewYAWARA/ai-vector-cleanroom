# Publishing Guide

This checklist is for publishing AI Vector Cleanroom to GitHub for the first
time.

## 1. Create the GitHub repository

Recommended repository name:

```text
ai-vector-cleanroom
```

Recommended description:

```text
Clean editable SVG drafts from AI-generated or bitmap logos.
```

Use `Public` visibility. If you upload this prepared folder, do not ask GitHub
to auto-create a README, license, or `.gitignore`; those files already exist.

## 2. Prepare local Git

Run these commands from this folder:

```powershell
python preflight_check.py
python -m py_compile vector_cleanroom.py clean_base.py trace_engine.py
git init -b main
git config user.name "張進逸 (Shinichi Chang)"
git config user.email "123047152+NewYAWARA@users.noreply.github.com"
git add .
git commit -m "Initial open source release"
```

If you prefer to keep your email private in commits, replace the email above
with your GitHub no-reply email before committing.

## 3. Connect to GitHub

After creating the empty GitHub repository, GitHub will show a remote URL.
Use your real username:

```powershell
git remote add origin https://github.com/NewYAWARA/ai-vector-cleanroom.git
git push -u origin main
```

## 4. Add repository metadata

Suggested topics:

```text
svg, vector, logo, raster-to-vector, vtracer, image-processing, ai-generated-art
```

Suggested website/about text:

```text
Clean editable SVG drafts from AI-generated or bitmap logos.
```

## 5. Create the first release

Create a GitHub Release:

- Tag: `v0.1.0-alpha`
- Title: `AI Vector Cleanroom v0.1.0-alpha (technical preview)`
- Notes:

```text
First public release of AI Vector Cleanroom (v0.1.0-alpha, technical preview).

Scope: flat, limited-palette logos and icons. Output is an editable SVG
draft that still needs human review.

- Converts PNG/JPG/WebP/BMP images into editable SVG vector drafts.
- Preserves visual stack order while grouping paths by color/layer.
- Optional geometry regularization for near-circles, rings, and rivet-like dots
  (conservative by default; --geometry normal|off available).
- Self-check reports both flat match and source match scores.
- Review HTML, report.json, and output notes for inspection.
- Tested with a synthetic pytest suite; CI on Ubuntu and Windows.

Maintained by 張進逸 (Shinichi Chang).
```

Mark the release as a **pre-release** in the GitHub release form.

## 6. Public wording

Use accurate wording:

```text
AI Vector Cleanroom turns flat bitmap logos and icons into clean, editable
SVG drafts (palette flattening, stack-order grouping, geometry
regularization). Alpha / technical preview: output needs human review,
especially for gradients, shadows, and multi-color artwork.
```

Avoid misleading wording:

```text
lossless vector recovery
perfect one-click vectorization
one-click 90%+ vectorization
production ready / designer ready without review
```

## 7. Do not publish restricted assets

Do not commit private, client-owned, trademarked, or unclear-license images.
Generated SVG outputs inherit the legal restrictions of the source image.
