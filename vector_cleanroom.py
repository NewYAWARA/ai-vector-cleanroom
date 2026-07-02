# -*- coding: utf-8 -*-
"""
AI Vector Cleanroom

Batch-process images in the input folder and create editable SVG vector drafts:
  result_<name>/
      <name>_vector.svg              editable vector paths, grouped by color/layer
      <name>_preview.png             preview rendered from the SVG
      source_reference.png           cleaned source reference
      review.html                    overlay review page
      report.json                    machine-readable run report
      OUTPUT_README.txt              output notes
  result_<name>.zip                  zipped output folder

If two inputs share the same stem (e.g. same.png and same.jpg), the extension
is appended to keep their outputs separate.

This tool does not recover original vector artwork from a bitmap. It creates a
clean, editable vector approximation that must still be reviewed by a human.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import shutil
import sys
import zipfile
from pathlib import Path

if getattr(sys, "frozen", False):
    BASE = Path(sys.executable).resolve().parent
else:
    BASE = Path(__file__).resolve().parent

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_inputs(input_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(p for p in input_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in EXTS)


def plan_output_names(paths):
    """Map each input path to a collision-free output base name.

    Inputs sharing a stem (same.png + same.jpg) get the extension appended,
    so results never overwrite each other.
    """
    by_stem = {}
    for p in paths:
        by_stem.setdefault(p.stem, []).append(p)
    plan = {}
    for stem, group in by_stem.items():
        if len(group) == 1:
            plan[group[0]] = stem
        else:
            for p in group:
                plan[p] = f"{stem}_{p.suffix.lstrip('.').lower()}"
    return plan


def render_svg_png(svg_path: Path, png_path: Path, size=2000, bg=0xffffff):
    """Render the SVG to PNG via svglib. Returns False if deps are missing."""
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        d = svg2rlg(str(svg_path))
        if not d or not d.width:
            return False
        s = size / float(d.width)
        d.scale(s, s)
        d.width *= s
        d.height *= s
        renderPM.drawToFile(d, str(png_path), fmt="PNG", bg=bg)
        return png_path.exists()
    except Exception:
        return False


def _match_percent(render_png: Path, reference_png: Path):
    """Pixel match (max channel diff < 48) between two images on white."""
    import numpy as np
    from PIL import Image
    ref = Image.open(reference_png).convert("RGBA")
    base = Image.new("RGB", ref.size, (255, 255, 255))
    base.paste(ref, (0, 0), ref)
    ren = Image.open(render_png).convert("RGB")
    if ren.size != base.size:
        ren = ren.resize(base.size)
    a = np.asarray(ren, dtype=np.int16)
    b = np.asarray(base, dtype=np.int16)
    return float((np.abs(a - b).max(2) < 48).mean() * 100)


def self_check(svg_path: Path, flat_png: Path, source_png: Path):
    """Render the SVG back and compare it against two references.

    Returns {"flat": float|None, "source": float|None}.
      flat   — fidelity to the flattened (palette-reduced) image. Measures
               how faithfully the SVG reproduces the tracing input.
      source — similarity to the cleaned source image. This is the honest
               quality number: gradients/shadows lost in flattening lower it.
    """
    out = {"flat": None, "source": None}
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        d = svg2rlg(str(svg_path))
        if not d or not d.width:
            return out
        tmp = svg_path.parent / "_selfcheck.png"
        renderPM.drawToFile(d, str(tmp), fmt="PNG", bg=0xffffff)
        try:
            out["flat"] = _match_percent(tmp, flat_png)
        except Exception:
            pass
        try:
            out["source"] = _match_percent(tmp, source_png)
        except Exception:
            pass
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return out


def data_url(path: Path):
    mime = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b}"


def make_review_html(out: Path, name: str, original_png: Path, svg_text: str, size):
    w, h = size
    svg_inline = svg_text.split("?>", 1)[-1].strip()
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{html.escape(name)} Review</title>
<style>
 *{{box-sizing:border-box}} body{{margin:0;font-family:system-ui,sans-serif;background:#eee;color:#222}}
 .bar{{position:sticky;top:0;display:flex;flex-wrap:wrap;gap:14px;align-items:center;padding:12px 16px;background:#fff;border-bottom:1px solid #ccc;z-index:5}}
 .bar label{{display:inline-flex;align-items:center;gap:8px;font-size:14px}} input[type=range]{{width:150px}}
 .wrap{{display:grid;place-items:center;padding:24px}}
 .stage{{position:relative;width:min(90vw,820px);aspect-ratio:{w}/{h};border:1px solid #bbb;overflow:hidden;background:#fff;
   background-image:linear-gradient(45deg,#ddd 25%,transparent 25%),linear-gradient(-45deg,#ddd 25%,transparent 25%),linear-gradient(45deg,transparent 75%,#ddd 75%),linear-gradient(-45deg,transparent 75%,#ddd 75%);
   background-size:32px 32px;background-position:0 0,0 16px,16px -16px,-16px 0}}
 .stage.white{{background:#fff}} .stage.black{{background:#111}}
 .stage img,.vec{{position:absolute;inset:0;width:100%;height:100%}} .stage img{{object-fit:contain;opacity:.5}}
 .vec svg{{width:100%;height:100%;display:block}}
 .note{{position:fixed;right:16px;bottom:14px;max-width:380px;padding:10px 12px;background:rgba(255,255,255,.92);border:1px solid #ccc;border-radius:8px;font-size:13px;line-height:1.5}}
</style></head><body>
 <div class="bar">
  <label>Source opacity <input id="o" type="range" min="0" max="100" value="50"></label>
  <label>Vector opacity <input id="v" type="range" min="0" max="100" value="100"></label>
  <label>Background <select id="bg"><option value="">Checkerboard</option><option value="white">White</option><option value="black">Black</option></select></label>
 </div>
 <div class="wrap"><div id="stage" class="stage">
   <img id="orig" src="{data_url(original_png)}" alt="source image">
   <div class="vec">{svg_inline}</div>
 </div></div>
 <div class="note">Adjust the opacity sliders to compare the cleaned vector with the source image. The SVG is editable in vector graphics software.</div>
 <script>
  var st=document.getElementById('stage');
  o.oninput=function(){{document.getElementById('orig').style.opacity=o.value/100}};
  v.oninput=function(){{document.querySelector('.vec').style.opacity=v.value/100}};
  bg.onchange=function(){{st.className='stage '+bg.value}};
 </script>
</body></html>"""
    p = out / "review.html"
    p.write_text(page, encoding="utf-8")
    return p


def make_output_readme(out: Path, name: str, palette, geometry_notes=None,
                       preview_is_fallback=False, scores=None):
    pal_lines = "\n".join(f"    {nm}: {hx}" for nm, hx in palette)
    geo_block = ""
    if geometry_notes:
        geo_lines = "\n".join(f"  - {g}" for g in geometry_notes)
        geo_block = f"""
Geometry regularization applied
-------------------------------
{geo_lines}
  Review the SVG visually; geometry detection is heuristic and may need manual correction.
"""
    if preview_is_fallback:
        preview_line = (f"  {name}_preview.png     WARNING: NOT rendered from the SVG.\n"
                        "                         SVG rendering packages were missing, so this\n"
                        "                         is the cleaned source image. Open review.html\n"
                        "                         or the SVG itself to inspect the real result.")
    else:
        preview_line = f"  {name}_preview.png     Preview rendered from the SVG."
    score_block = ""
    if scores and (scores.get("flat") is not None or scores.get("source") is not None):
        fm = f"{scores['flat']:.1f}%" if scores.get("flat") is not None else "n/a"
        sm = f"{scores['source']:.1f}%" if scores.get("source") is not None else "n/a"
        score_block = f"""
Self-check scores
-----------------
  flat match   {fm}   fidelity to the palette-flattened tracing input
  source match {sm}   similarity to the cleaned source image (honest quality
                      number; gradients and shadows lost in flattening lower it)
"""
    note = f"""{name} - Vector Cleanroom Output
=================================

This folder contains an editable vector approximation generated from a bitmap
source image. It is intended as a clean starting point for further review,
editing, and production cleanup.

Files
-----
  {name}_vector.svg      Editable SVG vector paths. Open with Illustrator,
                         Inkscape, Affinity Designer, Figma, etc.
{preview_line}
  source_reference.png   Source reference after background cleanup.
  review.html            Browser-based overlay page for visual comparison.
  report.json            Machine-readable run report.
  OUTPUT_README.txt      This file.

What the tool did
-----------------
  - Converted the image into SVG paths; no bitmap is embedded in the SVG.
  - Reduced noisy gradients and antialiasing into a smaller clean palette.
  - Preserved stack order so shapes that visually sit on top stay on top.
  - Grouped paths by color/layer for easier editing.
{geo_block}{score_block}
Limitations
-----------
  - Bitmap images do not contain original vector curves, font data, or layer
    structure, so this is an approximation rather than lossless recovery.
  - Text is converted to outline paths, not editable font text.
  - Highly detailed photos, soft shadows, and complex gradients are not the
    target use case; flat logos and graphic marks work best.

Detected palette
----------------
{pal_lines}
"""
    p = out / "OUTPUT_README.txt"
    p.write_text(note, encoding="utf-8")
    return p


def process_one(img_path: Path, out_base: str, args, output_dir: Path):
    from PIL import Image

    from trace_engine import _prepare_image
    from clean_base import build_clean_base

    warnings = []
    print(f"\n[ {img_path.name} ]")
    deliver = output_dir / f"result_{out_base}"
    if deliver.exists():
        shutil.rmtree(deliver)
    deliver.mkdir(parents=True)

    # 1) Source reference (full size) with the chosen background mode.
    clean_img, _sz, removed = _prepare_image(
        img_path, max_size=0, background=args.background,
        white_threshold=args.white_threshold, alpha_threshold=12,
    )
    ref_png = deliver / "source_reference.png"
    clean_img.save(ref_png)
    msg = " (outer light/checker background removed)" if removed else ""
    print(f"  Source reference OK{msg}")
    if removed:
        warnings.append("auto background removal was applied; if a light "
                        "design element touching the border disappeared, "
                        "re-run with --background keep")

    # 2) Clean vector result with optional geometry regularization.
    svg_path = deliver / f"{out_base}_vector.svg"
    flat_chk = deliver / "_flat_check.png"
    stats = build_clean_base(img_path, svg_path,
                             forced_colors=args.colors,
                             white_threshold=args.white_threshold,
                             background=args.background,
                             max_size=args.max_size,
                             geometry=args.geometry,
                             flat_out=flat_chk)
    pal_desc = ", ".join(f"{nm}({hx})" for nm, hx in stats.palette)
    print(f"  Vector OK: {stats.colors} color/layer groups -> {pal_desc}")
    for g in stats.geometry_notes:
        print(f"    - {g}")

    # 2b) Render back and compare against both references.
    scores = self_check(svg_path, flat_chk, ref_png)
    if scores["flat"] is not None or scores["source"] is not None:
        fm = f"{scores['flat']:.1f}%" if scores["flat"] is not None else "n/a"
        sm = f"{scores['source']:.1f}%" if scores["source"] is not None else "n/a"
        print(f"  Self-check: flat match {fm} / source match {sm}")
        if scores["source"] is not None and scores["source"] < 90:
            warnings.append("source match below 90%: the image likely has "
                            "gradients, shadows, or fine detail outside this "
                            "tool's target use case; review the overlay page")
            print("    WARNING: low source match, review the overlay page.")
    flat_chk.unlink(missing_ok=True)

    # 3) Preview. Never silently pass off the source as an SVG render.
    preview = deliver / f"{out_base}_preview.png"
    preview_is_fallback = not render_svg_png(svg_path, preview)
    if preview_is_fallback:
        bg = Image.new("RGB", clean_img.size, (255, 255, 255))
        bg.paste(clean_img, (0, 0), clean_img)
        bg.save(preview)
        warnings.append("SVG preview unavailable (svglib/reportlab missing); "
                        f"{preview.name} is the cleaned source image, not an "
                        "SVG render")
        print("  Preview: SVG render unavailable — wrote cleaned source image "
              "instead (install requirements-preview.txt for real previews)")
    else:
        print("  Preview OK")

    # 4) Review page, JSON report, output notes.
    make_review_html(deliver, out_base, ref_png, svg_path.read_text(encoding="utf-8"),
                     (stats.width, stats.height))
    report = {
        "input": img_path.name,
        "output_base": out_base,
        "size": [stats.width, stats.height],
        "palette": [{"name": nm, "hex": hx} for nm, hx in stats.palette],
        "groups": stats.colors,
        "paths": stats.n_paths,
        "background_removed": stats.removed_background,
        "geometry_level": args.geometry,
        "geometry_notes": stats.geometry_notes,
        "flat_match_percent": scores["flat"],
        "source_match_percent": scores["source"],
        "preview_is_svg_render": not preview_is_fallback,
        "warnings": warnings,
        "options": {
            "colors": args.colors,
            "white_threshold": args.white_threshold,
            "background": args.background,
            "max_size": args.max_size,
        },
    }
    (deliver / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_output_readme(deliver, out_base, stats.palette, stats.geometry_notes,
                       preview_is_fallback=preview_is_fallback, scores=scores)
    print("  Review page + report + output notes OK")

    # 5) Zip the result folder.
    zip_path = output_dir / f"result_{out_base}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(deliver.iterdir()):
            z.write(f, arcname=f"{deliver.name}/{f.name}")
    mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  Result package: {zip_path.name} ({mb:.2f} MB)")
    return zip_path


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="AI Vector Cleanroom — bitmap logo/icon to editable SVG draft")
    ap.add_argument("--input", type=Path, default=BASE / "input",
                    help="input folder (default: ./input)")
    ap.add_argument("--output", type=Path, default=BASE / "output",
                    help="output folder (default: ./output)")
    ap.add_argument("--colors", type=int, default=0,
                    help="force a fixed palette size; default 0 = auto-detect")
    ap.add_argument("--white-threshold", type=int, default=220,
                    help="threshold for light/checker background cleanup; default 220")
    ap.add_argument("--background", choices=["auto", "keep", "transparent"],
                    default="auto",
                    help="background handling: auto = heuristic removal of light "
                         "border-connected background; keep = never remove; "
                         "transparent = force removal (default: auto)")
    ap.add_argument("--max-size", type=int, default=2048,
                    help="downscale the longest side before tracing; 0 disables "
                         "(default: 2048)")
    ap.add_argument("--geometry", choices=["conservative", "normal", "off"],
                    default="conservative",
                    help="geometry regularization level (default: conservative; "
                         "normal additionally straightens ring/band edges into "
                         "mathematical arcs)")
    ap.add_argument("--no-geometry", action="store_true",
                    help=argparse.SUPPRESS)   # deprecated alias for --geometry off
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.no_geometry:
        args.geometry = "off"

    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_inputs(input_dir)
    if not images:
        print(f"No images found. Put PNG/JPG/WebP/BMP files in:\n  {input_dir}")
        return 1

    print("=" * 56)
    print("  AI Vector Cleanroom")
    print("=" * 56)
    print("Creates editable SVG vector drafts for each image in the input folder.")

    plan = plan_output_names(images)
    made, failed = [], []
    for img in images:
        try:
            made.append(process_one(img, plan[img], args, output_dir))
        except Exception as e:
            import traceback
            print(f"  [failed] {img.name}: {e}")
            traceback.print_exc()
            failed.append(img.name)
            shutil.rmtree(output_dir / f"result_{plan[img]}", ignore_errors=True)

    print("\n" + "=" * 56)
    if made:
        print("Done. Result packages:")
        for z in made:
            print(f"   {z.name}")
    if failed:
        print(f"\n{len(failed)} file(s) FAILED: {', '.join(failed)}")
    if not made:
        print("\nNo successful output.")
    print(f"\nOutput: {output_dir}")
    return 1 if failed or not made else 0


if __name__ == "__main__":
    sys.exit(main())
