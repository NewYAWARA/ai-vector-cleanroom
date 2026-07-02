# -*- coding: utf-8 -*-
"""Synthetic-fixture tests for AI Vector Cleanroom.

All fixtures are generated in-test with Pillow; no binary assets are stored
in the repository. Tests only require the core dependencies
(pillow / numpy / vtracer); preview packages are optional.
"""

import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clean_base import build_clean_base, detect_palette, _kmeans, _unique_colors  # noqa: E402
from trace_engine import _prepare_image  # noqa: E402
from vector_cleanroom import plan_output_names  # noqa: E402


# ---------- fixtures ----------

def circle_logo(path: Path, size=300, fill=(200, 30, 30)):
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([size * 0.2, size * 0.2, size * 0.8, size * 0.8], fill=fill)
    img.save(path)
    return path


def grid8(path: Path):
    cols = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 200, 0),
            (0, 0, 255), (255, 220, 0), (0, 200, 255), (200, 0, 180)]
    img = Image.new("RGB", (400, 200))
    d = ImageDraw.Draw(img)
    for i, c in enumerate(cols):
        d.rectangle([i * 50, 0, (i + 1) * 50 - 1, 199], fill=c)
    img.save(path)
    return cols


def wobbly_badge(path: Path, size=600, seed=7):
    """A deliberately jittered badge: bumpy disc + uneven ring of dots."""
    import random
    random.seed(seed)
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    c = size / 2
    pts = [(c + (size * 0.4 + random.uniform(-2, 2)) * math.cos(math.radians(a)),
            c + (size * 0.4 + random.uniform(-2, 2)) * math.sin(math.radians(a)))
           for a in range(0, 360, 4)]
    d.polygon(pts, fill=(20, 40, 90))
    for i in range(8):
        a = i * 45 + random.uniform(-2, 2)
        o = size * 0.32
        r = size * 0.03 + random.uniform(-1, 1)
        x = c + o * math.cos(math.radians(a))
        y = c + o * math.sin(math.radians(a))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(250, 250, 250))
    img.save(path)


# ---------- engine tests (review P1 reproductions) ----------

def test_ampersand_filename_produces_valid_xml(tmp_path):
    src = circle_logo(tmp_path / "a&b.png")
    out = tmp_path / "out.svg"
    build_clean_base(src, out)
    ET.parse(out)  # raises on invalid XML


def test_tiny_image_does_not_crash(tmp_path):
    img = Image.new("RGB", (2, 2))
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((1, 1), (0, 0, 255))
    img.save(tmp_path / "tiny.png")
    stats = build_clean_base(tmp_path / "tiny.png", tmp_path / "o.svg",
                             background="keep")
    assert (tmp_path / "o.svg").exists()
    assert stats.colors >= 1


def test_eight_distinct_colors_all_detected(tmp_path):
    grid8(tmp_path / "grid8.png")
    stats = build_clean_base(tmp_path / "grid8.png", tmp_path / "o.svg",
                             background="keep")
    assert len({hx for _, hx in stats.palette}) >= 8


def test_forced_colors_reaches_requested_count(tmp_path):
    grid8(tmp_path / "grid8.png")
    stats = build_clean_base(tmp_path / "grid8.png", tmp_path / "o.svg",
                             background="keep", forced_colors=8)
    assert len({hx for _, hx in stats.palette}) == 8


def test_background_keep_preserves_border_touching_light_area(tmp_path):
    img = Image.new("RGB", (300, 300), (230, 230, 230))
    ImageDraw.Draw(img).rectangle([100, 100, 200, 200], fill=(0, 0, 0))
    img.save(tmp_path / "gray.png")

    st_auto = build_clean_base(tmp_path / "gray.png", tmp_path / "a.svg",
                               background="auto")
    st_keep = build_clean_base(tmp_path / "gray.png", tmp_path / "k.svg",
                               background="keep")
    assert st_auto.removed_background is True
    assert st_keep.removed_background is False
    # the light gray must survive as a palette color in keep mode
    assert any(hx.lower() not in ("#000000",) and int(hx[1:3], 16) > 180
               for _, hx in st_keep.palette)


def test_prepare_image_background_modes(tmp_path):
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([30, 30, 70, 70], fill=(0, 0, 0))
    img.save(tmp_path / "w.png")
    _, _, removed_keep = _prepare_image(tmp_path / "w.png", max_size=0,
                                        background="keep",
                                        white_threshold=220, alpha_threshold=12)
    _, _, removed_auto = _prepare_image(tmp_path / "w.png", max_size=0,
                                        background="auto",
                                        white_threshold=220, alpha_threshold=12)
    assert removed_keep is False
    assert removed_auto is True


def test_kmeans_small_population_no_crash():
    import numpy as np
    pix = np.array([[255, 0, 0], [0, 0, 255], [255, 0, 0]], dtype=np.float32)
    cent = _kmeans(pix, 8)
    assert 1 <= len(cent) <= 2


def test_kmeans_dominant_color_keeps_clusters():
    import numpy as np
    # 99% white, 1% red: naive random init would often collapse clusters.
    pix = np.vstack([np.full((9900, 3), 255.0, dtype=np.float32),
                     np.tile(np.array([[200.0, 0.0, 0.0]], dtype=np.float32), (100, 1))])
    cent = _kmeans(pix, 2)
    dists = ((cent[None] - cent[:, None]) ** 2).sum(-1)
    assert len(cent) == 2 and dists.max() > 100 ** 2


def test_output_name_collision(tmp_path):
    a = tmp_path / "same.png"
    b = tmp_path / "same.jpg"
    a.touch(), b.touch()
    plan = plan_output_names([a, b])
    assert len(set(plan.values())) == 2
    assert plan[a] != plan[b]


def test_max_size_downscales_but_keeps_display_size(tmp_path):
    src = circle_logo(tmp_path / "big.png", size=900)
    stats = build_clean_base(src, tmp_path / "o.svg", max_size=300)
    assert stats.width == 900 and stats.height == 900   # display size
    root = ET.parse(tmp_path / "o.svg").getroot()
    vb = root.get("viewBox").split()
    assert int(vb[2]) <= 300                            # traced at reduced size


# ---------- geometry regularization ----------

def test_geometry_conservative_snaps_circles(tmp_path):
    wobbly_badge(tmp_path / "badge.png")
    stats = build_clean_base(tmp_path / "badge.png", tmp_path / "o.svg",
                             geometry="conservative")
    assert any("perfect circles" in n for n in stats.geometry_notes)
    svg = (tmp_path / "o.svg").read_text(encoding="utf-8")
    assert " A" in svg or "A" in svg  # arcs present


def test_geometry_off_produces_no_notes(tmp_path):
    wobbly_badge(tmp_path / "badge.png")
    stats = build_clean_base(tmp_path / "badge.png", tmp_path / "o.svg",
                             geometry="off")
    assert stats.geometry_notes == []


def test_geometry_preserves_non_circular_shapes(tmp_path):
    # A star must never be "regularized" into a circle.
    img = Image.new("RGB", (400, 400), (255, 255, 255))
    c = 200
    pts = []
    for i in range(10):
        r = 150 if i % 2 == 0 else 60
        a = math.radians(i * 36 - 90)
        pts.append((c + r * math.cos(a), c + r * math.sin(a)))
    ImageDraw.Draw(img).polygon(pts, fill=(0, 0, 0))
    img.save(tmp_path / "star.png")
    build_clean_base(tmp_path / "star.png", tmp_path / "o.svg", geometry="normal")
    svg = (tmp_path / "o.svg").read_text(encoding="utf-8")
    # a star flattened into a full circle would be two arc commands only
    assert svg.count("C") > 10


# ---------- CLI ----------

def _run_cli(args, cwd=ROOT):
    return subprocess.run([sys.executable, "-X", "utf8", "vector_cleanroom.py",
                           *args], cwd=str(cwd), capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def test_cli_failure_sets_exit_code(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    circle_logo(inp / "ok.png")
    (inp / "corrupt.png").write_bytes(b"not a png")
    r = _run_cli(["--input", str(inp), "--output", str(tmp_path / "out")])
    assert r.returncode == 1
    assert "FAILED" in r.stdout
    assert (tmp_path / "out" / "result_ok").is_dir()
    assert not (tmp_path / "out" / "result_corrupt").exists()


def test_cli_all_good_exit_zero_and_report(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    circle_logo(inp / "logo.png")
    r = _run_cli(["--input", str(inp), "--output", str(tmp_path / "out")])
    assert r.returncode == 0
    report = json.loads((tmp_path / "out" / "result_logo" / "report.json")
                        .read_text(encoding="utf-8"))
    assert report["groups"] >= 1
    assert report["paths"] >= 1
    assert isinstance(report["warnings"], list)


def test_cli_all_fail_reports_no_output(tmp_path):
    inp = tmp_path / "in"
    inp.mkdir()
    (inp / "bad.png").write_bytes(b"xx")
    r = _run_cli(["--input", str(inp), "--output", str(tmp_path / "out")])
    assert r.returncode == 1
    assert "No successful output" in r.stdout
