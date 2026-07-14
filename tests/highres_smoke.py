"""Focused black-box check for the 3000px one-pixel stroke regression.

This script deliberately converts only one fixture. The full release gate is
still ``run_tests.bat``; this is the fast loop for the high-resolution fix.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# The portable Python runtime uses an isolated ``._pth`` configuration and
# does not automatically add a directly executed script's folder to
# ``sys.path``.  Make the sibling test helpers importable explicitly.
TESTS_BOOT = Path(__file__).resolve().parent
if str(TESTS_BOOT) not in sys.path:
    sys.path.insert(0, str(TESTS_BOOT))

from generate_fixtures import generate
from test_regression import PYTHON, ROOT, TESTS, _near_color, _number


RUN = TESTS / "_highres_run"
INPUT_ALL = RUN / "all_fixtures"
INPUT = RUN / "input"
OUTPUT = RUN / "output"
NAME = "one_px_black_3000"


def check() -> None:
    if RUN.exists():
        shutil.rmtree(RUN)
    INPUT_ALL.mkdir(parents=True)
    INPUT.mkdir(parents=True)
    generate(INPUT_ALL)
    shutil.copy2(INPUT_ALL / f"{NAME}.png", INPUT / f"{NAME}.png")

    process = subprocess.run(
        [str(PYTHON), str(ROOT / "vector_cleanroom.py"),
         "--input", str(INPUT), "--output", str(OUTPUT)],
        cwd=str(ROOT), text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=300,
        env={**os.environ, "PYTHONUTF8": "1"},
    )
    (RUN / "process_stdout.txt").write_text(process.stdout, encoding="utf-8")
    (RUN / "process_stderr.txt").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise AssertionError(
            f"CLI exit code {process.returncode}\n{process.stdout}{process.stderr}"
        )

    result = OUTPUT / f"result_{NAME}"
    report = json.loads((result / "report.json").read_text(encoding="utf-8"))
    score = report.get("foreground_match_percent")
    if score is None or score < 95:
        raise AssertionError(f"foreground match {score!r} is below 95")
    if report.get("strokes") != 1:
        raise AssertionError(f"expected 1 stroke, got {report.get('strokes')}")
    if report.get("nodes_total", 999) > 3:
        raise AssertionError(f"expected <=3 nodes, got {report.get('nodes_total')}")

    svg = result / f"{NAME}_vector.svg"
    root = ET.parse(svg).getroot()
    canvas_width = _number(root.attrib["width"])
    viewbox_width = float(root.attrib["viewBox"].split()[2])
    strokes = [
        element for element in root.iter()
        if "stroke-width" in element.attrib
        and element.attrib.get("stroke", "").lower() not in {"", "none"}
    ]
    if len(strokes) != 1:
        raise AssertionError(f"expected one SVG stroke, got {len(strokes)}")
    stroke = strokes[0]
    displayed_width = float(stroke.attrib["stroke-width"]) * canvas_width / viewbox_width
    if abs(displayed_width - 1.0) > 0.25:
        raise AssertionError(f"displayed stroke width {displayed_width:.3f}px is not 1px")
    if not _near_color(stroke.attrib.get("stroke"), (0, 0, 0), 5):
        raise AssertionError(f"stroke color changed to {stroke.attrib.get('stroke')!r}")

    print(
        f"PASS {NAME}: foreground={score:.2f}, stroke={displayed_width:.2f}px, "
        f"nodes={report['nodes_total']}"
    )


if __name__ == "__main__":
    check()
