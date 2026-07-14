from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from exact_native_shapes import (
    EXACT_PIXEL_EQUIVALENCE,
    EXACT_STAGE,
    find_exact_linear_candidates,
    nativeize_exact_linear_paths,
    parse_open_linear_path,
)


SVG_NS = "http://www.w3.org/2000/svg"


def _write_svg(path: Path, body: str, *, extra: str = "") -> None:
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="160" height="100" '
        f'viewBox="0 0 160 100">{extra}{body}</svg>',
        encoding="utf-8",
    )


def _exact_guard(_before: Path, _after: Path, stage: str) -> dict[str, object]:
    return {
        "accepted": True,
        "external_render_check": "completed",
        "exact_pixel_array_equal": True,
        "required_equivalence": EXACT_PIXEL_EQUIVALENCE,
        "exact_before_pixel_sha256": "a" * 64,
        "exact_after_pixel_sha256": "a" * 64,
        "stage": stage,
    }


class ExactNativeShapeTests(unittest.TestCase):
    def test_parser_preserves_exact_absolute_and_relative_vertices(self):
        points = parse_open_linear_path(
            "m 10.25,20.5 5,-2.5 h 7.25 v 3 l -1.5,2")
        self.assertEqual(points, (
            (Decimal("10.25"), Decimal("20.5")),
            (Decimal("15.25"), Decimal("18.0")),
            (Decimal("22.50"), Decimal("18.0")),
            (Decimal("22.50"), Decimal("21.0")),
            (Decimal("21.00"), Decimal("23.0")),
        ))

    def test_parser_rejects_closed_curved_compound_and_malformed_paths(self):
        rejected = [
            "M0 0 L10 0 Z",
            "M0 0 L10 0 M20 0 L30 0",
            "M0 0 C1 1 2 2 3 3",
            "M0 0 A10 10 0 0 1 20 20",
            "M0 0 L10",
            "M0,,0 L10 10",
            "M0 0 L10 10,",
            "M0 0 L0 0",
        ]
        for data in rejected:
            with self.subTest(data=data):
                self.assertIsNone(parse_open_linear_path(data))

    def test_finder_accepts_only_unstyled_open_stroked_linear_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.svg"
            _write_svg(source, """
              <path id="line" d="M10 10 L40 10" fill="none" stroke="#111"/>
              <path id="poly" d="M10 20 L30 30 L50 20" fill="none" stroke="#222"/>
              <path id="filled" d="M10 40 L30 50" fill="#111" stroke="#111"/>
              <path id="curve" d="M10 60 C20 50 30 70 40 60" fill="none" stroke="#111"/>
              <path id="marked" d="M60 10 L90 10" fill="none" stroke="#111"
                    marker-end="url(#arrow)"/>
            """)
            candidates = find_exact_linear_candidates(source)
            self.assertEqual(
                [(item.element_id, item.native_tag) for item in candidates],
                [("line", "line"), ("poly", "polyline")],
            )

    def test_transaction_preserves_order_attributes_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.svg"
            output = root / "output.svg"
            _write_svg(source, """
              <circle id="before" cx="4" cy="4" r="2" fill="#000"/>
              <path id="line" d="M10 10 L40 10" fill="none" stroke="#111"
                    stroke-width="3" stroke-linecap="round" data-role="detail"/>
              <path id="poly" d="M10 20 L30 30 L50 20" fill="none" stroke="#222"
                    stroke-width="2" stroke-linejoin="round"/>
              <circle id="after" cx="80" cy="40" r="3" fill="#000"/>
            """)
            report = nativeize_exact_linear_paths(
                source, output, validator=_exact_guard)
            self.assertEqual(report["status"], "applied")
            self.assertEqual(report["candidate_count"], 2)
            self.assertEqual(report["line_count"], 1)
            self.assertEqual(report["polyline_count"], 1)
            self.assertTrue(report["internal_checks"]["drawable_order_and_ids_preserved"])
            self.assertEqual(report["render_guard"]["stage"], EXACT_STAGE)

            svg = ET.parse(output).getroot()
            drawables = [
                element for element in svg.iter()
                if element.tag.rsplit("}", 1)[-1]
                in {"path", "line", "polyline", "circle"}
            ]
            self.assertEqual(
                [element.get("id") for element in drawables],
                ["before", "line", "poly", "after"],
            )
            line = svg.find(f".//{{{SVG_NS}}}line")
            polyline = svg.find(f".//{{{SVG_NS}}}polyline")
            self.assertIsNotNone(line)
            self.assertIsNotNone(polyline)
            self.assertEqual(
                (line.get("x1"), line.get("y1"), line.get("x2"), line.get("y2")),
                ("10", "10", "40", "10"),
            )
            self.assertEqual(line.get("stroke-linecap"), "round")
            self.assertEqual(line.get("data-role"), "detail")
            self.assertEqual(polyline.get("points"), "10,20 30,30 50,20")

    def test_output_is_not_published_without_explicit_pixel_array_equality(self):
        guards = [
            None,
            lambda *_args: True,
            lambda *_args: {
                "accepted": True,
                "external_render_check": "unavailable",
                "exact_pixel_array_equal": True,
            },
            lambda *_args: {
                "accepted": True,
                "external_render_check": "completed",
                "exact_pixel_array_equal": False,
            },
            lambda *_args: {
                "accepted": True,
                "external_render_check": "completed",
                "exact_pixel_array_equal": True,
                "required_equivalence": EXACT_PIXEL_EQUIVALENCE,
            },
            lambda *_args: {
                "accepted": True,
                "external_render_check": "completed",
                "exact_pixel_array_equal": True,
                "required_equivalence": EXACT_PIXEL_EQUIVALENCE,
                "exact_before_pixel_sha256": "a" * 64,
                "exact_after_pixel_sha256": "b" * 64,
            },
        ]
        for index, validator in enumerate(guards):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.svg"
                output = root / "output.svg"
                _write_svg(
                    source,
                    '<path id="p" d="M10 10 L40 10" fill="none" stroke="#111"/>',
                )
                output.write_text("keep-existing-output", encoding="utf-8")
                report = nativeize_exact_linear_paths(
                    source, output, validator=validator)
                self.assertEqual(report["status"], "rejected")
                self.assertFalse(report["output_written"])
                self.assertEqual(
                    output.read_text(encoding="utf-8"), "keep-existing-output")

    def test_stylesheet_dependent_svg_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.svg"
            output = Path(directory) / "output.svg"
            _write_svg(
                source,
                '<path id="p" class="wire" d="M10 10 L40 10" '
                'fill="none" stroke="#111"/>',
                extra="<style>path { stroke-width: 4px; }</style>",
            )
            self.assertEqual(find_exact_linear_candidates(source), [])
            report = nativeize_exact_linear_paths(
                source, output, validator=_exact_guard)
            self.assertEqual(report["status"], "rejected")
            self.assertFalse(output.exists())

    def test_referenced_path_is_retained_but_unreferenced_path_is_nativeized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.svg"
            output = root / "output.svg"
            _write_svg(source, """
              <path id="referenced" d="M10 10 L40 10" fill="none" stroke="#111"/>
              <path id="free" d="M10 20 L40 20" fill="none" stroke="#111"/>
              <use href="#referenced"/>
            """)
            candidates = find_exact_linear_candidates(source)
            self.assertEqual(
                [(item.element_id, item.native_tag) for item in candidates],
                [("free", "line")],
            )
            report = nativeize_exact_linear_paths(
                source, output, validator=_exact_guard)
            self.assertEqual(report["status"], "applied")
            svg = ET.parse(output).getroot()
            self.assertIsNotNone(svg.find(f".//{{{SVG_NS}}}path[@id='referenced']"))
            self.assertIsNotNone(svg.find(f".//{{{SVG_NS}}}line[@id='free']"))

    def test_active_content_is_rejected_before_static_pixel_guard(self):
        bodies = [
            '<script>document.querySelector("path")</script>'
            '<path id="p" d="M10 10 L40 10" fill="none" stroke="#111"/>',
            '<path id="p" d="M10 10 L40 10" fill="none" stroke="#111" '
            'onclick="void(0)"/>',
            '<path id="p" d="M10 10 L40 10" fill="none" stroke="#111">'
            '<animate attributeName="stroke-width" values="1;2"/></path>',
        ]
        for body in bodies:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.svg"
                output = root / "output.svg"
                _write_svg(source, body)
                self.assertEqual(find_exact_linear_candidates(source), [])
                report = nativeize_exact_linear_paths(
                    source, output, validator=_exact_guard)
                self.assertEqual(report["status"], "rejected")
                self.assertFalse(output.exists())

    def test_malformed_source_returns_rejection_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.svg"
            output = root / "output.svg"
            source.write_text("<svg><path", encoding="utf-8")
            output.write_text("keep-existing-output", encoding="utf-8")
            report = nativeize_exact_linear_paths(
                source, output, validator=_exact_guard)
            self.assertEqual(report["status"], "rejected")
            self.assertIn("could not be parsed", report["reason"])
            self.assertEqual(
                output.read_text(encoding="utf-8"), "keep-existing-output")

    def test_real_renderer_proves_line_and_polyline_pixel_arrays_equal(self):
        from vector_cleanroom import validate_svg_stage_renders

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.svg"
            output = root / "output.svg"
            _write_svg(source, """
              <path id="line" d="M10.5 10.5 L140.5 75.5" fill="none"
                    stroke="#211815" stroke-width="7.25" stroke-linecap="round"/>
              <path id="poly" d="M15.5 80.5 L55.5 35.5 L100.5 82.5 L145.5 25.5"
                    fill="none" stroke="#8cfe01" stroke-width="3.5"
                    stroke-linecap="round" stroke-linejoin="round"/>
            """)
            report = nativeize_exact_linear_paths(
                source, output, validator=validate_svg_stage_renders)
            guard = report.get("render_guard", {})
            if guard.get("external_render_check") != "completed":
                self.skipTest("optional SVG renderer unavailable")
            self.assertEqual(report["status"], "applied", report)
            self.assertTrue(guard["exact_pixel_array_equal"])


if __name__ == "__main__":
    unittest.main()
