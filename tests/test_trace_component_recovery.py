from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from trace_component_recovery import recover_missing_source_components
from svg_renderer import render_svg_reference


class TraceComponentRecoveryTests(unittest.TestCase):
    def render(self, svg):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "input.svg", Path(folder) / "render.png"
            source.write_text(svg, encoding="utf-8")
            render_svg_reference(source, target, width=64, background="#ffffff")
            return np.asarray(Image.open(target).convert("RGBA"))

    def test_lost_same_colour_dot_is_restored_without_replacing_main_shape(self):
        raw='<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><path id="body" fill="#206040" d="M8 28H56V56H8Z"/></svg>'
        source=self.render(raw.replace('</svg>','<circle cx="20" cy="12" r="3" fill="#206040"/></svg>'))
        repaired, report=recover_missing_source_components(raw,source)
        self.assertEqual(report['recovered_count'],1)
        self.assertEqual(report['records'][0]['kind'],'foreground_component')
        self.assertIn('M8 28H56V56H8Z',repaired)
        self.assertLess(self.render(repaired)[12,20,1],150)

    def test_missing_enclosed_background_patch_is_restored(self):
        raw='<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><path fill="#206040" d="M8 8H56V56H8Z"/></svg>'
        source=self.render(raw.replace('</svg>','<circle cx="32" cy="24" r="4" fill="white"/></svg>'))
        repaired, report=recover_missing_source_components(raw,source)
        self.assertEqual(report['recovered_count'],1)
        self.assertEqual(report['records'][0]['kind'],'enclosed_background_compartment')
        self.assertGreater(self.render(repaired)[24,32,0],240)

    def test_existing_component_and_excluded_ownership_are_not_duplicated(self):
        raw='<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><circle cx="20" cy="12" r="3" fill="#206040"/></svg>'
        source=self.render(raw)
        self.assertEqual(recover_missing_source_components(raw,source)[1]['recovered_count'],0)
        empty='<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"/>'
        self.assertEqual(recover_missing_source_components(empty,source,excluded_mask=np.ones((64,64),dtype=bool))[1]['recovered_count'],0)

    def test_one_pixel_noise_and_two_colour_components_abstain(self):
        raw='<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"/>'
        source=np.full((64,64,4),255,dtype=np.uint8)
        source[10,10,:3]=(0,0,0)
        source[20:25,20:23,:3]=(200,0,0)
        source[20:25,23:26,:3]=(0,0,200)
        repaired,report=recover_missing_source_components(raw,source)
        self.assertEqual(report['recovered_count'],0)
        self.assertEqual(repaired,raw)

    def test_coordinate_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'coordinate'):
            recover_missing_source_components('<svg width="32" height="32"/>',np.full((64,64,4),255,dtype=np.uint8))


if __name__=='__main__':
    unittest.main()
