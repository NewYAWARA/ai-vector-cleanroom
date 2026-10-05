"""Safety regressions for conservative stroke reconstruction."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image, ImageDraw

from stroke_engine import extract_strokes


class StrokeSafetyTests(unittest.TestCase):
    def test_antialiased_thin_rules_recover_source_width_color_and_subpixel_centres(self):
        from pathlib import Path
        import tempfile
        from svg_renderer import render_svg_reference
        from vector_cleanroom import _match_percent
        ink=(38,57,76)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); original=folder/'original.svg'
            original.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'+''.join(
                f'<path d="M36 {y} H220" fill="none" stroke="#26394c" stroke-width="{width}" stroke-linecap="round"/>'
                for y,width in ((64,2),(126,4),(188,6)))+'</svg>')
            for size in (128,384):
                with self.subTest(size=size):
                    png=folder/f'source-{size}.png';render_svg_reference(original,png,width=size)
                    _,strokes,_,_=self._extract(Image.open(png),[(147,157,166),ink])
                    self.assertEqual(len(strokes),3)
                    for stroke, expected in zip(strokes,(2,4,6)):
                        self.assertEqual(stroke.n_nodes,2)
                        self.assertAlmostEqual(stroke.width,expected*size/256,delta=.03)
                        self.assertEqual(stroke.color,ink)
                        self.assertTrue(stroke.source_fit['coverage_not_binary_area'])
                    output=folder/f'reconstructed-{size}.svg'
                    output.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}">'+''.join(
                        f'<path d="{s.d}" fill="none" stroke="rgb{str(s.color)}" stroke-width="{s.width}" stroke-linecap="round"/>' for s in strokes)+'</svg>')
                    render_svg_reference(output,output.with_suffix('.png'),width=size)
                    a=np.asarray(Image.open(png).convert('RGB'),dtype=float)
                    b=np.asarray(Image.open(output.with_suffix('.png')).convert('RGB'),dtype=float)
                    source_ink=np.max(255-a,axis=2)>10
                    self.assertLess(float(np.abs(a-b).max(axis=2)[source_ink].mean()),1)
                    self.assertGreater(_match_percent(output.with_suffix('.png'),png,foreground_only=True),99.5)

    def test_pale_rule_is_not_darkened_by_a_collinear_darker_palette_swatch(self):
        from pathlib import Path
        import tempfile
        from svg_renderer import render_svg_reference
        pale=(147,157,166)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'pale.svg'
            source.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 64"><path d="M18 32 H110" fill="none" stroke="#939da6" stroke-width="2" stroke-linecap="round"/></svg>')
            render_svg_reference(source,source.with_suffix('.png'),width=128)
            _,strokes,_,_=self._extract(Image.open(source.with_suffix('.png')),[pale,(38,57,76)])
            self.assertEqual(len(strokes),1)
            self.assertEqual(strokes[0].color,pale)
            self.assertAlmostEqual(strokes[0].width,2,delta=.03)

    def test_full_pipeline_background_removal_retains_source_fit_and_editable_strokes(self):
        from pathlib import Path
        import tempfile
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        from vector_cleanroom import _match_percent
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory); source=folder/'source.svg'
            source.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">'+''.join(
                f'<path d="M18 {y} H110" fill="none" stroke="#26394c" stroke-width="{width}" stroke-linecap="round"/>'
                for y,width in ((32,1),(63,2),(94,3)))+'</svg>')
            render_svg_reference(source,source.with_suffix('.png'),width=128)
            output=folder/'actual.svg'
            stats=build_clean_base(source.with_suffix('.png'),output,strokes='on',gradients='off',geometry='conservative')
            self.assertEqual(len(stats.stroke_info),3)
            self.assertEqual([s['color'] for s in stats.stroke_info],['#26394c']*3)
            for s,width in zip(stats.stroke_info,(1,2,3)):
                self.assertAlmostEqual(s['width'],width,delta=.03)
                self.assertEqual(s['nodes'],2)
            render_svg_reference(output,output.with_suffix('.png'),width=128)
            details=_match_percent(output.with_suffix('.png'),source.with_suffix('.png'),foreground_only=True,return_details=True)
            self.assertGreater(details['score'],99.5)
            self.assertGreater(details['color_fidelity'],99.5)

    def test_subpixel_diagonal_rule_is_validated_as_an_editable_two_anchor_stroke(self):
        from pathlib import Path
        import tempfile
        from svg_renderer import render_svg_reference
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'diagonal.svg'
            source.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"><path d="M20.3 35.2 L110.3 70.2" fill="none" stroke="#26394c" stroke-width="2.5" stroke-linecap="round"/></svg>')
            render_svg_reference(source,source.with_suffix('.png'),width=128)
            _,strokes,_,_=self._extract(Image.open(source.with_suffix('.png')),[(38,57,76)])
            self.assertEqual(len(strokes),1)
            self.assertEqual(strokes[0].n_nodes,2)
            self.assertAlmostEqual(strokes[0].width,2.5,delta=.05)
            self.assertTrue(strokes[0].source_fit)

    def test_uniform_curve_remains_a_stroke_at_three_raster_resolutions(self):
        from pathlib import Path
        import tempfile
        from svg_renderer import render_svg_reference
        from vector_cleanroom import _match_percent
        ink=(40,110,155)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.svg'
            source.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"><path d="M36 168 C70 32 174 36 220 150" fill="none" stroke="#286e9b" stroke-width="16" stroke-linecap="round"/></svg>')
            for width in (128,256,384):
                with self.subTest(width=width):
                    png=Path(directory)/f'source-{width}.png'
                    render_svg_reference(source,png,width=width)
                    mask,strokes,owned,deferred=self._extract(Image.open(png),[ink])
                    self.assertEqual(len(strokes),1)
                    self.assertLessEqual(strokes[0].n_nodes,10)
                    self.assertEqual(strokes[0].source_fit['observation_model'],'antialiased_native_rgb')
                    self.assertGreaterEqual(int((mask & owned).sum()),int(mask.sum()*.99))
                    result=Path(directory)/f'stroke-{width}.svg'
                    s=strokes[0]
                    result.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {width}"><path d="{s.d}" fill="none" stroke="#286e9b" stroke-width="{s.width}" stroke-linecap="round" stroke-linejoin="round"/></svg>')
                    render_svg_reference(result,result.with_suffix('.png'),width=width)
                    self.assertGreater(_match_percent(result.with_suffix('.png'),png,foreground_only=True),98)

    def test_high_resolution_thick_blob_is_not_invented_as_stroke(self):
        image=Image.new('RGB',(512,512),'white')
        ImageDraw.Draw(image).ellipse((100,120,390,370),fill=(40,110,155))
        mask,strokes,owned,deferred=self._extract(image,[(40,110,155)])
        self.assertEqual(strokes,[])
        self.assertFalse(owned.any())

    @staticmethod
    def _extract(image, palette):
        den = np.asarray(image.convert("RGB"), dtype=np.float32)
        mask = np.abs(den - 255.0).max(axis=2) > 20.0
        strokes, owned, deferred = extract_strokes(
            mask, den, np.asarray(palette, dtype=np.uint8), (255, 255, 255))
        return mask, strokes, owned, deferred

    def test_multicolour_closed_ring_is_deferred_without_gaps(self):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        dark = (10, 75, 35)
        olive = (165, 165, 20)
        draw.arc((16, 16, 112, 112), 90, 270, fill=dark, width=8)
        draw.arc((16, 16, 112, 112), 270, 450, fill=olive, width=8)

        mask, strokes, owned, deferred = self._extract(
            image, (dark, olive))

        self.assertEqual(strokes, [])
        self.assertFalse(owned.any())
        self.assertGreaterEqual(int((deferred & mask).sum()),
                                int(mask.sum() * 0.98))

    def test_multicolour_curved_open_arc_is_deferred_as_one_component(self):
        image = Image.new("RGB", (160, 120), "white")
        draw = ImageDraw.Draw(image)
        dark = (8, 70, 35)
        olive = (170, 165, 15)
        draw.arc((20, 10, 140, 130), 185, 270, fill=dark, width=8)
        draw.arc((20, 10, 140, 130), 270, 355, fill=olive, width=8)

        mask, strokes, owned, deferred = self._extract(
            image, (dark, olive))

        self.assertEqual(strokes, [])
        self.assertFalse(owned.any())
        self.assertGreaterEqual(int((deferred & mask).sum()),
                                int(mask.sum() * 0.98))

    def test_bent_glyph_like_junction_is_not_rounded_into_three_arms(self):
        image = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(image)
        ink = (12, 70, 42)
        # 山-like topology: the side arms turn 90 degrees before reaching the
        # central junction, unlike a genuine straight T/Y/X line diagram.
        draw.line((20, 35, 20, 102, 108, 102, 108, 35),
                  fill=ink, width=12, joint="curve")
        draw.line((64, 18, 64, 102), fill=ink, width=12)

        mask, strokes, owned, deferred = self._extract(image, (ink,))

        self.assertEqual(strokes, [])
        self.assertFalse(owned.any())
        self.assertGreaterEqual(int((deferred & mask).sum()),
                                int(mask.sum() * 0.98))

    def test_two_colour_rule_has_native_proven_editable_strokes_and_intact_seam(self):
        image = Image.new("RGB", (160, 64), "white")
        draw = ImageDraw.Draw(image)
        red = (220, 20, 30)
        blue = (20, 70, 220)
        draw.line((12, 32, 80, 32), fill=red, width=10)
        draw.line((80, 32, 148, 32), fill=blue, width=10)

        mask, strokes, owned, deferred = self._extract(image, (red, blue))

        self.assertEqual(len(strokes), 2)
        self.assertFalse(deferred.any())
        self.assertGreaterEqual(int((owned & mask).sum()), int(mask.sum()*.98))
        self.assertTrue(all(s.source_fit['serialized_geometry_validated'] for s in strokes))
        # Both colours must stay independently selectable, with a source-proven
        # seam; fewer points alone would not prove this editing improvement.
        import tempfile
        from pathlib import Path
        import xml.etree.ElementTree as ET
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'two-colour.png'
            output = source.with_suffix('.svg')
            image.save(source)
            stats = build_clean_base(source, output, strokes='on', gradients='off', geometry='off')
            self.assertEqual(stats.n_strokes, 2)
            root = ET.fromstring(output.read_text(encoding='utf-8'))
            paths = [p for p in root.iter() if p.tag.endswith('path')]
            self.assertGreaterEqual(len(paths), 2)
            self.assertTrue(all(p.attrib.get('d') for p in paths))
            self.assertEqual({p.attrib.get('stroke') for p in paths}, {'#dc141e','#1446dc'})
            preview = source.with_name('actual.png')
            render_svg_reference(output, preview, width=160)
            actual = np.asarray(Image.open(preview).convert('RGB'), dtype=float)
            expected = np.asarray(image, dtype=float)
            self.assertLess(float(np.abs(actual - expected).mean()), 1.0)
            # The colour seam and both rectangle ends remain filled, not white.
            self.assertLess(float(actual[32, 12:149].mean()), 160)
            self.assertTrue(np.all(np.min(actual[32, 12:149], axis=1) < 100))
            np.testing.assert_array_equal(actual,expected)

    def test_butt_and_round_caps_have_native_source_and_endpoint_evidence(self):
        import io
        import resvg_py
        for cap in ('butt', 'round'):
            with self.subTest(cap=cap):
                svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="160" height="80">'
                       f'<path d="M20 40 H140" stroke="#26394c" stroke-width="8" stroke-linecap="{cap}"/></svg>')
                image = Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg_string=svg, background='white')))
                mask, strokes, owned, deferred = self._extract(image, [(38,57,76)])
                self.assertEqual(len(strokes), 1)
                stroke = strokes[0]
                self.assertEqual(stroke.linecap, cap)
                self.assertFalse(stroke.source_fit['cap_ambiguity'])
                self.assertLess(stroke.source_fit['endpoint_max_channel_mae'], 1)
                self.assertGreater(stroke.source_fit['competing_endpoint_max_channel_mae'],
                                   stroke.source_fit['endpoint_max_channel_mae'] + .05)
                self.assertEqual(stroke.source_fit['cap_models_compared'], ['butt','round'])
                self.assertFalse(deferred.any())
                self.assertGreaterEqual(int((owned & mask).sum()), int(mask.sum() * .99))

    def test_native_txy_graphs_keep_shared_editable_arms_and_solid_source_pixels(self):
        import tempfile
        from pathlib import Path
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        shapes={
            't':[(80,110,432,110),(256,110,256,430)],
            'x':[(90,90,422,422),(422,90,90,422)],
            'y':[(100,100,256,256),(412,100,256,256),(256,256,256,440)],
        }
        for name,lines in shapes.items():
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                source=Path(directory)/'original.png'
                image=Image.new('RGB',(512,512),'white')
                for line in lines:ImageDraw.Draw(image).line(line,fill='black',width=12)
                image.save(source)
                output=source.with_suffix('.svg')
                stats=build_clean_base(source,output,strokes='on',gradients='off')
                self.assertEqual(stats.n_strokes,4 if name=='x' else 3)
                self.assertTrue(all(s['nodes']==2 for s in stats.stroke_info))
                self.assertEqual(stats.palette_audit['stroke_reconstruction']['complex_strokes_without_native_cap_proof'],0)
                for stroke in stats.stroke_info:
                    proof=stroke['source_fit']
                    self.assertTrue(proof['serialized_geometry_validated'])
                    self.assertLessEqual(proof['original_solid_core_max_rgb_error'],6)
                    self.assertLessEqual(proof['ink_mean_max_channel_error'],12)
                    self.assertLessEqual(proof['render_evaluations'],120)
                render_svg_reference(output,source.with_name('actual.png'),width=512)
                actual=np.asarray(Image.open(source.with_name('actual.png')).convert('RGB'),dtype=float)
                expected=np.asarray(image,dtype=float)
                ink=np.min(expected,axis=2)<128
                p=np.pad(ink,1)
                core=np.logical_and.reduce([p[dy:dy+512,dx:dx+512] for dy in range(3) for dx in range(3)])
                self.assertLessEqual(float(np.abs(actual-expected).max(2)[core].max()),6)
                union=ink|(np.min(actual,axis=2)<250)
                self.assertLess(float(np.abs(actual-expected).max(2)[union].mean()),12)

    def test_native_graph_proof_rejects_a_hole_that_the_strokes_would_fill(self):
        from stroke_engine import _refine_native_component_strokes
        image=Image.new('RGB',(160,160),'white')
        draw=ImageDraw.Draw(image)
        draw.line((20,40,140,40),fill='black',width=10)
        draw.line((80,40,80,140),fill='black',width=10)
        mask,strokes,_,_=self._extract(image,[(0,0,0)])
        self.assertEqual(len(strokes),3)
        original=np.asarray(image.convert('RGBA')).copy()
        original[38:41,78:81,:3]=255
        result=_refine_native_component_strokes(strokes,np.asarray(image,dtype=float),np.argwhere(mask),original)
        self.assertIsNone(result)

    def test_line_over_flat_fill_has_pigment_proof_and_preserves_actual_fill(self):
        import tempfile
        from pathlib import Path
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        image=Image.new('RGB',(512,512),'white')
        draw=ImageDraw.Draw(image)
        draw.rounded_rectangle((80,120,432,390),radius=35,fill=(245,205,20))
        draw.line((55,300,457,190),fill=(51,51,51),width=8)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'overlap.png';image.save(source)
            output=source.with_suffix('.svg')
            stats=build_clean_base(source,output,strokes='on',gradients='off')
            self.assertEqual(stats.n_strokes,1)
            proof=stats.stroke_info[0]['source_fit']
            self.assertEqual(proof['policy'],'native_discrete_palette_line_occupancy')
            self.assertGreater(proof['source_paint_core_pixels'],100)
            self.assertTrue(proof['underlying_fill_requires_composed_scene_validation'])
            self.assertEqual(stats.stroke_info[0]['color'],'#333333')
            self.assertGreater(stats.palette_audit['stroke_underpaint']['inferred_occluded_working_pixels'],100)
            render_svg_reference(output,source.with_name('actual.png'),width=512)
            actual=np.asarray(Image.open(source.with_name('actual.png')).convert('RGB'),dtype=float)
            original=np.asarray(image,dtype=float)
            ink=np.max(255-original,axis=2)>20
            self.assertLess(float(np.abs(actual-original).max(2)[ink].mean()),3)
            np.testing.assert_array_equal(actual[150:180,180:220],original[150:180,180:220])

    def test_occluded_support_requires_equal_native_paint_on_both_sides(self):
        from stroke_engine import Stroke,infer_occluded_flat_paint
        den=np.full((32,64,3),(220,190,20),dtype=np.uint8)
        den[14:18,8:56]=(30,30,30)
        rgba=np.dstack((den,np.full((32,64),255,np.uint8)))
        visible=np.ones((32,64),bool);fill=visible.copy();fill[14:18,8:56]=False
        stroke=Stroke((30,30,30),4,'M8 16 L56 16',False,48,2,
            source_fit={'policy':'native_discrete_palette_line_occupancy'})
        palette=np.array([(220,190,20),(10,70,100)],np.uint8)
        before=rgba.copy()
        support,_=infer_occluded_flat_paint([stroke],rgba,den,visible,fill,palette)
        self.assertEqual(int(support.sum()),4*48)
        np.testing.assert_array_equal(rgba,before)
        rgba[18:,:,:3]=(10,70,100)
        support,_=infer_occluded_flat_paint([stroke],rgba,den,visible,fill,palette)
        self.assertFalse(support.any())
        rgba[:14,:,:3]=(220,190,20);rgba[18:,:,:3]=(220,190,20);rgba[:,:,3]=100
        support,_=infer_occluded_flat_paint([stroke],rgba,den,visible,fill,palette)
        self.assertFalse(support.any())

    def test_large_one_pixel_line_keeps_native_proof_with_unchanged_render_budget(self):
        import tempfile
        from pathlib import Path
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'large.png'
            image=Image.new('RGB',(3000,3000),'white')
            ImageDraw.Draw(image).line((100,1500,2900,1500),fill='black',width=1)
            image.save(source)
            stats=build_clean_base(source,source.with_suffix('.svg'),max_size=1024,strokes='on',gradients='off')
            self.assertEqual(stats.n_strokes,1)
            detail=stats.stroke_info[0];proof=detail['source_fit']
            self.assertAlmostEqual(detail['width']*3000/1024,1,delta=.001)
            x0,y0,x1,y1=proof['native_roi']
            self.assertLessEqual((x1-x0)*(y1-y0),32768)
            self.assertEqual(proof['source_ink_mean_max_channel_error'],0)
            render_svg_reference(source.with_suffix('.svg'),source.with_name('actual.png'),width=3000)
            actual=np.asarray(Image.open(source.with_name('actual.png')).convert('RGB'))
            np.testing.assert_array_equal(actual,np.asarray(image))

    def test_one_pixel_source_equivalent_cap_tie_stays_editable_and_is_not_claimed_resolved(self):
        import io
        import resvg_py
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="128" height="64">'
               '<path d="M18 32 H110" stroke="#26394c" stroke-width="1" stroke-linecap="round"/></svg>')
        image = Image.open(io.BytesIO(resvg_py.svg_to_bytes(svg_string=svg, background='white')))
        _, strokes, _, deferred = self._extract(image, [(38,57,76),(147,157,166)])
        self.assertEqual(len(strokes), 1)
        stroke = strokes[0]
        self.assertEqual(stroke.linecap, 'round')  # deterministic editable tie-break
        self.assertEqual(stroke.n_nodes, 2)
        self.assertAlmostEqual(stroke.width, 1, delta=.03)
        self.assertTrue(stroke.source_fit['cap_ambiguity'])
        self.assertTrue(stroke.source_fit['equivalent_at_source_resolution'])
        self.assertLess(stroke.source_fit['source_ink_mean_max_channel_error'], 1)
        self.assertFalse(deferred.any())

    def test_downsampled_trace_serializes_the_exact_native_source_fit_without_posthoc_recolor(self):
        import io
        import tempfile
        import xml.etree.ElementTree as ET
        import resvg_py
        from pathlib import Path
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'native.png'
            source.write_bytes(resvg_py.svg_to_bytes(svg_string=(
                '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="160">'
                '<path d="M40 80 H280" stroke="#26394c" stroke-width="8" stroke-linecap="butt"/></svg>'),
                background='white'))
            output = source.with_suffix('.svg')
            stats = build_clean_base(source, output, max_size=128, strokes='on', gradients='off', geometry='off')
            self.assertEqual(stats.n_strokes, 1)
            detail = stats.stroke_info[0]
            self.assertEqual(detail['linecap'], 'butt')
            self.assertEqual(detail['source_fit']['source_dimensions'], [320,160])
            self.assertTrue(detail['source_fit']['serialized_geometry_validated'])
            self.assertEqual(detail['color'], '#26394c')
            self.assertAlmostEqual(detail['width'], 3.2, delta=.001)
            root = ET.fromstring(output.read_text(encoding='utf-8'))
            stroke = next(p for p in root.iter() if p.attrib.get('id') == 'stroke-1')
            self.assertEqual(stroke.attrib['stroke-linecap'], detail['linecap'])
            self.assertEqual(stroke.attrib['stroke'], detail['color'])
            self.assertEqual(float(stroke.attrib['stroke-width']), detail['width'])
            preview = source.with_name('actual.png')
            render_svg_reference(output, preview, width=320)
            a = np.asarray(Image.open(source).convert('RGB'), dtype=float)
            b = np.asarray(Image.open(preview).convert('RGB'), dtype=float)
            self.assertEqual(float(np.abs(a-b).max()), 0.)

    def test_neighbouring_bars_and_unmodelled_caps_defer_whole_component(self):
        for kind in ('neighbours', 'slanted_cap'):
            with self.subTest(kind=kind):
                image = Image.new('RGB', (160,80), 'white')
                draw = ImageDraw.Draw(image)
                if kind == 'neighbours':
                    draw.rectangle((20,20,139,27), fill=(38,57,76))
                    draw.rectangle((20,30,139,37), fill=(38,57,76))
                else:
                    draw.polygon([(20,28),(139,28),(125,39),(20,39)], fill=(38,57,76))
                mask, strokes, owned, deferred = self._extract(image, [(38,57,76)])
                self.assertEqual(strokes, [])
                self.assertFalse(owned.any())
                self.assertGreaterEqual(int((mask & deferred).sum()), int(mask.sum() * .98))

    def test_fit_budget_exhaustion_never_reinstates_unverified_round_fallback(self):
        # Each isolated rectangular source has a proven butt cap; only the
        # first 24 get a search. The remaining two must stay complete fills.
        image = Image.new('RGB', (256,256), 'white')
        draw = ImageDraw.Draw(image)
        for y in range(8, 242, 9):
            draw.rectangle((20,y,119,y+1), fill=(38,57,76))
        den = np.asarray(image, dtype=np.float32)
        mask = np.abs(den - 255).max(axis=2) > 20
        audit = {}
        strokes, owned, deferred = extract_strokes(mask, den, np.array([(38,57,76)]), audit=audit)
        self.assertEqual(len(strokes), 24)
        self.assertTrue(all(s.linecap == 'butt' and s.source_fit for s in strokes))
        self.assertTrue(deferred.any())
        self.assertEqual(int((mask & ~(owned | deferred)).sum()), 0)
        self.assertEqual(int((owned & deferred & mask).sum()), 0)
        self.assertEqual(audit['deferred_component_reasons']['native_straight_cap_search_budget_exhausted'], 2)

    def test_private_e3_palette_outlines_do_not_become_round_strokes_at_either_resolution(self):
        from pathlib import Path
        import hashlib
        from clean_base import _produce_pre_gradient_state
        root = Path(__file__).resolve().parents[4]
        fixture = root/'work/cr-1005101832/exp/e3_gaps'
        if not (fixture/'in_1024/gaps.png').is_file():
            self.skipTest('Private review E3 fixture is not distributed')
        for size in (1024, 1254):
            with self.subTest(size=size):
                source = fixture/f'in_{size}/gaps.png'
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                state = _produce_pre_gradient_state(source, forced_colors=0,
                    white_threshold=245, background='auto', max_size=1024,
                    strokes='on', checkpoint=lambda *args: None)
                self.assertEqual(state['stroke_list'], [])
                audit = state['palette_audit']['stroke_reconstruction']
                self.assertGreater(audit['palette_fragment_candidate_pixels_deferred'], 0)
                self.assertGreater(audit['deferred_component_reasons']['native_straight_caps_not_source_validated'], 0)
                # Ownership rejection leaves the entire actual ink available
                # to fill tracing, including straight bars and large blocks.
                dark = np.min(state['den'], axis=2) < 100
                self.assertEqual(int((dark & state['visible'] & ~state['vis_fill']).sum()), 0)
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

    def test_closed_native_outline_has_no_false_missing_cap_failure(self):
        import tempfile
        from pathlib import Path
        from clean_base import build_clean_base
        yy, xx = np.ogrid[:128, :128]
        image = np.full((128,128,4), 255, dtype=np.uint8)
        image[np.abs(np.sqrt((xx-64)**2 + (yy-64)**2) - 40) <= 4, :3] = 0
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'ring.png'
            Image.fromarray(image).save(source)
            stats = build_clean_base(source, source.with_suffix('.svg'), max_size=64,
                strokes='on', gradients='off', geometry='conservative')
            self.assertGreater(stats.n_strokes, 0)
            self.assertTrue(all(s['closed'] for s in stats.stroke_info))
            audit = stats.palette_audit['stroke_reconstruction']
            self.assertEqual(audit['complex_strokes_without_native_cap_proof'], 0)
            self.assertEqual(audit['closed_strokes_without_native_geometry_fit'], stats.n_strokes)

    def test_low_contrast_gray_is_discovered_but_still_requires_native_cap_proof(self):
        import tempfile
        from pathlib import Path
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        image=Image.new('RGB',(512,512),'white')
        ImageDraw.Draw(image).line((56,256,456,256),fill=(221,221,221),width=12)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'gray.png';image.save(source)
            vector=source.with_suffix('.svg')
            stats=build_clean_base(source,vector,strokes='on',gradients='off')
            self.assertEqual(stats.n_strokes,1)
            detail=stats.stroke_info[0]
            self.assertEqual(detail['color'],'#dddddd')
            self.assertEqual(detail['linecap'],'butt')
            self.assertTrue(detail['source_fit']['serialized_geometry_validated'])
            render_svg_reference(vector,source.with_name('result.png'),width=512)
            actual=np.asarray(Image.open(source.with_name('result.png')).convert('RGB'))
            np.testing.assert_allclose(actual,np.asarray(image),atol=1)

    def test_soft_alpha_keeps_editable_width_color_and_serialized_alpha_proof(self):
        import tempfile
        from pathlib import Path
        from clean_base import build_clean_base
        from svg_renderer import render_svg_reference
        for color in ((0,0,0),(30,90,160)):
            with self.subTest(color=color), tempfile.TemporaryDirectory() as directory:
                image=Image.new('RGBA',(512,512),(0,0,0,0))
                ImageDraw.Draw(image).line((56,256,456,256),fill=(*color,100),width=12)
                source=Path(directory)/'alpha.png';image.save(source)
                vector=source.with_suffix('.svg')
                stats=build_clean_base(source,vector,strokes='on',gradients='off')
                self.assertEqual(stats.n_strokes,1)
                detail=stats.stroke_info[0]
                self.assertEqual(detail['color'],'#{:02x}{:02x}{:02x}'.format(*color))
                self.assertAlmostEqual(detail['opacity'],100/255,delta=.0001)
                self.assertEqual(detail['linecap'],'butt')
                self.assertTrue(detail['source_fit']['original_alpha_jointly_validated'])
                self.assertEqual(detail['source_fit']['serialized_opacity'],detail['opacity'])
                preview=source.with_name('result.png')
                render_svg_reference(vector,preview,width=512,background=None)
                actual=np.asarray(Image.open(preview).convert('RGBA'))
                np.testing.assert_allclose(actual[:,:,3],np.asarray(image)[:,:,3],atol=1)

    def test_short_curved_glyph_fragment_stays_a_fill(self):
        image = Image.new("RGB", (96, 96), "white")
        draw = ImageDraw.Draw(image)
        ink = (12, 70, 42)
        draw.arc((28, 28, 68, 68), 35, 155, fill=ink, width=9)

        mask, strokes, owned, deferred = self._extract(image, (ink,))

        self.assertEqual(strokes, [])
        self.assertFalse(owned.any())
        # Whether rejected before or at the explicit short-curve guard, all
        # source pixels remain available to the fill tracer.
        self.assertEqual(int((mask & ~owned).sum()), int(mask.sum()))

    def test_long_single_colour_arc_remains_an_editable_stroke(self):
        image = Image.new("RGB", (180, 180), "white")
        draw = ImageDraw.Draw(image)
        ink = (12, 70, 42)
        draw.arc((20, 20, 160, 160), 15, 215, fill=ink, width=8)

        mask, strokes, owned, deferred = self._extract(image, (ink,))

        self.assertGreaterEqual(len(strokes), 1)
        self.assertEqual(strokes[0].linecap,'butt')
        proof=strokes[0].source_fit
        self.assertEqual(proof['observation_model'],'two_opaque_paints_pixel_occupancy')
        self.assertTrue(all(p['occupancy_iou']>=.95 for p in proof['hard_binary_endpoint_checks']))
        self.assertFalse(deferred.any())
        self.assertGreaterEqual(int((owned & mask).sum()),
                                int(mask.sum() * 0.98))

    def test_binary_arc_cannot_hide_hole_notch_or_local_deformation_in_global_iou(self):
        from stroke_engine import _validate_native_open_curve
        image=Image.new('RGB',(180,180),'white')
        ImageDraw.Draw(image).arc((20,20,160,160),15,215,fill=(12,70,42),width=8)
        mask,strokes,_,_=self._extract(image,[(12,70,42)])
        self.assertEqual(len(strokes),1)
        source=np.asarray(image.convert('RGBA'))
        self.assertTrue(np.all(source[155:158,89:92,0]<100))
        for kind in ('hole','notch','deformation'):
            with self.subTest(kind=kind):
                changed=source.copy()
                if kind=='hole':changed[155:158,89:92,:3]=255
                elif kind=='notch':changed[154:162,89:92,:3]=255
                else:changed[156:165,87:95,:3]=(12,70,42)
                # Each change is tiny globally, but removing a hole/notch or
                # flattening a local bulge must fail the native shape proof.
                self.assertLess(float(np.abs(changed.astype(float)-source).mean()),.2)
                proposal=_validate_native_open_curve(strokes[0],np.asarray(image,dtype=float),np.argwhere(mask),changed)
                self.assertIsNone(proposal)

    def test_native_elbow_restores_straight_arms_corner_and_butt_tips(self):
        import re
        from stroke_engine import _render_native_strokes
        image=Image.new('RGB',(512,512),'white')
        ImageDraw.Draw(image).line(((80,90),(250,90),(250,400)),fill='black',width=14,joint='curve')
        for rotation in (0,1,2,3):
            with self.subTest(rotation=rotation):
                source=Image.fromarray(np.rot90(np.asarray(image),rotation).copy())
                mask,strokes,owned,deferred=self._extract(source,[(0,0,0)])
                self.assertEqual(len(strokes),1)
                stroke=strokes[0]
                self.assertEqual(stroke.n_nodes,3)
                self.assertEqual(stroke.linecap,'butt')
                self.assertAlmostEqual(stroke.width,14,places=3)
                self.assertEqual(re.findall('[A-Za-z]',stroke.d),['M','L','L'])
                points=np.array([float(v) for v in re.findall(r'[-+]?(?:\d*\.\d+|\d+)',stroke.d)]).reshape(3,2)
                axes=np.diff(points,axis=0)
                self.assertLess(abs(float(axes[0]@axes[1])),1e-9)
                self.assertTrue(np.all(np.min(np.abs(axes),axis=1)==0))
                self.assertTrue(stroke.source_fit['native_axis_aligned_elbow_proposal_used'])
                self.assertEqual(stroke.source_fit['original_solid_core_max_rgb_error'],0)
                self.assertTrue(all(p['occupancy_iou']>=.95 for p in stroke.source_fit['hard_binary_endpoint_checks']))
                preview=_render_native_strokes(strokes,[0,0,512,512],(512,512),(512,512))
                occupancy=preview[:,:,3]>=128
                self.assertGreaterEqual(float((occupancy&mask).sum()/(occupancy|mask).sum()),.99)
                self.assertGreaterEqual(int((owned&mask).sum()),int(mask.sum()*.98))
                self.assertFalse(deferred.any())

    def test_native_elbow_proposal_does_not_erase_hole_notch_or_wrong_tip(self):
        from stroke_engine import _validate_native_open_curve
        image=Image.new('RGB',(512,512),'white')
        ImageDraw.Draw(image).line(((80,90),(250,90),(250,400)),fill='black',width=14,joint='curve')
        mask,strokes,_,_=self._extract(image,[(0,0,0)])
        self.assertEqual(len(strokes),1)
        source=np.asarray(image.convert('RGBA'))
        for defect in ('hole','notch','wrong_tip'):
            with self.subTest(defect=defect):
                changed=source.copy()
                if defect=='hole':changed[89:92,140:143,:3]=255
                elif defect=='notch':changed[84:92,140:143,:3]=255
                else:changed[397:401,245:250,:3]=255
                candidate=_validate_native_open_curve(strokes[0],np.asarray(image,dtype=float),np.argwhere(mask),changed)
                self.assertIsNone(candidate)

    def test_private_tea_unproven_curve_cannot_trigger_stroke_palette_reestimation(self):
        from pathlib import Path
        from clean_base import _produce_pre_gradient_state
        source=Path(__file__).resolve().parents[4]/'work/p3/tea-resumed-full/result_tea/source_original.png'
        if not source.is_file():self.skipTest('Private tea source is not distributed')
        state=_produce_pre_gradient_state(source,forced_colors=0,white_threshold=220,
            background='auto',max_size=1024,strokes='on',checkpoint=lambda *a:None)
        self.assertEqual(state['stroke_list'],[])
        audit=state['palette_audit']['stroke_reconstruction']
        self.assertGreater(audit['deferred_component_reasons']['native_open_component_not_source_validated'],0)
        self.assertNotIn('fill_after_strokes',state['palette_audit'])


if __name__ == "__main__":
    unittest.main()
