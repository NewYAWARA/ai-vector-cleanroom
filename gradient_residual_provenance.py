"""Separate pre-existing trace scraps from changes actually caused by gradients.

This is exact local rendering evidence, not a small-error exemption. A palette
colour appearing in a distant gradient does not make that gradient responsible
for every old antialias pixel of the colour across the canvas.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image


def unchanged_remote_residual_support(before_rgba, after_rgba, residual, owned):
    """Return residual pixels with an exact unchanged 3x3 RGBA neighbourhood."""
    before, after = np.asarray(before_rgba), np.asarray(after_rgba)
    residual, owned = np.asarray(residual, dtype=bool), np.asarray(owned, dtype=bool)
    if before.dtype != np.uint8 or after.dtype != np.uint8 or before.shape != after.shape:
        raise ValueError("gradient baseline render mismatch")
    if before.ndim != 3 or before.shape[2] != 4 or residual.shape != before.shape[:2] or owned.shape != residual.shape:
        raise ValueError("gradient baseline support mismatch")
    height, width = residual.shape
    blocked = np.any(before != after, axis=2) | owned
    padded = np.pad(blocked, 1, constant_values=True)
    neighbourhood = np.logical_or.reduce([padded[y:y+height, x:x+width]
                                          for y in range(3) for x in range(3)])
    return residual & ~neighbourhood


def prove_preexisting_residuals(raw_svg, baseline_rgba, residual, owned, *, visible, source_rgb):
    """Re-run the ordinary trace and prove no local paint/alpha change occurred.

    The baseline uses the exact same VTracer settings and opaque routing
    surround as the gradient candidate. Original source error is recorded;
    an unchanged pre-existing error is not claimed to have been repaired.
    """
    import vtracer
    from svg_renderer import render_svg_reference
    from trace_component_recovery import recover_missing_source_components
    baseline = np.asarray(baseline_rgba)
    source = np.asarray(source_rgb)
    if baseline.dtype != np.uint8 or source.shape != baseline[:, :, :3].shape:
        raise ValueError("gradient baseline raster mismatch")
    height, width = baseline.shape[:2]
    with tempfile.TemporaryDirectory(prefix="aivc-gradient-baseline-") as temp:
        folder = Path(temp)
        png, svg = folder/"baseline.png", folder/"baseline.svg"
        Image.fromarray(baseline, "RGBA").save(png)
        vtracer.convert_image_to_svg_py(
            str(png), str(svg), colormode="color", hierarchical="cutout", mode="spline",
            filter_speckle=2, color_precision=8, layer_difference=0, corner_threshold=58,
            length_threshold=5.0, splice_threshold=45, path_precision=6)
        before_text, _ = recover_missing_source_components(
            svg.read_text(encoding="utf8"), baseline, excluded_mask=~np.asarray(visible, dtype=bool))
        arrays, provenance = {}, {}
        for name, text in (("before", before_text), ("after", raw_svg)):
            path = folder/(name+".svg")
            path.write_text(text, encoding="utf8")
            provenance[name] = render_svg_reference(path, path.with_suffix(".png"), width, background=None)
            with Image.open(path.with_suffix(".png")) as image:
                arrays[name] = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        if arrays["before"].shape != (height, width, 4):
            raise ValueError("gradient baseline canvas alignment failed")
        proven = unchanged_remote_residual_support(arrays["before"], arrays["after"], residual, owned)
        count = int(proven.sum())
        errors = {}
        for name, arr in arrays.items():
            alpha = arr[:, :, 3, None].astype(float)/255
            composed = arr[:, :, :3]*alpha + 255*(1-alpha)
            errors[name] = float(np.abs(composed-source)[proven].mean()) if count else None
        return proven, {
            "schema": "ai-vector-cleanroom.gradient-residual-provenance/v1",
            "status": "exact_unchanged_remote_support" if count else "no_unchanged_support",
            "baseline_raster_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
            "source_rgb_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
            "source_size": [width, height], "residual_pixels_examined": int(np.count_nonzero(residual)),
            "unchanged_residual_pixels": count, "unchanged_support_sha256": hashlib.sha256(proven.tobytes()).hexdigest(),
            "halo_radius_source_pixels": 1, "rgba_maximum_difference_in_all_proven_halos": 0,
            "source_rgb_mae_on_proven_support": errors, "renderer_provenance": provenance,
            "scope": "preexisting_trace_error_is_not_new_gradient_damage",
            "source_detail_recovery_claimed": False,
        }


def prove_unchanged_scene_residuals(before_svg, after_svg, residual, original_source_path,
                                  processed_rgba):
    """Compare the complete scene with/without only the withdrawn gradients.

    Raw tracing uses placeholder paints and is insufficient near another
    gradient. This transaction instead requires exact RGBA equality throughout
    every residual's halo at both original and processed source resolutions.
    The actual changed area must also not regress against either source.
    """
    from svg_renderer import render_svg_reference
    residual = np.asarray(residual, dtype=bool)
    processed = np.asarray(processed_rgba, dtype=np.uint8)
    if processed.shape != (*residual.shape, 4) or not residual.any():
        raise ValueError("complete_scene_residual_support_required")
    with Image.open(original_source_path) as image:
        original = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if max(original.shape[:2]) > 2048:
        raise ValueError("complete_scene_original_resolution_budget")
    from PIL import ImageFilter
    measurements = []
    with tempfile.TemporaryDirectory(prefix="aivc-gradient-scene-proof-") as temp:
        folder = Path(temp)
        paths = {}
        for name, payload in (("before", before_svg), ("after", after_svg)):
            paths[name] = folder/(name+".svg")
            paths[name].write_text(payload, encoding="utf8")
        for label, source in (("unmodified_input", original), ("processed_reference", processed)):
            height, width = source.shape[:2]
            support = np.asarray(Image.fromarray(residual).resize((width, height), Image.Resampling.NEAREST))
            halo = np.asarray(Image.fromarray(support.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(3))) > 0
            rendered, provenance = {}, {}
            for name in ("before", "after"):
                png = folder/(label+"-"+name+".png")
                provenance[name] = render_svg_reference(paths[name], png, width, background=None)
                with Image.open(png) as image:
                    rendered[name] = np.asarray(image.convert("RGBA"), dtype=np.uint8)
                if rendered[name].shape != source.shape:
                    raise ValueError("complete_scene_source_alignment_failed")
            if np.any(rendered["before"][halo] != rendered["after"][halo]):
                raise ValueError(label+"_residual_halo_rgba_changed")
            changes = np.any(rendered["before"] != rendered["after"], axis=2)
            changed_halo = np.asarray(Image.fromarray(changes.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(3))) > 0
            def composite(rgba):
                alpha = rgba[:, :, 3, None].astype(float)/255
                return rgba[:, :, :3]*alpha + 255*(1-alpha)
            target = composite(source)
            errors = {name: np.abs(composite(image)-target).mean(axis=2) for name, image in rendered.items()}
            metrics = {name: {"global_rgb_mae": float(error.mean()),
                              "changed_area_rgb_mae": float(error[changed_halo].mean()) if changes.any() else 0,
                              "changed_area_rgb_p95": float(np.percentile(error[changed_halo], 95)) if changes.any() else 0}
                       for name, error in errors.items()}
            if any(metrics["after"][key] > metrics["before"][key] + 1e-9
                   for key in ("global_rgb_mae", "changed_area_rgb_mae", "changed_area_rgb_p95")):
                raise ValueError(label+"_complete_scene_source_fidelity_worse")
            measurements.append({"source_reference_kind": label, "source_size": [width, height],
                                 "source_rgba_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                                 "residual_halo_pixels": int(halo.sum()), "residual_halo_maximum_rgba_difference": 0,
                                 "changed_pixels": int(changes.sum()), "measurements": metrics,
                                 "renderer_provenance": provenance})
    return {"schema": "ai-vector-cleanroom.gradient-residual-scene-transaction/v1",
            "status": "verified_no_new_residual_damage", "residual_pixels": int(residual.sum()),
            "before_svg_sha256": hashlib.sha256(before_svg.encode()).hexdigest(),
            "after_svg_sha256": hashlib.sha256(after_svg.encode()).hexdigest(),
            "measurements": measurements,
            "scope": "same_complete_scene_counterfactual_not_placeholder_trace",
            "preexisting_source_errors_repaired": False}
