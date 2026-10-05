"""Source-space gradient object proposal and model fitting.

This module is deliberately independent from the production tracer.  It does
not mutate an SVG, quantise colours, infer a background, or commit geometry.
Callers provide one explicit candidate ownership mask and receive a JSON-safe
proposal that can be rendered and transactionally validated by a later stage.

The fitter compares a solid baseline with two native-SVG-compatible models:

* an arbitrary-angle bounded two-to-five-stop ``linearGradient``;
* a rotated elliptical ``radialGradient`` represented by centre, radii and
  rotation (an SVG emitter can encode the ellipse with ``gradientTransform``).

All selection decisions use a deterministic coordinate-hash holdout.  A hard
internal colour boundary, a weak colour span, a poor tail error, or uncertain
radial geometry fails closed: the returned status is ``skipped`` and no model
is proposed.  Only NumPy is required.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

import numpy as np


SCHEMA = "ai-vector-cleanroom.gradient-object-proposal/v2"
ERROR_METRIC = "oklab_delta_e_100"


def _round(value: float, digits: int = 6) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite value cannot be serialised")
    return round(value, digits)


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert 0..255 sRGB values to OKLab without optional dependencies."""

    srgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 255.0) / 255.0
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    red, green, blue = np.moveaxis(linear, -1, 0)
    ll = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    mm = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    ss = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    ll, mm, ss = np.cbrt(ll), np.cbrt(mm), np.cbrt(ss)
    return np.stack(
        (
            0.2104542553 * ll + 0.7936177850 * mm - 0.0040720468 * ss,
            1.9779984951 * ll - 2.4285922050 * mm + 0.4505937099 * ss,
            0.0259040371 * ll + 0.7827717662 * mm - 0.8086757660 * ss,
        ),
        axis=-1,
    )


def _delta_e(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    actual_lab = _srgb_to_oklab(actual)
    predicted_lab = _srgb_to_oklab(predicted)
    # OKLab is approximately unit-scaled.  Multiplying by 100 produces a
    # designer-friendly magnitude similar to familiar Delta-E values.
    return 100.0 * np.linalg.norm(actual_lab - predicted_lab, axis=-1)


def _error_stats(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    error = _delta_e(actual, predicted)
    if error.size == 0:
        return {
            "samples": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    return {
        "samples": int(error.size),
        "mean": _round(error.mean(), 4),
        "median": _round(np.median(error), 4),
        "p90": _round(np.quantile(error, 0.90), 4),
        "p99": _round(np.quantile(error, 0.99), 4),
        "max": _round(error.max(), 4),
    }


def _hex(rgb: np.ndarray) -> str:
    values = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return "#{:02x}{:02x}{:02x}".format(*(int(value) for value in values))


def _stops(offsets: np.ndarray, colours: np.ndarray) -> list[dict[str, Any]]:
    """Return JSON-safe native SVG stops.

    Keeping offsets explicit is important: a real logo gradient is often a
    three-to-five-stop colour curve, not a two-colour plane.  The number of
    stops is still capped by the caller so this cannot turn into pixel-colour
    tracing under another name.
    """

    result = []
    for offset, colour in zip(offsets, colours):
        values = np.clip(np.rint(colour), 0, 255).astype(np.uint8)
        result.append(
            {
                "offset": _round(offset, 6),
                "color": _hex(values),
                "rgb": [int(value) for value in values],
            }
        )
    return result


def _stop_design(t: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation matrix for SVG colour stops."""

    values = _clip01(t).reshape(-1)
    offsets = np.asarray(offsets, dtype=np.float64).reshape(-1)
    if len(offsets) < 2 or offsets[0] != 0.0 or offsets[-1] != 1.0:
        raise ValueError("gradient stop offsets must start at 0 and end at 1")
    if np.any(np.diff(offsets) <= 1e-6):
        raise ValueError("gradient stop offsets must be strictly increasing")
    segment = np.searchsorted(offsets, values, side="right") - 1
    segment = np.clip(segment, 0, len(offsets) - 2)
    left = offsets[segment]
    right = offsets[segment + 1]
    fraction = (values - left) / np.maximum(1e-12, right - left)
    design = np.zeros((len(values), len(offsets)), dtype=np.float64)
    rows = np.arange(len(values))
    design[rows, segment] = 1.0 - fraction
    design[rows, segment + 1] = fraction
    return design


def _render_stops(
    t: np.ndarray, offsets: np.ndarray, colours: np.ndarray
) -> np.ndarray:
    return _stop_design(t, offsets) @ np.asarray(colours, dtype=np.float64)


def _fit_stops(
    t: np.ndarray, colours: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    design = _stop_design(t, offsets)
    coefficients, *_ = np.linalg.lstsq(design, colours, rcond=None)
    return np.clip(coefficients, 0.0, 255.0)


def _gradient_stop_offset_families(
    t: np.ndarray,
    colours: np.ndarray,
    training: np.ndarray,
    *,
    maximum_stops: int,
) -> list[np.ndarray]:
    """Propose a small deterministic family of 2..N-stop colour curves.

    A 1-D median colour profile is reduced with a Ramer-Douglas-Peucker-like
    farthest-residual rule.  Uniform alternatives keep the fit stable when a
    mask has very uneven occupancy along the gradient axis.  This is bounded
    model fitting (at most five stops), never per-band or per-pixel tracing.
    """

    maximum_stops = max(2, min(5, int(maximum_stops)))
    train_t = _clip01(t[training])
    train_colours = np.asarray(colours[training], dtype=np.float64)
    bin_count = 33
    bin_index = np.minimum(bin_count - 1, (train_t * bin_count).astype(int))
    # Group the training rows once.  The former loop built a full-length
    # boolean mask for every one of the 33 bins, repeatedly scanning the same
    # arrays.  Stable sorting preserves the original row order inside each
    # bin, so every median below consumes exactly the same values in exactly
    # the same order as ``train_t[bin_index == index]`` did.
    grouped_members = np.argsort(bin_index, kind="stable")
    bin_counts = np.bincount(bin_index, minlength=bin_count)
    bin_boundaries = np.empty(bin_count + 1, dtype=np.intp)
    bin_boundaries[0] = 0
    np.cumsum(bin_counts, out=bin_boundaries[1:])
    profile_t = []
    profile_colour = []
    for index in range(bin_count):
        start = int(bin_boundaries[index])
        stop = int(bin_boundaries[index + 1])
        if stop - start < 3:
            continue
        members = grouped_members[start:stop]
        profile_t.append(float(np.median(train_t[members])))
        profile_colour.append(np.median(train_colours[members], axis=0))

    families: list[np.ndarray] = [np.asarray((0.0, 1.0), dtype=np.float64)]
    if len(profile_t) >= 3:
        profile_t_array = np.asarray(profile_t, dtype=np.float64)
        profile_rgb = np.asarray(profile_colour, dtype=np.float64)
        selected = [0, len(profile_t_array) - 1]
        while len(selected) < maximum_stops:
            selected.sort()
            best_index = None
            best_error = -1.0
            for left_index, right_index in zip(selected[:-1], selected[1:]):
                if right_index <= left_index + 1:
                    continue
                left_t = profile_t_array[left_index]
                right_t = profile_t_array[right_index]
                fraction = (
                    (profile_t_array[left_index + 1 : right_index] - left_t)
                    / max(1e-9, right_t - left_t)
                )[:, None]
                prediction = (
                    profile_rgb[left_index][None, :] * (1.0 - fraction)
                    + profile_rgb[right_index][None, :] * fraction
                )
                errors = _delta_e(
                    profile_rgb[left_index + 1 : right_index], prediction
                )
                local = int(np.argmax(errors))
                value = float(errors[local])
                candidate_index = left_index + 1 + local
                if value > best_error + 1e-12 or (
                    abs(value - best_error) <= 1e-12
                    and (best_index is None or candidate_index < best_index)
                ):
                    best_error = value
                    best_index = candidate_index
            if best_index is None:
                break
            selected.append(best_index)
            offsets = [0.0]
            offsets.extend(float(profile_t_array[i]) for i in sorted(selected)[1:-1])
            offsets.append(1.0)
            offsets_array = np.asarray(offsets, dtype=np.float64)
            if np.all(np.diff(offsets_array) >= 0.04):
                families.append(offsets_array)

    for stop_count in range(3, maximum_stops + 1):
        families.append(np.linspace(0.0, 1.0, stop_count, dtype=np.float64))

    unique: dict[tuple[float, ...], np.ndarray] = {}
    for offsets in families:
        key = tuple(round(float(value), 5) for value in offsets)
        unique.setdefault(key, offsets)
    return sorted(unique.values(), key=lambda item: (len(item), tuple(item)))


def _erode_once(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    result = np.ones((height, width), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            result &= padded[dy : dy + height, dx : dx + width]
    return result


def _coordinate_hash(
    xs: np.ndarray, ys: np.ndarray, validation_seed: int = 0
) -> np.ndarray:
    """Return deterministic, well-mixed uint64 keys for pixel coordinates."""

    x = xs.astype(np.uint64)
    y = ys.astype(np.uint64)
    value = x * np.uint64(0x9E3779B185EBCA87)
    value ^= y * np.uint64(0xC2B2AE3D27D4EB4F)
    value ^= (x + np.uint64(0x165667B19E3779F9)) * (
        y + np.uint64(0x85EBCA77C2B2AE63)
    )
    salt = np.uint64(
        (int(validation_seed) * 0xD6E8FEB86659FD93 + 0xA0761D6478BD642F)
        & ((1 << 64) - 1)
    )
    value ^= salt
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value


def _stable_bounded_argsort(values: np.ndarray, limit: int) -> np.ndarray:
    """Return the first ``limit`` indices of a stable argsort exactly.

    A full stable sort is needlessly expensive when gradient fitting only
    consumes a bounded sample from a very large ownership mask.  Partitioning
    finds the cut value in linear time; the explicit equal-value handling
    retains the old stable tie order before the bounded subset is sorted.
    """

    keys = np.asarray(values)
    count = len(keys)
    limit = max(0, min(int(limit), count))
    if limit == 0:
        return np.empty(0, dtype=np.intp)
    if limit == count:
        return np.argsort(keys, kind="stable")
    threshold = np.partition(keys, limit - 1)[limit - 1]
    lower = np.flatnonzero(keys < threshold)
    equal = np.flatnonzero(keys == threshold)
    selected = np.concatenate((lower, equal[: limit - len(lower)]))
    bounded_order = np.argsort(keys[selected], kind="stable")
    return selected[bounded_order]


def _sample_and_split(
    ys: np.ndarray,
    xs: np.ndarray,
    max_samples: int,
    heldout_fraction: float,
    validation_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_keys = _coordinate_hash(xs, ys, validation_seed)
    sample_order = _stable_bounded_argsort(sample_keys, int(max_samples))
    sample_count = len(sample_order)
    heldout_count = max(16, int(round(sample_count * heldout_fraction)))
    heldout_count = min(heldout_count, max(1, sample_count // 2))
    # A second independently salted coordinate hash separates sampling from
    # validation assignment.  This prevents the heldout set being merely the
    # lowest-hash prefix of an already lowest-hash-bounded sample.
    split_keys = _coordinate_hash(
        xs[sample_order], ys[sample_order],
        validation_seed ^ 0x6A09E667F3BCC909,
    )
    split_order = np.argsort(split_keys, kind="stable")
    heldout = split_order[:heldout_count]
    training = split_order[heldout_count:]
    return sample_order, training, heldout


def _normalise_inputs(
    rgb: Any,
    candidate_mask: Any,
    alpha: Any,
    label_map: Any,
    alpha_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    pixels = np.asarray(rgb)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("rgb must be an HxWx3 or HxWx4 numeric array")
    if not np.issubdtype(pixels.dtype, np.number):
        raise TypeError("rgb must contain numeric values")
    pixels = pixels.astype(np.float64)
    if not np.isfinite(pixels).all():
        raise ValueError("rgb must contain only finite values")
    if pixels.shape[2] == 4:
        if alpha is None:
            alpha = pixels[:, :, 3]
        pixels = pixels[:, :, :3]
    pixels = np.clip(pixels, 0.0, 255.0)

    mask = np.asarray(candidate_mask)
    if mask.shape != pixels.shape[:2]:
        raise ValueError("candidate_mask must match the RGB height and width")
    if mask.ndim != 2:
        raise ValueError("candidate_mask must be two-dimensional")
    mask = mask.astype(bool)

    if alpha is None:
        alpha_values = np.full(mask.shape, 255.0, dtype=np.float64)
    else:
        alpha_values = np.asarray(alpha)
        if alpha_values.shape != mask.shape or alpha_values.ndim != 2:
            raise ValueError("alpha must be a two-dimensional array matching RGB")
        if not np.issubdtype(alpha_values.dtype, np.number):
            raise TypeError("alpha must contain numeric values")
        alpha_values = alpha_values.astype(np.float64)
        if not np.isfinite(alpha_values).all():
            raise ValueError("alpha must contain only finite values")
        if alpha_values.size and alpha_values.max() <= 1.0:
            alpha_values *= 255.0
        alpha_values = np.clip(alpha_values, 0.0, 255.0)

    labels = None
    if label_map is not None:
        labels = np.asarray(label_map)
        if labels.shape != mask.shape or labels.ndim != 2:
            raise ValueError("label_map must be two-dimensional and match RGB")

    mask &= alpha_values >= float(alpha_threshold)
    return pixels, alpha_values, mask, labels


def _mask_stats(
    mask: np.ndarray,
    fit_mask: np.ndarray,
    alpha: np.ndarray,
    labels: Optional[np.ndarray],
) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        bbox = None
    else:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    digest = hashlib.sha256()
    digest.update(str(mask.shape).encode("ascii"))
    digest.update(np.packbits(mask.reshape(-1), bitorder="little").tobytes())
    selected_alpha = alpha[mask]
    label_count = int(len(np.unique(labels[mask]))) if labels is not None and mask.any() else 0
    return {
        "image_size": [int(mask.shape[1]), int(mask.shape[0])],
        "pixels": int(mask.sum()),
        "interior_fit_pixels": int(fit_mask.sum()),
        "bbox_xywh": bbox,
        "mask_sha256": digest.hexdigest(),
        "alpha_min": _round(selected_alpha.min(), 3) if selected_alpha.size else None,
        "alpha_mean": _round(selected_alpha.mean(), 3) if selected_alpha.size else None,
        "alpha_max": _round(selected_alpha.max(), 3) if selected_alpha.size else None,
        "label_count": label_count,
    }


def _internal_edge_stats(
    rgb: np.ndarray,
    mask: np.ndarray,
    labels: Optional[np.ndarray],
) -> dict[str, Any]:
    actual_a = []
    actual_b = []
    label_a = []
    label_b = []
    for left, right, valid, lab_left, lab_right in (
        (
            rgb[:, :-1],
            rgb[:, 1:],
            mask[:, :-1] & mask[:, 1:],
            labels[:, :-1] if labels is not None else None,
            labels[:, 1:] if labels is not None else None,
        ),
        (
            rgb[:-1, :],
            rgb[1:, :],
            mask[:-1, :] & mask[1:, :],
            labels[:-1, :] if labels is not None else None,
            labels[1:, :] if labels is not None else None,
        ),
    ):
        if valid.any():
            actual_a.append(left[valid])
            actual_b.append(right[valid])
            if labels is not None:
                changed = valid & (lab_left != lab_right)
                if changed.any():
                    label_a.append(left[changed])
                    label_b.append(right[changed])

    if actual_a:
        edge_error = _delta_e(np.concatenate(actual_a), np.concatenate(actual_b))
    else:
        edge_error = np.empty(0, dtype=np.float64)
    if label_a:
        label_error = _delta_e(np.concatenate(label_a), np.concatenate(label_b))
    else:
        label_error = np.empty(0, dtype=np.float64)

    hard_threshold = 8.0
    hard_count = int((edge_error >= hard_threshold).sum())
    material_line = max(6, int(round(0.20 * math.sqrt(max(1, int(mask.sum()))))))
    label_hard = bool(
        label_error.size >= material_line
        and float(np.median(label_error)) >= 10.0
    )
    return {
        "neighbour_pairs": int(edge_error.size),
        "hard_delta_e_threshold": hard_threshold,
        "hard_pair_count": hard_count,
        "hard_pair_share": _round(hard_count / max(1, edge_error.size), 6),
        "material_line_pair_count": material_line,
        "max_delta_e": _round(edge_error.max(), 4) if edge_error.size else 0.0,
        "p99_delta_e": _round(np.quantile(edge_error, 0.99), 4) if edge_error.size else 0.0,
        "label_boundary_pairs": int(label_error.size),
        "label_boundary_median_delta_e": (
            _round(np.median(label_error), 4) if label_error.size else None
        ),
        "material_internal_hard_edge": bool(hard_count >= material_line),
        "material_hard_label_boundary": label_hard,
    }


def _candidate_summary(
    kind: str,
    model: dict[str, Any],
    stops: list[dict[str, Any]],
    train_error: dict[str, Any],
    heldout_error: dict[str, Any],
    reasons: list[str],
    accepted: bool,
) -> dict[str, Any]:
    return {
        "type": kind,
        "status": "accepted" if accepted else "rejected",
        "model": model,
        "stops": stops,
        "train_error": train_error,
        "heldout_error": heldout_error,
        "reasons": list(reasons),
    }


def _fit_solid(
    colours: np.ndarray,
    training: np.ndarray,
    heldout: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    fitting, selection = _nested_training_split(training)
    selection_colour = np.median(colours[fitting], axis=0)
    selection_error = _error_stats(
        colours[selection],
        np.broadcast_to(selection_colour, (len(selection), 3)),
    )
    colour = np.median(colours[training], axis=0)
    train_render = np.broadcast_to(colour, (len(training), 3))
    heldout_render = np.broadcast_to(colour, (len(heldout), 3))
    model = {
        "type": "solid",
        "color": _hex(colour),
        "rgb": [int(value) for value in np.clip(np.rint(colour), 0, 255)],
    }
    summary = _candidate_summary(
            "solid",
            model,
            [],
            _error_stats(colours[training], train_render),
            _error_stats(colours[heldout], heldout_render),
            ["baseline_only"],
            True,
        )
    summary["inner_selection_error"] = selection_error
    summary["inner_selection_score"] = _round(
        float(selection_error["mean"])
        + 0.08 * float(selection_error["p90"])
        + 0.04 * float(selection_error["p99"]), 6)
    return summary, colour


def _nested_training_split(training: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    training = np.asarray(training, dtype=np.int64)
    if len(training) >= 64:
        selector = np.arange(len(training), dtype=np.int64) % 5 == 0
        return training[~selector], training[selector]
    return training, training


def _fit_colour_curve(
    t: np.ndarray,
    colours: np.ndarray,
    training: np.ndarray,
    heldout: np.ndarray,
    *,
    maximum_stops: int,
) -> dict[str, Any]:
    """Fit the smallest useful bounded native-SVG colour curve.

    Stop positions and stop count are selected by a deterministic inner split
    of the training pixels.  The outer heldout pixels are rendered exactly once
    after selection, so the reported validation evidence is not silently used
    for model tuning.  This lets a useful fourth/fifth stop repair a real colour
    bend without rewarding unnecessary handles or overfitting the final gate.
    """
    training = np.asarray(training, dtype=np.int64)
    # Candidate admission already guarantees at least 32 train samples. Tiny
    # fixtures keep the old fit-only behaviour rather than inventing an
    # under-powered model-selection split.
    fitting, selection = _nested_training_split(training)
    variants = []
    for offsets in _gradient_stop_offset_families(
        t, colours, fitting, maximum_stops=maximum_stops
    ):
        fitted_colours = _fit_stops(t[fitting], colours[fitting], offsets)
        selection_prediction = _render_stops(
            t[selection], offsets, fitted_colours)
        selection_error = _error_stats(
            colours[selection], selection_prediction)
        # Complexity is deliberately meaningful: additional colour stops are
        # accepted only when they buy visible inner-validation improvement.
        score = (
            float(selection_error["mean"])
            + 0.08 * float(selection_error["p90"])
            + 0.04 * float(selection_error["p99"])
            + 0.10 * max(0, len(offsets) - 2)
        )
        variants.append(
            (score, len(offsets), tuple(float(v) for v in offsets), offsets,
             selection_error)
        )
    variants.sort(key=lambda item: (item[0], item[1], item[2]))
    (_score, _stop_count, _offset_key, offsets,
     selection_error) = variants[0]
    # Refit the chosen bounded model on every training pixel only after the
    # hyper-parameters are fixed.  The outer holdout remains untouched.
    stop_colours = _fit_stops(t[training], colours[training], offsets)
    train_prediction = _render_stops(t[training], offsets, stop_colours)
    train_error = _error_stats(colours[training], train_prediction)
    heldout_prediction = _render_stops(t[heldout], offsets, stop_colours)
    return {
        "offsets": offsets,
        "colours": stop_colours,
        "train_prediction": train_prediction,
        "heldout_prediction": heldout_prediction,
        "train_error": train_error,
        "heldout_error": _error_stats(colours[heldout], heldout_prediction),
        "variant_count": int(len(variants)),
        "selection_error": selection_error,
        "selection_score": _round(float(_score), 6),
        "selection_samples": int(len(selection)),
        "fit_samples": int(len(fitting)),
        "selection_strategy": "nested_deterministic_training_split",
    }


def _error_comparison(
    actual: np.ndarray,
    candidate_prediction: np.ndarray,
    solid_prediction: np.ndarray,
) -> dict[str, Any]:
    candidate_error = _delta_e(actual, candidate_prediction)
    solid_error = _delta_e(actual, solid_prediction)
    degradation = candidate_error - solid_error
    improvement = solid_error - candidate_error
    return {
        "mean_improvement": _round(float(solid_error.mean() - candidate_error.mean()), 4),
        "p90_improvement": _round(
            float(np.quantile(solid_error, 0.90) - np.quantile(candidate_error, 0.90)), 4
        ),
        "p99_improvement": _round(
            float(np.quantile(solid_error, 0.99) - np.quantile(candidate_error, 0.99)), 4
        ),
        "degradation_margin": 0.75,
        "degraded_share": _round(float(np.mean(degradation > 0.75)), 6),
        "material_degradation_margin": 2.0,
        "materially_degraded_share": _round(float(np.mean(degradation > 2.0)), 6),
        "p99_sample_degradation": _round(float(np.quantile(degradation, 0.99)), 4),
        "improvement_margin": 0.75,
        "improved_share": _round(float(np.mean(improvement > 0.75)), 6),
    }


def _gradient_acceptance(
    candidate: dict[str, Any],
    solid: dict[str, Any],
    edge_stats: dict[str, Any],
    colour_span: float,
    *,
    maximum_mean_error: float,
    maximum_p90_error: float,
    maximum_p99_error: float,
    maximum_degraded_share: float,
    maximum_materially_degraded_share: float,
    maximum_p99_sample_degradation: float,
) -> tuple[bool, list[str]]:
    error = candidate["heldout_error"]
    baseline = solid["heldout_error"]
    reasons = []
    if colour_span < 4.0:
        reasons.append("insufficient_gradient_colour_span")
    if edge_stats["material_internal_hard_edge"]:
        reasons.append("material_internal_hard_edge")
    if edge_stats["material_hard_label_boundary"]:
        reasons.append("material_hard_label_boundary")
    if error["mean"] > maximum_mean_error:
        reasons.append("heldout_mean_error_too_high")
    if error["p90"] > maximum_p90_error:
        reasons.append("heldout_p90_error_too_high")
    if error["p99"] > maximum_p99_error:
        reasons.append("heldout_p99_error_too_high")
    comparison = candidate.get("comparison_to_solid") or {}
    mean_gain = float(comparison.get("mean_improvement", baseline["mean"] - error["mean"]))
    p90_gain = float(comparison.get("p90_improvement", baseline["p90"] - error["p90"]))
    if mean_gain < max(1.0, 0.18 * float(baseline["mean"])):
        reasons.append("mean_improvement_over_solid_not_material")
    if p90_gain < max(0.5, 0.08 * float(baseline["p90"])):
        reasons.append("tail_improvement_over_solid_not_material")
    if float(comparison.get("degraded_share", 1.0)) > maximum_degraded_share:
        reasons.append("degraded_share_exceeds_safety_budget")
    if (
        float(comparison.get("materially_degraded_share", 1.0))
        > maximum_materially_degraded_share
    ):
        reasons.append("materially_degraded_share_exceeds_safety_budget")
    if (
        float(comparison.get("p99_sample_degradation", float("inf")))
        > maximum_p99_sample_degradation
    ):
        reasons.append("heldout_sample_tail_degradation_too_high")
    # A strict per-sample tail comparison can reject a coherent gradient that
    # improves the great majority of an object while a small textured patch is
    # locally closer to the old flat band.  Permit only a narrowly bounded,
    # auditable exception: every absolute error gate must already pass, at
    # least 80% of pixels must improve, fewer than 7% may materially regress,
    # and this must be the sole remaining objection.  Caller-supplied stricter
    # tail limits remain absolute and never take this branch.
    tail_exception = bool(
        maximum_p99_sample_degradation >= 3.0
        and reasons == ["heldout_sample_tail_degradation_too_high"]
        and mean_gain >= 5.0
        and p90_gain >= 5.0
        and float(comparison.get("p99_improvement", 0.0)) >= 5.0
        and float(comparison.get("improved_share", 0.0)) >= 0.80
        and float(comparison.get("degraded_share", 1.0)) <= 0.10
        and float(comparison.get("materially_degraded_share", 1.0)) <= 0.07
        and float(comparison.get("p99_sample_degradation", float("inf"))) <= 8.0
    )
    candidate["dominant_improvement_tail_exception"] = {
        "used": tail_exception,
        "policy": "strict_absolute_errors_plus_dominant_heldout_improvement",
        "minimum_improved_share": 0.80,
        "maximum_degraded_share": 0.10,
        "maximum_materially_degraded_share": 0.07,
        "maximum_p99_sample_degradation": 8.0,
        "caller_stricter_limit_can_be_overridden": False,
    }
    if tail_exception:
        reasons.remove("heldout_sample_tail_degradation_too_high")
    return not reasons, reasons


def _fit_linear(
    xs: np.ndarray,
    ys: np.ndarray,
    colours: np.ndarray,
    training: np.ndarray,
    heldout: np.ndarray,
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    solid_colour: np.ndarray,
    maximum_stops: int,
    linear_direction_candidates: int,
) -> tuple[dict[str, Any], float]:
    train_x = xs[training].astype(np.float64)
    train_y = ys[training].astype(np.float64)
    x_mid = float(train_x.mean())
    y_mid = float(train_y.mean())
    x_scale = max(1.0, float(np.ptp(all_xs)))
    y_scale = max(1.0, float(np.ptp(all_ys)))
    design = np.column_stack(
        (
            np.ones(len(training)),
            (train_x - x_mid) / x_scale,
            (train_y - y_mid) / y_scale,
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, colours[training], rcond=None)
    gradient_matrix = np.vstack(
        (coefficients[1] / x_scale, coefficients[2] / y_scale)
    )
    left, singular, _right = np.linalg.svd(gradient_matrix, full_matrices=False)
    degenerate_regression = bool(
        not len(singular) or singular[0] <= 1e-9)
    if degenerate_regression:
        direction = np.asarray((1.0, 0.0))
        rank_ratio = 1.0
    else:
        direction = left[:, 0]
        rank_ratio = float(singular[1] / singular[0]) if len(singular) > 1 else 0.0
    colour_slope = direction @ gradient_matrix
    dominant_channel = int(np.abs(colour_slope).argmax())
    if colour_slope[dominant_channel] < 0.0:
        direction = -direction
        colour_slope = -colour_slope
    initial_direction = direction.astype(np.float64, copy=True)
    initial_projection_span = float(np.ptp(
        all_xs * initial_direction[0] + all_ys * initial_direction[1]))
    predicted_colour_span = float(np.linalg.norm(
        colour_slope * initial_projection_span))
    observed_channel_span = (
        np.quantile(colours[training], 0.95, axis=0)
        - np.quantile(colours[training], 0.05, axis=0))
    observed_colour_span = float(np.linalg.norm(observed_channel_span))
    first_order_span_ratio = (
        predicted_colour_span / observed_colour_span
        if observed_colour_span > 1e-9 else 1.0)

    def canonical_axis(axis):
        axis = np.asarray(axis, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            axis = np.asarray((1.0, 0.0))
        else:
            axis = axis / norm
        if axis[0] < -1e-12 or (abs(axis[0]) <= 1e-12 and axis[1] < 0.0):
            axis = -axis
        return axis

    def projected(axis):
        ux_value, uy_value = float(axis[0]), float(axis[1])
        all_projection = all_xs * ux_value + all_ys * uy_value
        minimum = float(all_projection.min())
        maximum = float(all_projection.max())
        axis_span = max(1e-9, maximum - minimum)
        values = (xs * ux_value + ys * uy_value - minimum) / axis_span
        return values, minimum, maximum, axis_span

    requested_search_count = max(1, int(linear_direction_candidates))
    # A first-order RGB plane cannot identify a symmetric dark-light-dark
    # ramp: the two slopes cancel even though a single spatial axis explains
    # the paint.  Escalate ambiguous/degenerate regressions to a bounded axis
    # search instead of incorrectly rejecting the native gradient outright.
    regression_fallback = bool(
        degenerate_regression or rank_ratio > 0.25
        or first_order_span_ratio < 0.20)
    search_count = min(8, max(
        requested_search_count, 5 if regression_fallback else 1))
    if search_count == 1:
        full_axes = [initial_direction]
        prescreen_count = 1
    else:
        axes = [canonical_axis(initial_direction)]
        for angle in np.linspace(0.0, math.pi, 24, endpoint=False):
            axes.append(canonical_axis((math.cos(angle), math.sin(angle))))
        base_angle = math.atan2(initial_direction[1], initial_direction[0])
        for delta in (-math.pi / 8, -math.pi / 24,
                      math.pi / 24, math.pi / 8):
            angle = base_angle + delta
            axes.append(canonical_axis((math.cos(angle), math.sin(angle))))
        unique_axes = {}
        for axis in axes:
            unique_axes.setdefault(
                (round(float(axis[0]), 8), round(float(axis[1]), 8)), axis)
        fitting, selection = _nested_training_split(training)
        prescreen = []
        for key, axis in sorted(unique_axes.items()):
            values, _minimum, _maximum, _span = projected(axis)
            fit_t = values[fitting]
            design = np.column_stack((
                np.ones(len(fitting)), fit_t, fit_t * fit_t,
                fit_t * fit_t * fit_t,
            ))
            coefficients, *_ = np.linalg.lstsq(
                design, colours[fitting], rcond=None)
            select_t = values[selection]
            select_design = np.column_stack((
                np.ones(len(selection)), select_t, select_t * select_t,
                select_t * select_t * select_t,
            ))
            selection_error = _error_stats(
                colours[selection], select_design @ coefficients)
            score = (float(selection_error["mean"])
                     + 0.08 * float(selection_error["p90"])
                     + 0.04 * float(selection_error["p99"]))
            prescreen.append((score, key, axis))
        prescreen.sort(key=lambda item: (item[0], item[1]))
        full_axes = [item[2] for item in prescreen[:search_count]]
        prescreen_count = len(prescreen)

    axis_fits = []
    for axis in full_axes:
        values, minimum, maximum, axis_span = projected(axis)
        curve = _fit_colour_curve(
            values, colours, training, heldout, maximum_stops=maximum_stops)
        selection_error = curve["selection_error"]
        score = (float(selection_error["mean"])
                 + 0.08 * float(selection_error["p90"])
                 + 0.04 * float(selection_error["p99"])
                 + 0.10 * max(0, len(curve["offsets"]) - 2))
        key = (round(float(axis[0]), 8), round(float(axis[1]), 8))
        axis_fits.append((score, key, axis, values, minimum, maximum,
                          axis_span, curve))
    axis_fits.sort(key=lambda item: (item[0], item[1]))
    (axis_score, _axis_key, direction, t, t_min, t_max, span,
     colour_curve) = axis_fits[0]
    ux, uy = float(direction[0]), float(direction[1])
    offsets = colour_curve["offsets"]
    stop_colours = colour_curve["colours"]

    centroid_x = float(all_xs.mean())
    centroid_y = float(all_ys.mean())
    perpendicular_x = centroid_x - ux * (centroid_x * ux + centroid_y * uy)
    perpendicular_y = centroid_y - uy * (centroid_x * ux + centroid_y * uy)
    model = {
        "type": "linear",
        "svg_type": "linearGradient",
        "gradient_units": "userSpaceOnUse",
        "x1": _round(perpendicular_x + ux * t_min, 5),
        "y1": _round(perpendicular_y + uy * t_min, 5),
        "x2": _round(perpendicular_x + ux * t_max, 5),
        "y2": _round(perpendicular_y + uy * t_max, 5),
        "direction": [_round(ux, 8), _round(uy, 8)],
        "axis_span_px": _round(span, 5),
        "colour_field_rank_ratio": _round(rank_ratio, 6),
        "first_order_predicted_colour_span": _round(
            predicted_colour_span, 6),
        "observed_robust_colour_span": _round(observed_colour_span, 6),
        "first_order_to_observed_span_ratio": _round(
            first_order_span_ratio, 6),
        "stop_count": int(len(offsets)),
        "bounded_maximum_stops": int(maximum_stops),
        "colour_curve_variant_count": int(colour_curve["variant_count"]),
        "colour_curve_selection_strategy": colour_curve["selection_strategy"],
        "colour_curve_fit_samples": int(colour_curve["fit_samples"]),
        "colour_curve_selection_samples": int(
            colour_curve["selection_samples"]),
        "linear_axis_prescreen_count": int(prescreen_count),
        "linear_axis_full_fit_count": int(len(axis_fits)),
        "linear_axis_requested_full_fit_count": int(requested_search_count),
        "linear_axis_regression_fallback_used": regression_fallback,
        "linear_axis_inner_selection_score": _round(axis_score, 6),
        "linear_axis_initial_abs_cosine": _round(abs(float(
            canonical_axis(initial_direction) @ canonical_axis(direction))), 6),
        "linear_axis_selection_uses_outer_heldout": False,
    }
    stop_list = _stops(offsets, stop_colours)
    train_error = colour_curve["train_error"]
    heldout_error = colour_curve["heldout_error"]
    # The first-order rank is diagnostic only.  Non-monotonic multi-stop
    # gradients can have a high rank or near-zero endpoint slope even when a
    # bounded one-axis curve fits extremely well.  Hard eligibility therefore
    # comes from inner selection, protected heldout errors and edge guards.
    reasons = []
    colour_span = max(
        float(_delta_e(stop_colours[i : i + 1], stop_colours[j : j + 1])[0])
        for i in range(len(stop_colours))
        for j in range(i + 1, len(stop_colours))
    )
    summary = _candidate_summary(
        "linear", model, stop_list, train_error, heldout_error, reasons, False
    )
    summary["inner_selection_error"] = colour_curve["selection_error"]
    summary["inner_selection_score"] = _round(axis_score, 6)
    summary["comparison_to_solid"] = _error_comparison(
        colours[heldout],
        colour_curve["heldout_prediction"],
        np.broadcast_to(solid_colour, (len(heldout), 3)),
    )
    return (
        summary,
        colour_span,
    )


def _radial_geometry_candidate(
    xs: np.ndarray,
    ys: np.ndarray,
    colours: np.ndarray,
    training: np.ndarray,
    heldout: np.ndarray,
    all_xs: np.ndarray,
    all_ys: np.ndarray,
    polarity: int,
    solid_colour: np.ndarray,
    maximum_stops: int,
) -> Optional[tuple[dict[str, Any], float]]:
    train_colours = colours[training]
    colour_center = train_colours.mean(axis=0)
    centered = train_colours - colour_center
    _u, singular, right = np.linalg.svd(centered, full_matrices=False)
    if not len(singular) or singular[0] <= 1e-6:
        return None
    colour_axis = right[0]
    dominant_channel = int(np.abs(colour_axis).argmax())
    if colour_axis[dominant_channel] < 0:
        colour_axis = -colour_axis
    progress = (colours - colour_center) @ colour_axis
    low = float(np.quantile(progress[training], 0.01))
    high = float(np.quantile(progress[training], 0.99))
    if high - low <= 1e-6:
        return None
    z = _clip01((progress - low) / (high - low))
    if polarity < 0:
        z = 1.0 - z

    x0, x1 = float(all_xs.min()), float(all_xs.max())
    y0, y1 = float(all_ys.min()), float(all_ys.max())
    x_mid, y_mid = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x_scale = max(1.0, (x1 - x0) / 2.0)
    y_scale = max(1.0, (y1 - y0) / 2.0)
    xn = (xs - x_mid) / x_scale
    yn = (ys - y_mid) / y_scale
    train_design = np.column_stack(
        (
            xn[training] ** 2,
            xn[training] * yn[training],
            yn[training] ** 2,
            xn[training],
            yn[training],
            np.ones(len(training)),
        )
    )
    coefficients, *_ = np.linalg.lstsq(
        train_design, z[training] ** 2, rcond=None
    )
    quad = np.asarray(
        (
            (coefficients[0], coefficients[1] / 2.0),
            (coefficients[1] / 2.0, coefficients[2]),
        ),
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(quad)
    if eigenvalues[0] <= 1e-5 or eigenvalues[-1] / eigenvalues[0] > 100.0:
        return None
    linear = np.asarray((coefficients[3], coefficients[4]))
    try:
        center_normalised = -0.5 * np.linalg.solve(quad, linear)
    except np.linalg.LinAlgError:
        return None
    if np.abs(center_normalised).max() > 1.30:
        return None

    def radial_coordinate(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        points = np.column_stack(
            (
                (x_values - x_mid) / x_scale - center_normalised[0],
                (y_values - y_mid) / y_scale - center_normalised[1],
            )
        )
        return np.sqrt(
            np.maximum(0.0, np.einsum("ni,ij,nj->n", points, quad, points))
        )

    all_radius = radial_coordinate(all_xs, all_ys)
    radius_scale = float(np.quantile(all_radius, 0.998))
    if radius_scale <= 1e-7:
        return None
    sample_radius = radial_coordinate(xs, ys)
    t = _clip01(sample_radius / radius_scale)
    if float(np.quantile(sample_radius, 0.01)) / radius_scale > 0.16:
        return None
    if int((t < 0.20).sum()) < max(8, int(0.002 * len(t))):
        return None

    colour_curve = _fit_colour_curve(
        t, colours, training, heldout, maximum_stops=maximum_stops
    )
    offsets = colour_curve["offsets"]
    stop_colours = colour_curve["colours"]
    train_error = colour_curve["train_error"]
    heldout_error = colour_curve["heldout_error"]

    predicted_quadratic = train_design @ coefficients
    geometry_rmse = float(
        np.sqrt(np.mean((predicted_quadratic - z[training] ** 2) ** 2))
    )

    inverse_scale = np.diag((1.0 / x_scale, 1.0 / y_scale))
    pixel_quad = inverse_scale @ quad @ inverse_scale
    pixel_eigenvalues, pixel_eigenvectors = np.linalg.eigh(pixel_quad)
    radii = radius_scale / np.sqrt(pixel_eigenvalues)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    major_vector = pixel_eigenvectors[:, order[0]]
    rotation = math.degrees(math.atan2(major_vector[1], major_vector[0]))
    while rotation >= 90.0:
        rotation -= 180.0
    while rotation < -90.0:
        rotation += 180.0
    center_x = x_mid + center_normalised[0] * x_scale
    center_y = y_mid + center_normalised[1] * y_scale

    model = {
        "type": "radial",
        "svg_type": "radialGradient",
        "gradient_units": "userSpaceOnUse",
        "center": [_round(center_x, 5), _round(center_y, 5)],
        "radius_x": _round(float(radii[0]), 5),
        "radius_y": _round(float(radii[1]), 5),
        "rotation_degrees": _round(rotation, 5),
        "geometry_fit_rmse": _round(geometry_rmse, 6),
        "polarity": "centre_to_edge" if polarity > 0 else "edge_to_centre",
        "stop_count": int(len(offsets)),
        "bounded_maximum_stops": int(maximum_stops),
        "colour_curve_variant_count": int(colour_curve["variant_count"]),
        "colour_curve_selection_strategy": colour_curve["selection_strategy"],
        "colour_curve_fit_samples": int(colour_curve["fit_samples"]),
        "colour_curve_selection_samples": int(
            colour_curve["selection_samples"]),
    }
    reasons = []
    if geometry_rmse > 0.10:
        reasons.append("elliptical_geometry_fit_uncertain")
    colour_span = max(
        float(_delta_e(stop_colours[i : i + 1], stop_colours[j : j + 1])[0])
        for i in range(len(stop_colours))
        for j in range(i + 1, len(stop_colours))
    )
    summary = _candidate_summary(
        "radial", model, _stops(offsets, stop_colours), train_error,
        heldout_error, reasons, False
    )
    summary["inner_selection_error"] = colour_curve["selection_error"]
    summary["inner_selection_score"] = _round(
        float(colour_curve["selection_score"]), 6)
    summary["comparison_to_solid"] = _error_comparison(
        colours[heldout],
        colour_curve["heldout_prediction"],
        np.broadcast_to(solid_colour, (len(heldout), 3)),
    )
    return (
        summary,
        colour_span,
    )


def _confidence(
    selected: dict[str, Any], solid: dict[str, Any], colour_span: float
) -> float:
    error = float(selected["heldout_error"]["mean"])
    tail = float(selected["heldout_error"]["p90"])
    baseline = float(solid["heldout_error"]["mean"])
    gain = max(0.0, baseline - error) / max(1.0, baseline)
    fit_score = max(0.0, 1.0 - error / 8.0)
    tail_score = max(0.0, 1.0 - tail / 12.0)
    span_score = min(1.0, max(0.0, colour_span / 20.0))
    return _round(
        min(1.0, 0.36 * fit_score + 0.30 * gain + 0.20 * tail_score
            + 0.14 * span_score),
        4,
    )


def _public_error(reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "error",
        "model": None,
        "stops": [],
        "confidence": 0.0,
        "mask": None,
        "error": None,
        "validation": None,
        "candidates": [],
        "reasons": [reason[:240]],
        "scope_note": (
            "No SVG mutation is authorised by an error result; callers must "
            "retain the existing vector candidate."
        ),
    }


def propose_gradient_object(
    rgb: Any,
    candidate_mask: Any,
    *,
    alpha: Any = None,
    label_map: Any = None,
    alpha_threshold: float = 12.0,
    min_pixels: int = 128,
    max_samples: int = 60000,
    heldout_fraction: float = 0.20,
    validation_seed: int = 0,
    allow_radial: bool = True,
    maximum_stops: int = 5,
    linear_direction_candidates: int = 1,
    maximum_mean_error: float = 5.5,
    maximum_p90_error: float = 8.5,
    maximum_p99_error: float = 13.0,
    maximum_degraded_share: float = 0.28,
    maximum_materially_degraded_share: float = 0.10,
    maximum_p99_sample_degradation: float = 3.0,
) -> dict[str, Any]:
    """Return a fail-closed, JSON-safe gradient object proposal.

    ``candidate_mask`` is authoritative ownership supplied by the caller; this
    function never expands it across object boundaries.  Pixels with alpha
    below ``alpha_threshold`` are excluded.  The proposal contains no raw mask
    array, only deterministic mask evidence and model parameters.

    A ``status`` of ``"proposed"`` means the selected native gradient beat the
    solid baseline on held-out pixels and passed absolute error, colour-span,
    one-dimensionality/ellipse, and hard-edge gates.  ``"skipped"`` and
    ``"error"`` explicitly authorise no replacement.
    """

    try:
        if isinstance(min_pixels, bool) or int(min_pixels) < 16:
            raise ValueError("min_pixels must be an integer of at least 16")
        if isinstance(max_samples, bool) or int(max_samples) < 64:
            raise ValueError("max_samples must be an integer of at least 64")
        if isinstance(validation_seed, bool) or not isinstance(
            validation_seed, (int, np.integer)
        ):
            raise ValueError("validation_seed must be an integer")
        if (
            isinstance(maximum_stops, bool)
            or not isinstance(maximum_stops, (int, np.integer))
            or not 2 <= int(maximum_stops) <= 5
        ):
            raise ValueError("maximum_stops must be an integer between 2 and 5")
        if (
            isinstance(linear_direction_candidates, bool)
            or not isinstance(linear_direction_candidates, (int, np.integer))
            or not 1 <= int(linear_direction_candidates) <= 8
        ):
            raise ValueError(
                "linear_direction_candidates must be an integer between 1 and 8"
            )
        heldout_fraction = float(heldout_fraction)
        if not 0.10 <= heldout_fraction <= 0.40:
            raise ValueError("heldout_fraction must be between 0.10 and 0.40")
        for name, value in (
            ("alpha_threshold", alpha_threshold),
            ("maximum_mean_error", maximum_mean_error),
            ("maximum_p90_error", maximum_p90_error),
            ("maximum_p99_error", maximum_p99_error),
            ("maximum_p99_sample_degradation", maximum_p99_sample_degradation),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        for name, value in (
            ("maximum_degraded_share", maximum_degraded_share),
            ("maximum_materially_degraded_share", maximum_materially_degraded_share),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        pixels, alpha_values, mask, labels = _normalise_inputs(
            rgb, candidate_mask, alpha, label_map, alpha_threshold
        )
        area = int(mask.sum())
        if area < int(min_pixels):
            fit_mask = mask.copy()
            return {
                **_public_error("candidate_mask_too_small"),
                "status": "skipped",
                "mask": _mask_stats(mask, fit_mask, alpha_values, labels),
                "reasons": ["candidate_mask_too_small"],
            }

        interior = _erode_once(mask)
        fit_mask = (
            interior
            if int(interior.sum()) >= max(int(min_pixels), int(round(0.55 * area)))
            else mask.copy()
        )
        all_ys, all_xs = np.nonzero(mask)
        ys, xs = np.nonzero(fit_mask)
        order, training, heldout = _sample_and_split(
            ys, xs, int(max_samples), heldout_fraction, int(validation_seed)
        )
        # Work only with the deterministic bounded sample from here onward.
        ys = ys[order]
        xs = xs[order]
        colours = pixels[ys, xs]
        if len(training) < 32 or len(heldout) < 16:
            raise ValueError("candidate did not provide enough train/holdout pixels")

        mask_info = _mask_stats(mask, fit_mask, alpha_values, labels)
        edge_stats = _internal_edge_stats(pixels, fit_mask, labels)
        solid, solid_colour = _fit_solid(colours, training, heldout)
        linear, linear_span = _fit_linear(
            xs, ys, colours, training, heldout,
            all_xs.astype(np.float64), all_ys.astype(np.float64),
            solid_colour, int(maximum_stops),
            int(linear_direction_candidates),
        )
        candidates = [solid, linear]
        selectable: list[tuple[dict[str, Any], float]] = [
            (solid, 0.0), (linear, linear_span)]

        if allow_radial:
            radial_options = []
            for polarity in (1, -1):
                fitted = _radial_geometry_candidate(
                    xs, ys, colours, training, heldout,
                    all_xs.astype(np.float64), all_ys.astype(np.float64),
                    polarity,
                    solid_colour, int(maximum_stops),
                )
                if fitted is None:
                    continue
                radial, radial_span = fitted
                radial_options.append((radial, radial_span))
            if radial_options:
                radial_options.sort(
                    key=lambda item: (
                        float(item[0]["inner_selection_score"]),
                        item[0]["model"]["polarity"],
                    )
                )
                best_radial, best_span = radial_options[0]
                candidates.append(best_radial)
                selectable.append((best_radial, best_span))

        # Choose exactly one paint class from the inner split.  The outer
        # coordinate holdout is then a single accept-or-skip gate: a failed
        # winner never falls back to whichever alternative happened to look
        # better on that protected evidence.
        intrinsically_eligible = [
            item for item in selectable
            if item[0]["type"] == "solid" or not item[0].get("reasons")
        ]
        selected, selected_span = min(
            intrinsically_eligible,
            key=lambda item: (
                float(item[0]["inner_selection_score"])
                + (0.35 if item[0]["type"] == "radial" else 0.0),
                {"solid": 0, "linear": 1, "radial": 2}[item[0]["type"]],
            ),
        )
        for candidate in candidates:
            if candidate is selected:
                candidate["status"] = (
                    "selected_inner_baseline" if candidate["type"] == "solid"
                    else "selected_for_outer_validation")
            elif candidate.get("reasons") and candidate["type"] != "solid":
                candidate["status"] = "ineligible_before_outer_validation"
            else:
                candidate["status"] = "not_selected_by_inner_validation"

        common_reasons = []
        if edge_stats["material_internal_hard_edge"]:
            common_reasons.append("material_internal_hard_edge")
        if edge_stats["material_hard_label_boundary"]:
            common_reasons.append("material_hard_label_boundary")
        if selected["type"] == "solid":
            reasons = []
            reasons.extend(common_reasons)
            reasons.append("solid_baseline_preferred")
            reasons.append("no_gradient_model_passed_validation")
            return {
                "schema": SCHEMA,
                "status": "skipped",
                "model": None,
                "stops": [],
                "confidence": 0.0,
                "mask": mask_info,
                "error": {
                    "metric": ERROR_METRIC,
                    "solid_baseline": solid["heldout_error"],
                },
                "validation": {
                    "strategy": "deterministic_coordinate_hash_holdout",
                    "model_selection_strategy": (
                        "inner_training_model_selection_then_single_outer_accept_or_skip"),
                    "model_selection_uses_outer_heldout": False,
                    "outer_holdout_role": (
                        "selected_model_accept_or_skip_no_fallback"),
                    "validation_seed": int(validation_seed),
                    "heldout_fraction": _round(heldout_fraction, 4),
                    "training_samples": int(len(training)),
                    "heldout_samples": int(len(heldout)),
                    "internal_edges": edge_stats,
                    "passed": False,
                },
                "candidates": candidates,
                "reasons": list(dict.fromkeys(reasons)),
                "scope_note": (
                    "No SVG mutation is authorised; retain the existing "
                    "candidate or request human review."
                ),
            }
        selected_ok, selected_reasons = _gradient_acceptance(
            selected, solid, edge_stats, selected_span,
            maximum_mean_error=float(maximum_mean_error),
            maximum_p90_error=float(maximum_p90_error),
            maximum_p99_error=float(maximum_p99_error),
            maximum_degraded_share=float(maximum_degraded_share),
            maximum_materially_degraded_share=float(
                maximum_materially_degraded_share),
            maximum_p99_sample_degradation=float(
                maximum_p99_sample_degradation),
        )
        selected["reasons"] = list(dict.fromkeys(
            selected.get("reasons", []) + selected_reasons))
        selected["status"] = "accepted" if selected_ok else "rejected"
        if not selected_ok:
            reasons = list(dict.fromkeys(
                common_reasons + selected["reasons"]
                + ["no_gradient_model_passed_validation"]))
            return {
                "schema": SCHEMA,
                "status": "skipped",
                "model": None,
                "stops": [],
                "confidence": 0.0,
                "mask": mask_info,
                "error": {
                    "metric": ERROR_METRIC,
                    "solid_baseline": solid["heldout_error"],
                },
                "validation": {
                    "strategy": "deterministic_coordinate_hash_holdout",
                    "model_selection_strategy": (
                        "inner_training_model_selection_then_single_outer_accept_or_skip"),
                    "model_selection_uses_outer_heldout": False,
                    "outer_holdout_role": (
                        "selected_model_accept_or_skip_no_fallback"),
                    "validation_seed": int(validation_seed),
                    "heldout_fraction": _round(heldout_fraction, 4),
                    "training_samples": int(len(training)),
                    "heldout_samples": int(len(heldout)),
                    "internal_edges": edge_stats,
                    "passed": False,
                },
                "candidates": candidates,
                "reasons": reasons,
                "scope_note": (
                    "The inner-selected paint model failed the protected outer "
                    "holdout. No fallback model or SVG mutation is authorised."
                ),
            }
        selected_error = selected["heldout_error"]
        solid_error = solid["heldout_error"]
        comparison = selected["comparison_to_solid"]
        confidence = _confidence(selected, solid, selected_span)
        reasons = [
            "gradient_materially_better_than_solid",
            "heldout_validation_passed",
            (
                "elliptical_radial_colour_field_fit"
                if selected["type"] == "radial"
                else "arbitrary_angle_linear_colour_field_fit"
            ),
        ]
        return {
            "schema": SCHEMA,
            "status": "proposed",
            "model": selected["model"],
            "stops": selected["stops"],
            "confidence": confidence,
            "mask": mask_info,
            "error": {
                "metric": ERROR_METRIC,
                "train": selected["train_error"],
                "heldout": selected_error,
                "solid_baseline": solid_error,
                "heldout_mean_improvement": _round(
                    float(comparison["mean_improvement"]), 4
                ),
                "heldout_p90_improvement": _round(
                    float(comparison["p90_improvement"]), 4
                ),
                "heldout_p99_improvement": _round(
                    float(comparison["p99_improvement"]), 4
                ),
                "stop_colour_span": _round(selected_span, 4),
                "endpoint_colour_span": _round(selected_span, 4),
                "comparison_to_solid": comparison,
            },
            "validation": {
                "strategy": "deterministic_coordinate_hash_holdout",
                "model_selection_strategy": (
                    "inner_training_model_selection_then_single_outer_accept_or_skip"),
                "axis_and_stop_selection_uses_outer_heldout": False,
                "model_class_selection_uses_outer_heldout": False,
                "model_selection_uses_outer_heldout": False,
                "outer_holdout_role": (
                    "selected_model_accept_or_skip_no_fallback"),
                "validation_seed": int(validation_seed),
                "heldout_fraction": _round(heldout_fraction, 4),
                "training_samples": int(len(training)),
                "heldout_samples": int(len(heldout)),
                "maximum_mean_error": _round(maximum_mean_error, 4),
                "maximum_p90_error": _round(maximum_p90_error, 4),
                "maximum_p99_error": _round(maximum_p99_error, 4),
                "maximum_degraded_share": _round(maximum_degraded_share, 6),
                "maximum_materially_degraded_share": _round(
                    maximum_materially_degraded_share, 6
                ),
                "maximum_p99_sample_degradation": _round(
                    maximum_p99_sample_degradation, 4
                ),
                "internal_edges": edge_stats,
                "dominant_improvement_tail_exception": selected.get(
                    "dominant_improvement_tail_exception", {"used": False}),
                "passed": True,
            },
            "selection_evidence": {
                "objective_order": [
                    "candidate_ownership_mask_is_authoritative",
                    "heldout_paint_error_within_budget",
                    "bounded_native_paint_model_complexity",
                ],
                "paint_model_complexity": {
                    "native_gradient_count": 1,
                    "stop_count": int(selected["model"]["stop_count"]),
                    "maximum_stop_count": int(maximum_stops),
                },
                "linear_direction_search": {
                    "requested_full_fit_candidates": int(
                        linear_direction_candidates),
                    "actual_full_fit_candidates": int(
                        selected["model"].get(
                            "linear_axis_full_fit_count", 0)
                        if selected["type"] == "linear" else 0),
                    "outer_heldout_used_for_axis_selection": False,
                },
                "axis_and_stop_selection_uses_outer_heldout": False,
                "model_class_selection_uses_outer_heldout": False,
                "model_selection_uses_outer_heldout": False,
                "outer_holdout_role": (
                    "selected_model_accept_or_skip_no_fallback"),
                "heldout_error_budget_passed": True,
                "tail_and_degraded_share_budget_passed": True,
                "geometry_selection_authorised": False,
                "geometry_note": (
                    "A later stage must minimise anchors subject to its "
                    "explicit geometry error budget; this paint fitter does "
                    "not trace pixel edges."
                ),
            },
            "candidates": candidates,
            "reasons": reasons,
            "scope_note": (
                "This is a source-space paint proposal only.  Geometry, "
                "stack order, holes and final native-SVG rendering still "
                "require a separate transactional validator."
            ),
        }
    except Exception as exc:  # Fail closed at the public integration boundary.
        return _public_error(f"gradient_object_engine_error:{type(exc).__name__}:{exc}")


# A descriptive alias for integration code and tests that prefer an explicit
# verb-object name.  Both names intentionally share the same implementation.
fit_gradient_object_proposal = propose_gradient_object


__all__ = [
    "ERROR_METRIC",
    "SCHEMA",
    "fit_gradient_object_proposal",
    "propose_gradient_object",
]
