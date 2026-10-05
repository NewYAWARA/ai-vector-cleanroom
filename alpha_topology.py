"""Detect new enclosed transparency gaps and erased holes in SVG proposals.

Checks coverage on a transparent canvas at three alpha thresholds. It is a
render-resolution guard, not a proof of continuous geometric equivalence.
One-pixel correspondence tolerates edge rasterisation, not new interior holes.
"""
import numpy as np
from PIL import Image


def _holes(alpha, threshold):
    from stroke_engine import connected_components
    labels, count = connected_components(alpha < threshold)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:,0], labels[:,-1])))
    sizes = np.bincount(labels.ravel(), minlength=count+1)
    keep = sizes >= 2
    keep[border] = False
    keep[0] = False
    return labels, keep


def _near(mask):
    h,w = mask.shape
    padded = np.pad(mask,1)
    return np.logical_or.reduce([padded[y:y+h,x:x+w] for y in range(3) for x in range(3)])


def compare_alpha_topology(before_png, after_png):
    with Image.open(before_png) as image:
        before = np.asarray(image.convert('RGBA'))[:,:,3]
    with Image.open(after_png) as image:
        after = np.asarray(image.convert('RGBA'))[:,:,3]
    record = {'method': 'enclosed_alpha_regions_at_32_128_224', 'minimum_region_pixels': 2,
              'correspondence_tolerance_pixels': 1, 'thresholds': [], 'accepted': True}
    if before.shape != after.shape:
        return {**record, 'accepted': False, 'reason': 'render_dimensions_differ'}
    if np.array_equal(before, after):
        return {**record, 'identical_alpha': True}
    for threshold in (32,128,224):
        a,ak = _holes(before,threshold)
        b,bk = _holes(after,threshold)
        near_a,near_b = _near(ak[a]),_near(bk[b])
        # An enclosed transparent region must correspond to an existing hole.
        # Exterior background is not a matching hole even if it is very close.
        unmatched_b = np.flatnonzero(bk).tolist()
        unmatched_a = np.flatnonzero(ak).tolist()
        touched_b = set(np.unique(b[near_a]).tolist())
        touched_a = set(np.unique(a[near_b]).tolist())
        created = [label for label in unmatched_b if label not in touched_b]
        erased = [label for label in unmatched_a if label not in touched_a]
        row = {'alpha_threshold': threshold, 'holes_before': int(ak.sum()), 'holes_after': int(bk.sum()),
               'created_regions': len(created), 'erased_regions': len(erased)}
        record['thresholds'].append(row)
        if created or erased or row['holes_before'] != row['holes_after']:
            record['accepted'] = False
    return record


def _unique_label_pairs(first, second):
    """Lexicographic unique label pairs without expensive 2-column sorting."""
    first = np.asarray(first, dtype=np.int64)
    second = np.asarray(second, dtype=np.int64)
    if first.shape != second.shape:
        raise ValueError('alpha_component_pair_shape_mismatch')
    if not first.size:
        return np.empty((0, 2), dtype=np.int64)
    if first.min() < 0 or second.min() < 0:
        raise ValueError('alpha_component_negative_label')
    radix = int(second.max()) + 1
    # Native references are capped at 32M pixels, so even pixel-count labels
    # produce <2**51 keys. Check explicitly rather than trusting the caller.
    if radix > np.iinfo(np.int64).max or int(first.max()) > (np.iinfo(np.int64).max - int(second.max())) // radix:
        raise ValueError('alpha_component_pair_overflow')
    keys = np.unique(first * radix + second)
    return np.column_stack(np.divmod(keys, radix))


def _component_correspondence(before, after):
    from stroke_engine import connected_components
    arrays = (before, after)
    rows = []
    identical = bool(np.array_equal(before, after))
    if not identical:
        for threshold in (32, 128, 224):
            maps, kept = [], []
            for alpha in arrays:
                labels, count = connected_components(alpha >= threshold)
                eligible = np.bincount(labels.ravel(), minlength=count + 1) >= 2
                eligible[0] = False
                maps.append(np.where(eligible[labels], labels, 0))
                kept.append(np.flatnonzero(eligible))
            a, b = maps
            common = (a != 0) & (b != 0)
            pairs = _unique_label_pairs(a[common], b[common])
            degree_a = np.bincount(pairs[:, 0], minlength=int(a.max()) + 1) if len(pairs) else np.zeros(int(a.max()) + 1, dtype=int)
            degree_b = np.bincount(pairs[:, 1], minlength=int(b.max()) + 1) if len(pairs) else np.zeros(int(b.max()) + 1, dtype=int)
            # Tolerate a one-pixel raster shift only for still-unmatched
            # regions. Expanding every component would mistake unchanged
            # close neighbours for merges. Direct split/merge evidence stays.
            if np.any(degree_a[kept[0]] == 0) and np.any(degree_b[kept[1]] == 0):
                padded = np.pad(b, 1)
                nearby = []
                for dy in range(3):
                    for dx in range(3):
                        shifted = padded[dy:dy + a.shape[0], dx:dx + a.shape[1]]
                        match = (a != 0) & (shifted != 0) & (degree_a[a] == 0) & (degree_b[shifted] == 0)
                        if match.any():
                            nearby.append(np.column_stack((a[match], shifted[match])))
                if nearby:
                    combined = np.concatenate(nearby)
                    near_pairs = _unique_label_pairs(combined[:, 0], combined[:, 1])
                    degree_a += np.bincount(near_pairs[:, 0], minlength=len(degree_a))
                    degree_b += np.bincount(near_pairs[:, 1], minlength=len(degree_b))
            row = {'alpha_threshold': threshold, 'components_before': len(kept[0]),
                   'components_after': len(kept[1]),
                   'unmatched_or_split_before': int(np.sum(degree_a[kept[0]] != 1)),
                   'unmatched_or_merged_after': int(np.sum(degree_b[kept[1]] != 1))}
            rows.append(row)
            if (row['components_before'] != row['components_after']
                    or row['unmatched_or_split_before'] or row['unmatched_or_merged_after']):
                raise ValueError("whole_alpha_components_changed")
    return rows


def compare_alpha_components(before_png, after_png):
    """Require the original one-to-one foreground component correspondence.

    This separately callable check never authorises a topology repair. A
    source-reconstruction transaction must independently prove any hole change.
    The ordinary composed-alpha guard continues to require unchanged holes.
    """
    arrays = []
    for path in (before_png, after_png):
        with Image.open(path) as image:
            if 'A' not in image.getbands():
                raise ValueError('whole_alpha_requires_rgba_renderer')
            arrays.append(np.asarray(image.convert('RGBA'), dtype=np.int16)[:, :, 3])
    before, after = arrays
    if before.shape != after.shape:
        raise ValueError('whole_alpha_render_dimensions_changed')
    return {'external_render_check': 'completed', 'accepted': True,
            'component_thresholds': _component_correspondence(before, after),
            'component_minimum_region_pixels': 2,
            'component_correspondence_tolerance_pixels': 1,
            'component_correspondence': 'one_to_one_overlap_then_1px_for_unmatched_regions',
            'scope': 'connected_components_only_not_hole_or_coverage_authorization'}


def compare_composed_alpha(before_png, after_png, *, check_coverage=True):
    """Opt-in composed coverage/component guard; raise ValueError on failure.

    The legacy compare_alpha_topology contract and its callers stay unchanged.
    A geometry-budgeted fitter may opt out of alpha-magnitude comparison;
    holes and one-to-one connected-component correspondence remain mandatory.
    """
    import numpy as np
    from PIL import Image

    arrays = []
    for path in (before_png, after_png):
        with Image.open(path) as image:
            if 'A' not in image.getbands():
                raise ValueError("whole_alpha_requires_rgba_renderer")
            arrays.append(np.asarray(image.convert('RGBA'), dtype=np.int16)[:, :, 3])
    before, after = arrays
    if before.shape != after.shape:
        raise ValueError("whole_alpha_render_dimensions_changed")
    holes = compare_alpha_topology(before_png, after_png)
    if holes.get('accepted') is not True:
        raise ValueError("whole_alpha_holes_changed")
    union = (before > 8) | (after > 8)
    error = np.abs(before - after)
    mean = float(error[union].mean()) if union.any() else 0.0
    changed_fraction = float((error[union] > 16).mean()) if union.any() else 0.0
    if check_coverage and (mean > 1.0 or changed_fraction > 0.02):
        raise ValueError("whole_alpha_coverage_guard_failed")
    identical = bool(np.array_equal(before, after))
    rows = _component_correspondence(before, after)
    return {'external_render_check': 'completed', 'accepted': True,
            'holes': holes, 'identical_alpha': identical, 'component_thresholds': rows,
            'component_minimum_region_pixels': 2, 'component_correspondence_tolerance_pixels': 1,
            'component_correspondence': 'one_to_one_overlap_then_1px_for_unmatched_regions',
            'coverage_mean_absolute_alpha_error': mean, 'coverage_changed_fraction_over_16': changed_fraction,
            'coverage_checked': bool(check_coverage),
            'coverage_policy': 'bounded_alpha_magnitude' if check_coverage else 'geometry_error_contract_separately_enforced',
            'scope': 'composed_render_resolution_guard_not_continuous_geometry_proof'}


