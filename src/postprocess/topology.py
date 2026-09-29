import numpy as np
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter1d, label


def calculate_iou_2d(mask_a, mask_b):
    intersect = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    if union == 0:
        return 0.0
    return float(intersect / (union + 1e-5))


def get_centroid_2d(mask_2d):
    coords = np.argwhere(mask_2d > 0)
    if len(coords) == 0:
        return None
    y_c, x_c = coords.mean(axis=0)
    return np.array([x_c, y_c], dtype=np.float32)


def fill_small_holes(mask_2d, max_hole_area=200):
    if np.sum(mask_2d) == 0:
        return mask_2d

    holes = binary_fill_holes(mask_2d) & ~mask_2d
    labeled_holes, num_features = label(holes)
    if num_features == 0:
        return mask_2d

    areas = np.bincount(labeled_holes.ravel())
    areas[0] = 0

    filled_mask = mask_2d.copy()
    for label_idx in range(1, len(areas)):
        if 0 < areas[label_idx] < max_hole_area:
            filled_mask[labeled_holes == label_idx] = True
    return filled_mask


def keep_largest_3d_component(mask_3d):
    labeled_3d, num_features = label(mask_3d)
    if num_features == 0:
        return mask_3d.astype(np.uint8)

    volumes = np.bincount(labeled_3d.ravel())
    volumes[0] = 0
    return (labeled_3d == volumes.argmax()).astype(np.uint8)


def _slice_connected(mask_a, mask_b, min_iou, centroid_jump_limit):
    if np.sum(mask_a) == 0 or np.sum(mask_b) == 0:
        return False

    if calculate_iou_2d(mask_a, mask_b) >= min_iou:
        return True

    c_a = get_centroid_2d(mask_a)
    c_b = get_centroid_2d(mask_b)
    if c_a is None or c_b is None:
        return False
    return float(np.linalg.norm(c_a - c_b)) <= centroid_jump_limit


def keep_dominant_temporal_chain(
    mask_3d,
    min_slice_area=50,
    min_overlap_iou=0.03,
    centroid_jump_limit=40.0,
    rescue_slices=2,
):
    areas = np.asarray([np.sum(mask_3d[z]) for z in range(mask_3d.shape[0])])
    candidate = areas >= min_slice_area

    chains = []
    current = []
    for z in range(mask_3d.shape[0]):
        if not candidate[z]:
            if current:
                chains.append(current)
                current = []
            continue

        if not current:
            current = [z]
            continue

        prev_z = current[-1]
        if _slice_connected(
            mask_3d[prev_z], mask_3d[z], min_overlap_iou, centroid_jump_limit
        ):
            current.append(z)
        else:
            chains.append(current)
            current = [z]

    if current:
        chains.append(current)

    if not chains:
        return np.zeros_like(mask_3d, dtype=np.uint8)

    best_chain = max(chains, key=lambda chain: float(np.sum(areas[chain])) + 1200.0 * len(chain))
    keep = np.zeros(mask_3d.shape[0], dtype=bool)
    keep[best_chain] = True

    # Recover very thin anatomical tips next to the dominant chain only when
    # they are spatially consistent with the current boundary slice.
    start, end = min(best_chain), max(best_chain)
    for z in range(start - 1, max(-1, start - rescue_slices - 1), -1):
        if z < 0 or areas[z] < min_slice_area * 0.6:
            break
        if _slice_connected(mask_3d[z], mask_3d[z + 1], min_overlap_iou * 0.5, centroid_jump_limit * 1.2):
            keep[z] = True
        else:
            break

    for z in range(end + 1, min(mask_3d.shape[0], end + rescue_slices + 1)):
        if areas[z] < min_slice_area * 0.6:
            break
        if _slice_connected(mask_3d[z - 1], mask_3d[z], min_overlap_iou * 0.5, centroid_jump_limit * 1.2):
            keep[z] = True
        else:
            break

    pruned = mask_3d.copy()
    pruned[~keep] = 0
    return pruned.astype(np.uint8)


def postprocess_prob_volume(
    prob_3d,
    threshold=0.35,
    opening=True,
    fill_holes=True,
    fill_small_holes_only=False,
    max_hole_area=200,
    min_size_filter=50,
    min_overlap_iou=0.03,
    centroid_jump_limit=40.0,
    sigma_area=2.0,
):
    pred = np.zeros_like(prob_3d, dtype=np.uint8)
    for z in range(prob_3d.shape[0]):
        binary_slice = (prob_3d[z] > threshold).astype(np.uint8)
        if np.sum(binary_slice) == 0:
            continue
        if opening:
            binary_slice = binary_opening(binary_slice, structure=np.ones((3, 3))).astype(
                np.uint8
            )
        if fill_small_holes_only:
            binary_slice = fill_small_holes(binary_slice, max_hole_area=max_hole_area)
        pred[z] = binary_slice.astype(np.uint8)

    pred = keep_largest_3d_component(pred)

    if fill_holes:
        pred = binary_fill_holes(pred).astype(np.uint8)

    pred = keep_dominant_temporal_chain(
        pred,
        min_slice_area=min_size_filter,
        min_overlap_iou=min_overlap_iou,
        centroid_jump_limit=centroid_jump_limit,
    )

    areas = np.asarray([np.sum(pred[z]) for z in range(pred.shape[0])], dtype=np.float32)
    smooth_areas = gaussian_filter1d(areas, sigma=sigma_area, mode="nearest")
    for z in range(pred.shape[0]):
        if smooth_areas[z] < min_size_filter:
            pred[z] = 0

    return pred.astype(np.uint8)
