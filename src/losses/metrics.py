import numpy as np
import SimpleITK as sitk
import logging

try:
    from medpy.metric.binary import dc, hd95
    HAS_MEDPY = True
except ImportError:
    HAS_MEDPY = False
    logging.warning(
        "medpy is unavailable; HD95 will use pixel-space fallback. "
        "Install medpy for physical-distance metrics in millimeters."
    )

def calculate_3d_dice(pred_arr, gt_arr):
    """Compute the volumetric Dice coefficient for binary 3D arrays."""
    pred_arr = (pred_arr > 0).astype(np.uint8)
    gt_arr = (gt_arr > 0).astype(np.uint8)

    intersection = np.sum(pred_arr * gt_arr)
    total = np.sum(pred_arr) + np.sum(gt_arr)

    if total == 0:
        return 1.0
    return (2.0 * intersection) / total

def calculate_3d_hd95(pred_arr, gt_arr, spacing=None):
    """Compute the 95th-percentile Hausdorff distance in physical units when available."""
    pred_arr = (pred_arr > 0).astype(np.uint8)
    gt_arr = (gt_arr > 0).astype(np.uint8)

    if np.sum(pred_arr) == 0 and np.sum(gt_arr) == 0:
        return 0.0
    if np.sum(pred_arr) == 0 or np.sum(gt_arr) == 0:
        return 100.0

    if HAS_MEDPY:
        try:
            # Reverse SimpleITK's (x, y, z) spacing for NumPy's (z, y, x) order.
            numpy_spacing = spacing[::-1] if spacing is not None else None
            return hd95(pred_arr, gt_arr, voxelspacing=numpy_spacing)
        except Exception as e:
            logging.error(f"Medpy HD95 calculation failed: {e}; using pixel-space fallback.")

    from scipy.ndimage import distance_transform_edt
    def get_surface(mask):
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask) ^ mask

    surface_pred = get_surface(pred_arr)
    surface_gt = get_surface(gt_arr)

    if not surface_pred.any() or not surface_gt.any():
        return 100.0

    dist_pred = distance_transform_edt(~surface_pred)
    dist_gt = distance_transform_edt(~surface_gt)

    d_gt_to_pred = dist_pred[surface_gt]
    d_pred_to_gt = dist_gt[surface_pred]

    all_distances = np.concatenate([d_gt_to_pred, d_pred_to_gt])
    pixel_hd95 = np.percentile(all_distances, 95)
    if spacing is not None:
        return pixel_hd95 * np.mean(spacing)
    return pixel_hd95
