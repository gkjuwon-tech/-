"""M4: did the generator keep the propagated (already-known) pixels?"""
import numpy as np
from skimage.metrics import structural_similarity
from skimage.feature import canny
from scipy import ndimage


def edge_fscore(a, b, mask, tol=2):
    ea = canny(a.mean(2) / 255.0, sigma=1.5) & mask
    eb = canny(b.mean(2) / 255.0, sigma=1.5) & mask
    da = ndimage.distance_transform_edt(~ea)
    db = ndimage.distance_transform_edt(~eb)
    p = (da[eb] <= tol).mean() if eb.any() else 1.0
    r = (db[ea] <= tol).mean() if ea.any() else 1.0
    return float(2 * p * r / (p + r + 1e-9))


def preservation(guide, out, known):
    """guide/out: HxWx3 uint8 at the same resolution; known: bool mask of propagated pixels."""
    k = ndimage.binary_erosion(known, iterations=3)  # ignore jagged borders
    diff = np.abs(guide.astype(int) - out.astype(int)).mean(2)
    _, ssim_map = structural_similarity(guide, out, channel_axis=2, full=True)
    return {
        "mean_abs_err": float(diff[k].mean()),
        "ssim": float(ssim_map.mean(2)[k].mean()),
        "edge_f": edge_fscore(guide, out, k),
    }
