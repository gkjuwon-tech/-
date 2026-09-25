"""Restore propagated (known) pixels over a generator output and feather the seam."""
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation


def align_to_guide(gen, guide, known):
    """Undo small global shifts the generator introduced (estimated on known pixels)."""
    g = guide.mean(2) * known
    o = gen.mean(2) * known
    shift, _, _ = phase_cross_correlation(g, o, upsample_factor=4)
    if np.abs(shift).max() > 12:  # implausible, keep as is
        return gen, (0.0, 0.0)
    out = np.stack([ndimage.shift(gen[..., c].astype(np.float32), shift, order=1, mode="nearest")
                    for c in range(3)], -1)
    return np.clip(out, 0, 255).astype(np.uint8), tuple(float(s) for s in shift)


def restore_and_blend(gen, guide, known, erode=2, feather=2.5):
    """alpha=1 deep inside known region (exact propagated pixels), fades to the generator
    output across a narrow band at the seam, 0 in newly generated regions."""
    core = ndimage.binary_erosion(known, iterations=erode)
    alpha = ndimage.gaussian_filter(core.astype(np.float32), feather)
    alpha = np.where(core, np.maximum(alpha, 0.5), alpha)
    alpha = ndimage.gaussian_filter(alpha, 1.0)[..., None]
    out = alpha * guide.astype(np.float32) + (1 - alpha) * gen.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8), alpha[..., 0]


def trusted_known(gen, guide, known, gen_mask, sigma=3.0, max_diff=28.0):
    """Known pixels are only trusted where the proxy and the generator broadly agree.
    Agreement is judged on a blurred colour field, so scale-level detail differences are
    still overwritten by the propagated pixels, while places where the proxy geometry is
    wrong (e.g. a thin wing hull projected as a slab) are left to the generator."""
    gb = np.stack([ndimage.gaussian_filter(guide[..., c].astype(np.float32), sigma) for c in range(3)], -1)
    ob = np.stack([ndimage.gaussian_filter(gen[..., c].astype(np.float32), sigma) for c in range(3)], -1)
    agree = np.abs(gb - ob).mean(2) < max_diff
    t = known & gen_mask & agree
    return ndimage.binary_opening(t, iterations=1)
