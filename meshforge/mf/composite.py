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
