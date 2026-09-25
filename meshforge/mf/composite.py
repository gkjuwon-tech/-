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


def blur_rgb(img, sigma):
    return np.stack([ndimage.gaussian_filter(img[..., c].astype(np.float32), sigma) for c in range(3)], -1)


def agree(a, b, sigma=3.0, max_diff=28.0):
    """Coarse agreement of two images (geometry-level, not scale-level)."""
    return np.abs(blur_rgb(a, sigma) - blur_rgb(b, sigma)).mean(2) < max_diff


def certify(per_source, witness, witness_mask):
    """Two-witness rule. per_source: list of (guide, known) propagated from each finished view.
    A target pixel is certified when two independent images agree on it:
    two finished source views agree with each other, or a source view agrees with the
    independently drawn sheet view at this angle (the witness). Returns (guide, certified)."""
    guide = np.zeros_like(per_source[0][0])
    known_any = np.zeros(witness_mask.shape, bool)
    cert = np.zeros(witness_mask.shape, bool)
    for i, (gi, ki) in enumerate(per_source):
        ok_w = ki & witness_mask & agree(gi, witness)
        ok_s = np.zeros_like(ok_w)
        for j, (gj, kj) in enumerate(per_source):
            if j != i:
                ok_s |= ki & kj & agree(gi, gj)
        take = (ok_w | ok_s) & ~cert
        guide[take] = gi[take]
        cert |= take
        fill = ki & ~known_any
        known_any |= ki
    cert = ndimage.binary_opening(cert, iterations=1)
    return guide, cert, known_any


def fragments(mask, min_area=150):
    """Number of floating fragments (small disconnected pieces) in a silhouette."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return int((sizes < min_area).sum())
