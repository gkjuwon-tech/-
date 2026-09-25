"""Make ground-truth inputs look like what real tools produce ('no greenhouse flowers'):
monocular-estimator normals (~20-30 deg error, smooth biases, lost detail, per-view lighting bias),
ragged silhouettes and wrong cameras."""
import numpy as np
from scipy import ndimage


def _rotate(N, axis_angle):
    th = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    k = axis_angle / np.maximum(th, 1e-9)
    c, s = np.cos(th), np.sin(th)
    return N * c + np.cross(k, N) * s + k * (k * N).sum(-1, keepdims=True) * (1 - c)


def degrade_normals(N, mask, rng, smooth_deg=18.0, smooth_sigma=25.0, detail_blur=2.0,
                    pixel_deg=8.0, global_deg=6.0):
    # detail loss (estimators over-smooth)
    Nb = np.stack([ndimage.gaussian_filter(N[..., c], detail_blur) for c in range(3)], -1)
    # smooth, spatially varying bias (the classic monocular 'bent surface')
    f = np.stack([ndimage.gaussian_filter(rng.standard_normal(mask.shape), smooth_sigma) for _ in range(3)], -1)
    f /= f[mask].std() + 1e-9
    aa = f * np.deg2rad(smooth_deg) / np.sqrt(3)
    # per-view global bias ("lighting misread")
    g = rng.standard_normal(3); g = g / np.linalg.norm(g) * np.deg2rad(global_deg)
    # per-pixel noise
    p = rng.standard_normal(mask.shape + (3,)) * np.deg2rad(pixel_deg) / np.sqrt(3)
    Nn = _rotate(Nb, aa + g + p)
    Nn /= np.maximum(np.linalg.norm(Nn, axis=-1, keepdims=True), 1e-9)
    Nn[Nn[..., 2] < 0.02, 2] = 0.02
    Nn /= np.linalg.norm(Nn, axis=-1, keepdims=True)
    Nn[~mask] = 0
    return Nn


def degrade_mask(mask, rng, ragged=2, shift=3):
    m = mask.copy()
    noise = ndimage.gaussian_filter(rng.standard_normal(mask.shape), 3)
    grow = ndimage.binary_dilation(m, iterations=ragged) & (noise > 0.15)
    shrink = m & ~ndimage.binary_erosion(m, iterations=ragged) & (noise < -0.15)
    m = (m | grow) & ~shrink
    dy, dx = rng.integers(-shift, shift + 1, 2)
    return np.roll(np.roll(m, dy, 0), dx, 1), (int(dy), int(dx))
