"""Projective propagation: carry pixels from source views to a target view via the proxy."""
import numpy as np

from .hull import project, render, SIZE


def visible_from(grid, occ, xyz, t, off, tol=4.0):
    """For surface points xyz (HxWx3), return (u, v, visible) in view t."""
    _, depth, _ = render(grid, occ, t, off)
    u, v, w = project((xyz[..., 0], xyz[..., 1], xyz[..., 2]), t, off)
    ui = np.clip(np.round(u).astype(int), 0, SIZE - 1)
    vi = np.clip(np.round(v).astype(int), 0, SIZE - 1)
    vis = np.isfinite(w) & (w >= depth[vi, ui] - tol)
    return ui, vi, vis


def propagate(grid, occ, offsets, sources, target):
    """sources: {t: rgb}. Returns (target_rgb_guide, filled_mask, target_silhouette).
    Each target pixel takes the colour of the source view that sees it most frontally."""
    sil, _, xyz = render(grid, occ, target, offsets.get(target, 0.0))
    out = np.zeros((SIZE, SIZE, 3), np.uint8)
    best = np.full((SIZE, SIZE), np.inf)
    filled = np.zeros((SIZE, SIZE), bool)
    for t, rgb in sources.items():
        ui, vi, vis = visible_from(grid, occ, xyz, t, offsets.get(t, 0.0))
        vis &= sil
        d = abs(((target - t + 180) % 360) - 180)  # prefer angularly close sources
        take = vis & (d < best)
        out[take] = rgb[vi[take], ui[take]]
        best[take] = d
        filled |= take
    return out, filled, sil
