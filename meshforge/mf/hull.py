"""Orthographic, 0-elevation visual hull with per-view offset calibration and voting.

Conventions (pixel units of a canonical 512 cell, grid step `step`):
  view azimuth t:  u = X cos t - Z sin t + C + off[t]      (image column)
                   v = Y                                    (image row)
                   w = X sin t + Z cos t                    (toward camera, bigger = closer)
  t = 0 looks at the front (+Z); t = 90 sees the front pointing to image-left.
"""
import numpy as np

SIZE = 512
C = SIZE / 2


class Grid:
    def __init__(self, step=2, half=256):
        self.step = step
        xs = np.arange(-half, half, step, dtype=np.float32)
        ys = np.arange(0, SIZE, step, dtype=np.float32)
        self.xs, self.ys = xs, ys
        X, Y, Z = np.meshgrid(xs, ys, xs, indexing="ij")  # (x, y, z)
        self.X, self.Y, self.Z = X, Y, Z
        self.shape = X.shape


def project(grid_or_pts, t, off=0.0):
    X, Y, Z = grid_or_pts
    r = np.deg2rad(t)
    u = X * np.cos(r) - Z * np.sin(r) + C + off
    w = X * np.sin(r) + Z * np.cos(r)
    return u, Y, w


def inside(mask, u, v):
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    ok = (ui >= 0) & (ui < mask.shape[1]) & (vi >= 0) & (vi < mask.shape[0])
    res = np.zeros(u.shape, bool)
    res[ok] = mask[vi[ok], ui[ok]]
    return res


def carve(grid, masks, offsets, min_votes=None):
    votes = np.zeros(grid.shape, np.int16)
    for t, m in masks.items():
        u, v, _ = project((grid.X, grid.Y, grid.Z), t, offsets.get(t, 0.0))
        votes += inside(m, u, v)
    n = len(masks)
    return votes >= (n if min_votes is None else min_votes)


def render(grid, occ, t, off=0.0):
    """Orthographic render of occupied voxels: returns (silhouette, depth w, xyz map)."""
    X, Y, Z = grid.X[occ], grid.Y[occ], grid.Z[occ]
    u, v, w = project((X, Y, Z), t, off)
    ui = np.clip(np.round(u).astype(int), 0, SIZE - 1)
    vi = np.clip(np.round(v).astype(int), 0, SIZE - 1)
    depth = np.full((SIZE, SIZE), -np.inf, np.float32)
    np.maximum.at(depth, (vi, ui), w)
    # splat to fill grid-step gaps
    s = grid.step
    from scipy import ndimage
    depth = ndimage.grey_dilation(depth, size=(s + 1, s + 1))
    sil = np.isfinite(depth)
    # recover surface xyz from (u, v, w)
    r = np.deg2rad(t)
    uu, vv = np.meshgrid(np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32))
    a = uu - C - off
    xyz = np.stack([a * np.cos(r) + depth * np.sin(r), vv,
                    -a * np.sin(r) + depth * np.cos(r)], -1)
    return sil, depth, xyz


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def reprojection_scores(grid, occ, masks, offsets):
    return {t: iou(render(grid, occ, t, offsets.get(t, 0.0))[0], m) for t, m in masks.items()}


def calibrate_offsets(grid, masks, fixed=0, search=24, step=2, rounds=2):
    """Coordinate descent on per-view horizontal offsets to maximize min reprojection IoU."""
    offsets = {t: 0.0 for t in masks}
    for _ in range(rounds):
        for t in masks:
            if t == fixed:
                continue
            best = (-1, 0.0)
            for d in np.arange(-search, search + 1, step):
                offsets[t] = float(d)
                occ = carve(grid, masks, offsets)
                sc = reprojection_scores(grid, occ, masks, offsets)
                val = min(sc.values())
                if val > best[0]:
                    best = (val, float(d))
            offsets[t] = best[1]
    return offsets


def calibrate_angle(grid, base_masks, base_offsets, mask, candidates, search=12, step=3):
    """Find the true azimuth (and offset) of a view whose nominal angle the generator ignored."""
    best = (-1.0, None, 0.0)
    for a in candidates:
        for d in range(-search, search + 1, step):
            masks = dict(base_masks); masks[a] = mask
            off = dict(base_offsets); off[a] = float(d)
            occ = carve(grid, masks, off)
            v = reprojection_scores(grid, occ, {a: mask}, off)[a]
            if v > best[0]:
                best = (v, a, float(d))
    return best
