"""Robust multi-view reconstruction from silhouettes + noisy normal maps (CPU only).
1. visual hull from (ragged) silhouettes with camera calibration
2. per-view depth by integrating the noisy normals, pinned to the hull only at silhouette rims
   (the one place a hull is exact), gradient terms switched off across occlusion edges
3. TSDF fusion of all views (independent noise averages out), carving by silhouettes
4. marching cubes"""
import numpy as np
from scipy import ndimage
from skimage import measure

from .hull import Grid, carve, render, project, calibrate_offsets, SIZE
from .photometric import integrate, normals_from_depth


def view_depth(N, mask, hull_depth, rim_px=3, lam_rim=1.0, lam_in=0.01):
    rim = mask & ~ndimage.binary_erosion(mask, iterations=rim_px)
    lam = np.where(rim, lam_rim, lam_in)
    hd = np.where(mask, hull_depth, np.nan)
    gy, gx = np.gradient(np.nan_to_num(hd, nan=0.0))
    jump = np.hypot(gx, gy)
    w = np.exp(-np.maximum(jump - 1.5, 0) / 2.0) * np.clip(N[..., 2] / 0.3, 0.05, 1.0)
    return integrate_lam(N, mask, hd, lam, w)


def integrate_lam(N, mask, anchor, lam, weights):
    # integrate() takes a scalar lambda; emulate a lambda map by scaling the anchor rows
    from scipy import sparse
    from scipy.sparse.linalg import lsqr
    H, W = mask.shape
    idx = -np.ones(mask.shape, int); idx[mask] = np.arange(mask.sum())
    nz = np.maximum(N[..., 2], 0.15)
    p, q = -N[..., 0] / nz, -N[..., 1] / nz
    rows, cols, vals, b = [], [], [], []
    r = 0
    for dy, dx, gr in ((0, 1, p), (1, 0, q)):
        a = mask[:H - dy, :W - dx] & mask[dy:, dx:]
        i0 = idx[:H - dy, :W - dx][a]; i1 = idx[dy:, dx:][a]
        gv = 0.5 * (gr[:H - dy, :W - dx][a] + gr[dy:, dx:][a])
        ww = np.sqrt(np.minimum(weights[:H - dy, :W - dx][a], weights[dy:, dx:][a]))
        rr = np.arange(r, r + len(i0))
        rows += [rr, rr]; cols += [i1, i0]; vals += [ww, -ww]; b.append(ww * gv); r += len(i0)
    n = int(mask.sum())
    ok = np.isfinite(anchor[mask])
    rr = np.arange(r, r + ok.sum())
    rows.append(rr); cols.append(np.arange(n)[ok]); vals.append(lam[mask][ok]); b.append((lam * np.nan_to_num(anchor))[mask][ok])
    r += ok.sum()
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, n))
    z = lsqr(A, np.concatenate(b), atol=1e-7, btol=1e-7, iter_lim=3000)[0]
    D = np.full(mask.shape, np.nan, np.float32); D[mask] = z
    return D


def tsdf_fuse(grid, views, offsets, trunc=6.0):
    """views: {t: (mask, depth, normals)}. Returns fused field (negative inside) and weights."""
    F = np.zeros(grid.shape, np.float32)
    Wt = np.zeros(grid.shape, np.float32)
    carved = np.zeros(grid.shape, bool)
    for t, (m, D, N) in views.items():
        u, v, w = project((grid.X, grid.Y, grid.Z), t, offsets.get(t, 0.0))
        ui = np.clip(np.round(u).astype(int), 0, SIZE - 1)
        vi = np.clip(np.round(v).astype(int), 0, SIZE - 1)
        inside = m[vi, ui]
        carved |= ~inside
        d = D[vi, ui]
        sdf = d - w                                   # >0: voxel in front of the surface (empty)
        valid = inside & np.isfinite(sdf) & (sdf > -trunc)
        conf = np.clip(N[..., 2], 0.1, 1.0)[vi, ui]
        val = np.clip(sdf / trunc, -1, 1)
        F[valid] += conf[valid] * val[valid]
        Wt[valid] += conf[valid]
    out = np.where(Wt > 0, F / np.maximum(Wt, 1e-6), -1.0)    # unobserved interior -> inside
    out[carved] = 1.0
    return out, Wt


def mesh_from_field(grid, F):
    verts, faces, _, _ = measure.marching_cubes(np.pad(F, 1, constant_values=1.0), 0.0)
    verts = (verts - 1) * grid.step
    verts[:, 0] += grid.xs[0]; verts[:, 2] += grid.xs[0]; verts[:, 1] += grid.ys[0]
    return verts, faces


def reconstruct(masks, normals, calibrate=True, step=2, log=print):
    g4, g = Grid(step=4), Grid(step=step)
    off = calibrate_offsets(g4, masks, search=6, step=1, rounds=1) if calibrate else {t: 0.0 for t in masks}
    occ = carve(g, masks, off)
    views = {}
    for t in masks:
        sil, hd, _ = render(g, occ, t, off.get(t, 0.0))
        m = masks[t] & sil
        D = view_depth(normals[t], m, hd)
        views[t] = (m, D, normals[t])
    F, W = tsdf_fuse(g, views, off)
    hull_mesh = mesh_from_field(g, np.where(occ, -1.0, 1.0).astype(np.float32))
    return mesh_from_field(g, F), hull_mesh, off, views
