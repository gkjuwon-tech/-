"""Ground-truth renders of a real mesh in the MeshForge turntable frame (orthographic, 0 elevation)."""
import numpy as np
import trimesh
from scipy import ndimage

from mf.hull import C, SIZE


def load_mesh(path, margin=26):
    m = trimesh.load(path)
    v = m.vertices.copy()
    v[:, 1] *= -1                                    # world Y up -> image rows down
    center = (v.max(0) + v.min(0)) / 2
    v -= center
    reach = np.sqrt((v[:, 0] ** 2 + v[:, 2] ** 2).max())
    s = (SIZE / 2 - margin) / reach
    v *= s
    v[:, 1] += C
    vn = m.vertex_normals.copy(); vn[:, 1] *= -1
    return v, vn


def render_gt(v, vn, t):
    """Returns mask, depth w, camera-space normals (x right, y down, z toward camera)."""
    r = np.deg2rad(t)
    u = v[:, 0] * np.cos(r) - v[:, 2] * np.sin(r) + C
    w = v[:, 0] * np.sin(r) + v[:, 2] * np.cos(r)
    nx = vn[:, 0] * np.cos(r) - vn[:, 2] * np.sin(r)
    nz = vn[:, 0] * np.sin(r) + vn[:, 2] * np.cos(r)
    ui = np.clip(np.round(u).astype(int), 0, SIZE - 1)
    vi = np.clip(np.round(v[:, 1]).astype(int), 0, SIZE - 1)
    depth = np.full((SIZE, SIZE), -np.inf)
    np.maximum.at(depth, (vi, ui), w)
    win = w >= depth[vi, ui] - 1e-6
    N = np.zeros((SIZE, SIZE, 3))
    N[vi[win], ui[win]] = np.stack([nx, vn[:, 1], nz], -1)[win]
    cov = np.isfinite(depth)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(cov, iterations=2))
    # fill splat holes from the nearest covered pixel
    idx = ndimage.distance_transform_edt(~cov, return_distances=False, return_indices=True)
    N = N[idx[0], idx[1]]; depth = depth[idx[0], idx[1]]
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    N[N[..., 2] < 0] *= [1, 1, -1]
    N[~mask] = 0
    depth[~mask] = np.nan
    return mask, depth, N


def base_image(N, mask, albedo=(0.46, 0.62, 0.50)):
    """Soft, mostly frontal studio lighting on a jade-coloured statue."""
    key = np.array([-0.3, -0.45, 0.84]); key /= np.linalg.norm(key)
    s = 0.25 + 0.75 * np.clip(N @ key, 0, None)
    img = np.full(mask.shape + (3,), 128.0)
    img[mask] = np.clip(s[mask, None] * np.array(albedo) * 255 * 1.25, 0, 255)
    return img.astype(np.uint8)


def angular_error(Na, Nb, mask):
    d = np.degrees(np.arccos(np.clip((Na * Nb).sum(-1), -1, 1)))[mask]
    return {"mean": float(d.mean()), "median": float(np.median(d)),
            "<11.25": float((d < 11.25).mean()), "<22.5": float((d < 22.5).mean()), "<30": float((d < 30).mean())}
