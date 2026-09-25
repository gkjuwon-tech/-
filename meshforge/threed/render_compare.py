"""Shaded comparison renders (CPU point splatting): ground truth vs reconstructions.
usage: python render_compare.py OUT.png GT.ply(normalized by cameras.json) CAMS.json MESH1.ply [MESH2.ply ...]"""
import json, sys
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

out, gt, cams, *meshes = sys.argv[1:]
meta = json.load(open(cams))
norm = meta["normalization"]


def load(path, is_gt):
    m = trimesh.load(path, process=False)
    if is_gt:
        m.vertices = m.vertices * norm["applied_scale"] + np.array(norm["applied_offset"])
    return m


def render(m, az, el, res=640, samples=6_000_000):
    V = m.vertices
    center = (V.max(0) + V.min(0)) / 2
    ext = np.linalg.norm(V.max(0) - V.min(0)) * 0.62
    a, e = np.radians(az), np.radians(el)
    back = np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    right = np.cross([0, 0, 1.0], back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    pts, fi = trimesh.sample.sample_surface(m, samples, seed=0)
    n = m.face_normals[fi]
    S = 2 * res
    rel = pts - center
    c = np.round((rel @ right / (2 * ext) + 0.5) * S).astype(int)
    r = np.round((0.5 - rel @ up / (2 * ext)) * S).astype(int)
    w = rel @ back
    ok = (c >= 0) & (c < S) & (r >= 0) & (r < S)
    c, r, w, n = c[ok], r[ok], w[ok], n[ok]
    zb = np.full((S, S), -np.inf); np.maximum.at(zb, (r, c), w)
    win = w >= zb[r, c] - 1e-9
    N = np.zeros((S, S, 3)); N[r[win], c[win]] = n[win]
    cov = np.isfinite(zb)
    L1 = -0.35 * right + 0.55 * up + 0.76 * back
    L2 = 0.6 * right + 0.1 * up + 0.4 * back
    sh = 0.18 + 0.72 * np.clip(N @ (L1 / np.linalg.norm(L1)), 0, 1) + 0.2 * np.clip(N @ (L2 / np.linalg.norm(L2)), 0, 1)
    img = np.where(cov[..., None], np.clip(sh, 0, 1)[..., None] * np.array([232, 226, 214]), np.array([46, 48, 54]))
    img = img.reshape(res, 2, res, 2, 3).mean((1, 3))
    return img.astype(np.uint8)


rows = []
models = [load(gt, True)] + [load(p, False) for p in meshes]
for az, el in ((0, 10), (60, 20), (160, 25)):
    rows.append(np.concatenate([render(m, az, el) for m in models], 1))
Image.fromarray(np.concatenate(rows, 0)).save(out)
print("wrote", out)
