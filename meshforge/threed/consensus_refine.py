"""Corrector, second stage: cross-view consensus.
Every view's estimator error is independent, so a surface fused from all views is a better
low-frequency reference than any single view. Render the fused mesh's normals back into each
view; keep the view's own high-frequency detail, replace its low frequencies by the consensus.
usage: python consensus_refine.py DATA/NAME MESH.ply --out DATA/NAME_cons [--sigma 0.03]"""
import argparse, json, os, shutil
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage


def render_normals(mesh, M, ortho, R, samples=3_000_000):
    right, up, back, loc = M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3]
    pts, fi = trimesh.sample.sample_surface(mesh, samples, seed=0)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fi], pts)
    nrm = (mesh.vertex_normals[mesh.faces[fi]] * bary[:, :, None]).sum(1)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    rel = pts - loc
    c = np.round((rel @ right / ortho + 0.5) * R - 0.5).astype(int)
    r = np.round((0.5 - rel @ up / ortho) * R - 0.5).astype(int)
    w = -(rel @ back)
    ok = (c >= 0) & (c < R) & (r >= 0) & (r < R)
    c, r, w, nrm = c[ok], r[ok], w[ok], nrm[ok]
    zb = np.full((R, R), np.inf); np.minimum.at(zb, (r, c), w)
    win = w <= zb[r, c] + 1e-9
    N = np.zeros((R, R, 3)); N[r[win], c[win]] = np.stack([nrm[win] @ right, nrm[win] @ up, nrm[win] @ back], -1)
    cov = np.isfinite(zb)
    idx = ndimage.distance_transform_edt(~cov, return_distances=False, return_indices=True)
    N = N[idx[0], idx[1]]
    N[N[..., 2] < 0, 2] = 0
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
    return N.astype(np.float32), ndimage.binary_closing(cov, iterations=2)


def masked_blur(F, m, s):
    w = ndimage.gaussian_filter(m.astype(np.float32), s)
    out = np.stack([ndimage.gaussian_filter(F[..., k] * m, s) for k in range(3)], -1)
    return out / np.maximum(w[..., None], 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("mesh"); ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=0.03, help="low/high split, fraction of image size")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "views", "cameras.json")))
    R, ortho = meta["resolution"][0], meta["ortho_scale"]
    mesh = trimesh.load(a.mesh, process=False)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(a.src, a.out, ignore=shutil.ignore_patterns("recon"))
    s = a.sigma * R
    for v, info in meta["views"].items():
        n = np.load(os.path.join(a.src, "normals", f"{v}.npy")).astype(np.float32)
        m = np.linalg.norm(n, axis=-1) > 0.5
        Nc, cov = render_normals(mesh, np.array(info["matrix_world"]), ortho, R)
        both = m & cov
        detail = n - masked_blur(n, m, s)
        low = masked_blur(Nc, both, s)
        out = np.where(both[..., None], low + detail, n)
        out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9)
        out[~m] = 0
        out[m & (out[..., 2] < 0), 2] = 0
        np.save(os.path.join(a.out, "normals", f"{v}.npy"), out.astype(np.float32))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
