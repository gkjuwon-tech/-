"""Corrector, consensus stage v2: estimate each view's estimator bias as a SMOOTH ROTATION FIELD
against the normals of a fused surface, and undo it on the view's own normals.

Why a rotation field and not a low/high frequency swap: a monocular estimator's error is a slowly
varying tilt of otherwise well-shaped detail. Rotating the input back keeps every scale and
wrinkle exactly as the estimator drew it, and only removes the tilt. The fused surface is the
consensus of all views (their errors are independent), so it is the better reference for the
tilt, even when it is too blurry to be a reference for detail.

Robust: pixels where input and consensus disagree by more than --max-deg (occlusion edges,
places the fused surface got wrong) do not vote; the field is a normalised Gaussian average of
per-pixel rotations with a large sigma; the rim constraint (normal perpendicular to the outline)
votes too, with high weight.
usage: python consensus_rot.py DATA/NAME MESH.ply --out DATA/NAME_rot [--sigma 0.06]"""
import argparse, json, os, shutil, sys
import numpy as np
import trimesh
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from consensus_refine import render_normals          # noqa: E402
from corrector import min_rotation, apply_rotation, rim_targets  # noqa: E402


def rotation_field(n, ref, w, sigma):
    aa = min_rotation(n, ref)
    num = np.stack([ndimage.gaussian_filter(aa[..., k] * w, sigma) for k in range(3)], -1)
    den = ndimage.gaussian_filter(w, sigma)
    return num / np.maximum(den[..., None], 1e-6), den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("mesh"); ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--max-deg", type=float, default=50.0)
    ap.add_argument("--rim-weight", type=float, default=4.0)
    ap.add_argument("--iters", type=int, default=2, help="re-estimate on the corrected normals")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.src, "views", "cameras.json")))
    R, ortho = meta["resolution"][0], meta["ortho_scale"]
    mesh = trimesh.load(a.mesh, process=False)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    shutil.copytree(a.src, a.out, ignore=shutil.ignore_patterns("recon"))
    for v, info in meta["views"].items():
        n = np.load(os.path.join(a.src, "normals", f"{v}.npy")).astype(np.float64)
        m = np.linalg.norm(n, axis=-1) > 0.5
        n[m] /= np.linalg.norm(n[m], axis=-1, keepdims=True)
        Nc, cov = render_normals(mesh, np.array(info["matrix_world"]), ortho, R)
        rim_ok, rim_t = rim_targets(m.astype(np.float32))
        cur = n.copy()
        for it in range(a.iters):
            ref = Nc.astype(np.float64).copy()
            w = (m & cov).astype(np.float64)
            ang = np.degrees(np.arccos(np.clip((cur * ref).sum(-1), -1, 1)))
            w *= np.clip(1 - ang / a.max_deg, 0, 1) ** 2          # soft robust weight
            # outline: the exact normal, allowing the estimator's slight tilt toward the camera
            rt = rim_t.astype(np.float64).copy(); rt[..., 2] = np.clip(cur[..., 2], 0, 0.35)
            rt /= np.maximum(np.linalg.norm(rt, axis=-1, keepdims=True), 1e-9)
            ref[rim_ok] = rt[rim_ok]; w[rim_ok] = a.rim_weight
            field, den = rotation_field(cur, ref, w, a.sigma * R)
            conf = np.clip(den / (np.percentile(den[m], 90) + 1e-9), 0, 1)[..., None]
            cur = apply_rotation(cur, field * conf)
            cur /= np.maximum(np.linalg.norm(cur, axis=-1, keepdims=True), 1e-9)
        cur[~m] = 0
        cur[m & (cur[..., 2] < 0), 2] = 0
        cur[m] /= np.maximum(np.linalg.norm(cur[m], axis=-1, keepdims=True), 1e-9)
        np.save(os.path.join(a.out, "normals", f"{v}.npy"), cur.astype(np.float32))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
