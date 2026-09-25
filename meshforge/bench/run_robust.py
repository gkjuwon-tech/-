"""'Desert cactus' benchmark: reconstruct the Stanford dragon from Era3D/Wonder3D-style inputs
(6 orthographic views) with monocular-estimator-grade normals, ragged masks and wrong cameras."""
import json, os, sys, time
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from PIL import Image
from bench.gt import load_mesh, render_gt, angular_error
from bench.degrade import degrade_normals, degrade_mask
from mf.recon import reconstruct

od = "bench/out/robust"; os.makedirs(od, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
VIEWS = [0, 45, 90, 180, 270, 315]
v, vn = load_mesh("bench/data/dragon_recon/dragon_vrip.ply")
gt_mesh = trimesh.load("bench/data/dragon_recon/dragon_vrip.ply")
gt_mesh.vertices = v
gt_pts, fi = trimesh.sample.sample_surface(gt_mesh, 150000, seed=0)
gt_nrm = gt_mesh.face_normals[fi]
diag = np.linalg.norm(v.max(0) - v.min(0))

def evaluate(verts, faces, name):
    m = trimesh.Trimesh(verts, faces, process=False)
    p, f2 = trimesh.sample.sample_surface(m, 150000, seed=1)
    n = m.face_normals[f2]
    d1, i1 = cKDTree(gt_pts).query(p); d2, i2 = cKDTree(p).query(gt_pts)
    cd = (d1.mean() + d2.mean()) / 2 / diag * 100
    fs = {}
    for tau in (0.5, 1.0):
        t = tau / 100 * diag
        pr, rc = (d1 < t).mean(), (d2 < t).mean()
        fs[tau] = 2 * pr * rc / (pr + rc + 1e-9)
    nc = np.abs((n * gt_nrm[i1]).sum(1)).mean()
    r = {"chamfer_%diag": round(cd, 3), "F@0.5%": round(fs[0.5], 3), "F@1%": round(fs[1.0], 3), "normal_consistency": round(nc, 3)}
    log(f"{name:<34}", r)
    m.export(f"{od}/{name.split()[0]}.ply")
    return r

gt = {t: render_gt(v, vn, t) for t in VIEWS}
results = {}
conds = {
    "clean": dict(noisy=False),
    "desert": dict(noisy=True, smooth_deg=18, pixel_deg=8, global_deg=6, ragged=2, shift=3, az_err=3),
    "desert_hard": dict(noisy=True, smooth_deg=26, pixel_deg=12, global_deg=10, ragged=3, shift=4, az_err=5),
}
for cname, c in conds.items():
    rng = np.random.default_rng(7)
    masks, normals, errs = {}, {}, []
    for t in VIEWS:
        m, d, N = gt[t]
        if c["noisy"]:
            Nn = degrade_normals(N, m, rng, smooth_deg=c["smooth_deg"], pixel_deg=c["pixel_deg"], global_deg=c["global_deg"])
            errs.append(angular_error(Nn, N, m)["mean"])
            mm, _ = degrade_mask(m, rng, ragged=c["ragged"], shift=c["shift"])
            # wrong camera: the image was really taken a few degrees off the nominal azimuth
            dt = rng.uniform(-c["az_err"], c["az_err"])
            m2, d2, N2 = render_gt(v, vn, t + dt)
            Nn = degrade_normals(N2, m2, rng, smooth_deg=c["smooth_deg"], pixel_deg=c["pixel_deg"], global_deg=c["global_deg"])
            mm, _ = degrade_mask(m2, rng, ragged=c["ragged"], shift=c["shift"])
            masks[t], normals[t] = mm, Nn
        else:
            masks[t], normals[t] = m, N
    if errs:
        log(f"[{cname}] input normal error: mean {np.mean(errs):.1f} deg per view (Marigold-grade target 20-30)")
    (vr, fr), (vh, fh), off, views = reconstruct(masks, normals, calibrate=c["noisy"])
    results[cname] = {"hull": evaluate(vh, fh, f"{cname}_hull  (silhouettes only)"),
                      "recon": evaluate(vr, fr, f"{cname}_recon (hull + normals, TSDF)")}
    if cname == "desert":
        vis = [((normals[t] * 0.5 + 0.5) * 255 * masks[t][..., None]).astype(np.uint8) for t in (0, 90)]
        gtv = [((gt[t][2] * 0.5 + 0.5) * 255 * gt[t][0][..., None]).astype(np.uint8) for t in (0, 90)]
        Image.fromarray(np.concatenate([np.concatenate(gtv, 1), np.concatenate(vis, 1)], 0)).save(f"{od}/inputs_gt_vs_desert.png")
json.dump(results, open(f"{od}/results.json", "w"), indent=1)
