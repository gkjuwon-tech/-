"""Render a ground-truth mesh into the input format of gkjuwon-tech/3d stage2.py:
  <out>/views/{cameras.json, mask/, rgb/}, <out>/normals/<view>.npy (camera space, x right, y up, z toward camera)
Point-splat renderer with 2x supersampling (CPU, no Blender).
usage: python make_views.py MESH OUT --layout six|zoo14 [--res 1024]"""
import argparse, json, os
import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

LAYOUTS = {
    "six": [["01_front", 0, 0], ["d045", 45, 0], ["02_right", 90, 0], ["03_back", 180, 0],
            ["04_left", 270, 0], ["d315", 315, 0]],
    "zoo14": [["01_front", 0, 0], ["02_right", 90, 0], ["03_back", 180, 0], ["04_left", 270, 0],
              ["05_top", 0, 89.99], ["06_bottom", 0, -89.99],
              ["07_az45_up", 45, 45], ["08_az135_up", 135, 45], ["09_az225_up", 225, 45],
              ["10_az315_up", 315, 45], ["11_az45_dn", 45, -45], ["12_az135_dn", 135, -45],
              ["13_az225_dn", 225, -45], ["14_az315_dn", 315, -45]],
}


def camera(az, el, dist=5.0):
    a, e = np.radians(az), np.radians(el)
    loc = dist * np.array([np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
    back = loc / np.linalg.norm(loc)
    upw = np.array([0, 0, 1.0]) if abs(el) < 89 else np.array([0, 1.0 if el > 0 else -1.0, 0]) * 1.0
    right = np.cross(upw, back); right /= np.linalg.norm(right)
    up = np.cross(back, right)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, back, loc
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh"); ap.add_argument("out")
    ap.add_argument("--layout", default="six"); ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=8_000_000)
    ap.add_argument("--yup", action="store_true", help="mesh is Y-up (Stanford scans)")
    a = ap.parse_args()
    m = trimesh.load(a.mesh)
    v = m.vertices.astype(np.float64)
    if a.yup:
        v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
    # normalization recorded the way eval_hull.py expects: height 1, centred, base at z=0
    lo, hi = v.min(0), v.max(0)
    scale = 1.0 / (hi[2] - lo[2])
    offset = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]) * scale
    gt_path = os.path.join(a.out, "gt_mesh.ply")
    os.makedirs(os.path.join(a.out, "views", "mask"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "views", "rgb"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "normals"), exist_ok=True)
    trimesh.Trimesh(v, m.faces, process=False).export(gt_path)          # GT in its own frame
    V = v * scale + offset
    mesh = trimesh.Trimesh(V, m.faces, process=False)
    vn = mesh.vertex_normals
    pts, fi = trimesh.sample.sample_surface(mesh, a.samples, seed=0)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[fi], pts)
    nrm = (vn[mesh.faces[fi]] * bary[:, :, None]).sum(1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    ext = V.max(0) - V.min(0)
    ortho = float(np.linalg.norm(ext) * 1.08)
    center = (V.max(0) + V.min(0)) / 2
    R, S = a.res, 2 * a.res
    meta = {"ortho_scale": ortho, "resolution": [R, R], "yaw_deg": 0.0,
            "normalization": {"applied_scale": scale, "applied_offset": offset.tolist()}, "views": {}}
    key = np.array([-0.35, 0.5, 0.79]); key /= np.linalg.norm(key)
    for name, az, el in LAYOUTS[a.layout]:
        M = camera(az, el); M[:3, 3] += center
        right, up, back, loc = M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3]
        rel = pts - loc
        col = ((rel @ right) / ortho + 0.5) * S - 0.5
        row = (0.5 - (rel @ up) / ortho) * S - 0.5
        w = -(rel @ back)
        ci, ri = np.round(col).astype(int), np.round(row).astype(int)
        ok = (ci >= 0) & (ci < S) & (ri >= 0) & (ri < S)
        ci, ri, w2, n2 = ci[ok], ri[ok], w[ok], nrm[ok]
        zb = np.full((S, S), np.inf); np.minimum.at(zb, (ri, ci), w2)
        win = w2 <= zb[ri, ci] + 1e-9
        nc = np.zeros((S, S, 3)); nc[ri[win], ci[win]] = np.stack([n2[win] @ right, n2[win] @ up, n2[win] @ back], -1)
        cov = np.isfinite(zb)
        full = ndimage.binary_fill_holes(ndimage.binary_closing(cov, iterations=2))
        idx = ndimage.distance_transform_edt(~cov, return_distances=False, return_indices=True)
        nc = nc[idx[0], idx[1]]
        # downsample 2x: coverage -> anti-aliased mask, normals -> averaged
        mask = full.reshape(R, 2, R, 2).mean((1, 3))
        n = (nc * full[..., None]).reshape(R, 2, R, 2, 3).sum((1, 3))
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
        n[mask < 0.5] = 0
        n[(mask >= 0.5) & (n[..., 2] < 0), 2] *= -1
        np.save(os.path.join(a.out, "normals", f"{name}.npy"), n.astype(np.float32))
        Image.fromarray((mask * 255).round().astype(np.uint8)).save(os.path.join(a.out, "views", "mask", f"{name}.png"))
        s = np.clip(0.25 + 0.75 * np.clip(n @ key, 0, None), 0, 1)
        rgb = np.full((R, R, 3), 128.0); sel = mask >= 0.5
        rgb[sel] = s[sel, None] * np.array([150, 190, 160])
        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(a.out, "views", "rgb", f"{name}.png"))
        meta["views"][name] = {"matrix_world": M.tolist()}
        print(name, "coverage", round(float((mask > 0.5).mean()), 3), flush=True)
    json.dump(meta, open(os.path.join(a.out, "views", "cameras.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
