"""8뷰 뼈대 융합: 실루엣 hull + DA3 깊이 8장 → 하나의 표면 (TSDF).

정답 데이터는 채점에만 쓰고, 융합 자체는 사진(마스크)·DA3 깊이·카메라만 쓴다.
출력: work/fused.ply, work/fused_depth.npz (뷰별 융합 표면 깊이, 1024px)
사용: python scripts/fuse_skeleton.py [--voxel 0.003]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import trimesh
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topo.raster import render  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--depth", default="work/da3_mv504_pose.npz")
ap.add_argument("--voxel", type=float, default=0.003)
ap.add_argument("--trunc", type=float, default=0.015)
ap.add_argument("--mode", choices=["tsdf", "carve"], default="carve",
                help="tsdf: 가중 평균 / carve: hull에서 확실한 빈 공간만 깎기")
ap.add_argument("--free", type=float, default=0.03, help="carve: 표면 앞 이 거리 이상이면 '확실한 빈 공간'")
ap.add_argument("--votes", type=int, default=2, help="carve: 빈 공간 판정에 필요한 뷰 수")
args = ap.parse_args()
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:5.1f}s] {msg}", flush=True)


cams = json.load(open("inputs/lucy_8v/cameras.json"))
views = cams["views"]
H, W = cams["height"], cams["width"]
alphas = [cv2.imread(f"inputs/lucy_8v/{v['name']}.png", cv2.IMREAD_UNCHANGED)[..., 3] / 255.0 for v in views]
masks = [a >= 0.5 for a in alphas]
# 경계에서 과하게 깎지 않도록 1px 여유
masks_d = [cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for m in masks]
Ks = [np.array(v["K"]) for v in views]
Rs = [np.array(v["R"]) for v in views]
ts = [np.array(v["t"]) for v in views]
da3 = np.load(args.depth)
D_lo, C_lo = da3["depth"], da3["conf"]


def project(P, i):
    Pc = P @ Rs[i].T + ts[i]
    z = Pc[:, 2]
    u = Ks[i][0, 0] * Pc[:, 0] / z + Ks[i][0, 2]
    v = Ks[i][1, 1] * Pc[:, 1] / z + Ks[i][1, 2]
    return u, v, z


def carve(P):
    keep = np.ones(len(P), bool)
    for i in range(len(views)):
        u, v, _ = project(P, i)
        x, y = np.floor(u).astype(int), np.floor(v).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        ins = np.zeros(len(P), bool)
        ins[ok] = masks_d[i][y[ok], x[ok]]
        keep &= ins
    return keep


# 1) 거친 hull로 범위 잡기
g = np.arange(-0.6, 0.6, 0.01)
gy = np.arange(-0.05, 1.05, 0.01)
X, Y, Z = np.meshgrid(g, gy, g, indexing="ij")
P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
k = carve(P)
lo, hi = P[k].min(0) - 0.02, P[k].max(0) + 0.02
log(f"coarse hull bbox {np.round(lo, 3)} ~ {np.round(hi, 3)}")

# 2) 정밀 격자
vx = args.voxel
axes = [np.arange(lo[d], hi[d], vx) for d in range(3)]
shape = tuple(len(a) for a in axes)
X, Y, Z = np.meshgrid(*axes, indexing="ij")
P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
del X, Y, Z
hull = carve(P)
log(f"grid {shape} = {len(P) / 1e6:.1f}M voxels, hull {hull.mean() * 100:.1f}%")

# 3) 뷰별 hull 깊이 (hull 복셀을 투영해 가장 가까운 z) → DA3 깊이 스케일 정렬
Ph = P[hull]
aligned = []
for i in range(len(views)):
    u, v, z = project(Ph, i)
    x, y = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    hd = np.full(H * W, np.inf)
    np.minimum.at(hd, y[ok] * W + x[ok], z[ok])
    hd = hd.reshape(H, W)
    hd = cv2.erode(np.where(np.isfinite(hd), hd, 1e9).astype(np.float32), np.ones((3, 3)))  # 복셀 사이 틈 메우기
    d = cv2.resize(D_lo[i], (W, H), interpolation=cv2.INTER_LINEAR)
    sel = masks[i] & (hd < 1e8)
    # hull은 실제 표면보다 항상 앞(가깝)다 → 가장 앞쪽 부분(볼록부)에서 맞춘다: 반복 재가중 L1
    A = np.stack([d[sel], np.ones(sel.sum())], 1)
    b = hd[sel]
    wts = np.ones(len(b))
    for _ in range(10):
        s, c = np.linalg.lstsq(A * wts[:, None], b * wts, rcond=None)[0]
        r = s * d[sel] + c - b
        wts = 1.0 / np.sqrt(np.abs(r) + 1e-3)
    aligned.append((s * cv2.resize(D_lo[i], (W, H), interpolation=cv2.INTER_LINEAR) + c).astype(np.float32))
    log(f"{views[i]['name']} align scale {s:.3f} shift {c:.3f}")

# 4) TSDF 융합 (hull 안쪽 복셀만)
tsdf = np.zeros(len(Ph), np.float32)
wsum = np.zeros(len(Ph), np.float32)
tr = args.trunc
for i in range(len(views)):
    u, v, z = project(Ph, i)
    x, y = np.floor(u).astype(int), np.floor(v).astype(int)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    dd = np.zeros(len(Ph), np.float32)
    dd[ok] = aligned[i][y[ok], x[ok]]
    inm = np.zeros(len(Ph), bool)
    inm[ok] = masks[i][y[ok], x[ok]]
    sdf = dd - z                                   # +: 표면 앞(빈 공간), -: 표면 뒤
    use = inm & (sdf > -tr)
    conf = cv2.resize(C_lo[i], (W, H))
    wv = np.zeros(len(Ph), np.float32)
    wv[ok] = conf[y[ok], x[ok]]
    wv = wv / (np.median(conf[masks[i]]) + 1e-6)
    s_cl = np.clip(sdf, -tr, tr)
    tsdf[use] += (s_cl * wv)[use]
    wsum[use] += wv[use]
if args.mode == "tsdf":
    field = np.full(len(P), tr, np.float32)           # hull 밖 = 빈 공간
    field[hull] = np.where(wsum > 0, tsdf / np.maximum(wsum, 1e-9), -tr)
    field = field.reshape(shape)
    level, pad = 0.0, tr
else:
    free_votes = np.zeros(len(Ph), np.int16)
    for i in range(len(views)):
        u, v, z = project(Ph, i)
        x, y = np.floor(u).astype(int), np.floor(v).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        dd = np.full(len(Ph), -np.inf, np.float32)
        dd[ok] = np.where(masks[i][y[ok], x[ok]], aligned[i][y[ok], x[ok]], -np.inf)
        free_votes += (dd - z > args.free)
    occ = np.zeros(len(P), np.float32)
    occ[hull] = free_votes < args.votes
    from scipy.ndimage import gaussian_filter
    field = -gaussian_filter(occ.reshape(shape), 1.0)   # 음수 = 속 (tsdf와 부호 맞춤)
    level, pad = -0.5, 0.0
    log(f"carve: hull 중 {(free_votes >= args.votes).mean() * 100:.1f}% 깎음")
log("fusion done")

vs, fs, _, _ = measure.marching_cubes(np.pad(field, 1, constant_values=pad), level, spacing=(vx, vx, vx))
vs += lo - vx
mesh = trimesh.Trimesh(vs, fs[:, ::-1])
mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))
trimesh.smoothing.filter_taubin(mesh, iterations=10)
mesh.export("work/fused.ply")
log(f"mesh {len(mesh.faces)} faces")

# 5) 채점: 뷰별 융합 깊이 → 법선 오차, 실루엣 IoU
out_depth = []
print("view      IoU     법선오차(중앙°)  DA3정렬 법선오차")
for i in range(len(views)):
    r = render(mesh.vertices, mesh.faces, Ks[i], Rs[i], ts[i], H, W, ssaa=1)
    fd = r["depth"]
    out_depth.append(fd)
    iou = (masks[i] & (fd > 0)).sum() / (masks[i] | (fd > 0)).sum()
    gt_n = np.load(f"data/lucy/turnaround_8v/{views[i]['name']}_normal.npy") @ Rs[i].T
    ev = cv2.erode(masks[i].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    def nerr(dmap):
        yy, xx = np.mgrid[0:H, 0:W] + 0.5
        Pc = np.dstack([(xx - Ks[i][0, 2]) / Ks[i][0, 0] * dmap, (yy - Ks[i][1, 2]) / Ks[i][1, 1] * dmap, dmap])
        dx = np.zeros_like(Pc)
        dy = np.zeros_like(Pc)
        dx[:, :-1] = Pc[:, 1:] - Pc[:, :-1]
        dy[:-1] = Pc[1:] - Pc[:-1]
        n = np.cross(dy, dx)
        n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
        e = ev & (dmap > 0) & np.isfinite(n).all(-1)
        return np.median(np.degrees(np.arccos(np.clip((n[e] * gt_n[e]).sum(-1), -1, 1))))

    print(f"{views[i]['name']}  {iou:.4f}   {nerr(fd):6.2f}          {nerr(np.where(masks[i], aligned[i], 0)):6.2f}")
np.savez("work/fused_depth.npz", depth=np.array(out_depth, np.float32))
log("done")
