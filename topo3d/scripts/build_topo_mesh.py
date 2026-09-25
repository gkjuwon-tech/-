"""토폴로지 ID 통합 메시: 8뷰 법선 → 뷰별 깊이 → 소유권(가장 정면으로 본 뷰) → 쿼드 → 하나의 메시.

사용: python scripts/build_topo_mesh.py [--normals work/moge_zoom_fused.npz --key zoom]
출력: work/topo/topo_mesh.obj (쿼드, 뷰별 그룹 = 토폴로지 ID), results/topo/*.png, 채점표
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from topo.integrate import integrate_normals  # noqa: E402
from topo.lift import lift  # noqa: E402
from topo.raster import quads_to_tris, render  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--normals", default="work/moge_zoom_fused.npz")
ap.add_argument("--key", default="zoom")
ap.add_argument("--skeleton", default="work/fused_depth.npz")
ap.add_argument("--tol", type=float, default=0.02, help="가시성 판정 깊이 허용 오차")
ap.add_argument("--min-region", type=int, default=30, help="이보다 작은 소유 조각/구멍은 정리")
ap.add_argument("--detail-sigma", type=float, default=8.0,
                help="적분 깊이에서 이 크기(px)보다 큰 굴곡은 버리고 뼈대를 따름 (뷰 간 표류 방지). 0이면 끔")
ap.add_argument("--out", default="work/topo")
args = ap.parse_args()
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:5.1f}s] {m}", flush=True)


cams = json.load(open("inputs/lucy_8v/cameras.json"))
views = cams["views"]
H, W = cams["height"], cams["width"]
NV = len(views)
Ks = [np.array(v["K"]) for v in views]
Rs = [np.array(v["R"]) for v in views]
ts = [np.array(v["t"]) for v in views]
Cs = [np.array(v["C"]) for v in views]
masks = [cv2.imread(f"inputs/lucy_8v/{v['name']}.png", cv2.IMREAD_UNCHANGED)[..., 3] >= 128 for v in views]
Nc = np.nan_to_num(np.load(args.normals)[args.key].astype(np.float64))
Nc /= np.linalg.norm(Nc, axis=-1, keepdims=True) + 1e-9
skel = np.load(args.skeleton)["depth"].astype(np.float64)


def fill_nearest(z, valid):
    if valid.all():
        return z
    _, idx = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    yy, xx = np.nonzero(valid)
    lut = np.zeros((idx.max() + 1, 2), int)
    lut[idx[valid]] = np.stack([yy, xx], 1)
    s = lut[idx]
    return z[s[..., 0], s[..., 1]]


# 1) 뷰별 깊이 (뼈대에 앵커한 법선 적분)
Z = []
for i in range(NV):
    za = fill_nearest(skel[i], skel[i] > 0)
    za = cv2.GaussianBlur(za.astype(np.float32), (0, 0), 2).astype(np.float64)
    zi = integrate_normals(Nc[i], masks[i], za, Ks[i][0, 0])
    if args.detail_sigma > 0:
        m = masks[i].astype(np.float32)
        d = ((zi - za) * m).astype(np.float32)
        low = cv2.GaussianBlur(d, (0, 0), args.detail_sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), args.detail_sigma), 1e-4)
        zi = za + (d - low) * m
    Z.append(zi)
log("뷰별 적분 완료")

# 진단: 뷰 간 깊이 불일치 (i의 점을 j에 투영했을 때 j 깊이와 차이, 마스크 안쪽)
dis = []
for i in range(NV):
    j = (i + 1) % NV
    K = Ks[i]
    zz = Z[i][masks[i]]
    yy_, xx_ = np.nonzero(masks[i])
    Xc = np.stack([(xx_ + 0.5 - K[0, 2]) / K[0, 0] * zz, (yy_ + 0.5 - K[1, 2]) / K[1, 1] * zz, zz], 1)
    P = (Xc - ts[i]) @ Rs[i]
    Pc = P @ Rs[j].T + ts[j]
    u = (Ks[j][0, 0] * Pc[:, 0] / Pc[:, 2] + Ks[j][0, 2]).astype(int)
    v = (Ks[j][1, 1] * Pc[:, 1] / Pc[:, 2] + Ks[j][1, 2]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ok[ok] &= masks[j][v[ok], u[ok]]
    diff = np.abs(Z[j][v[ok], u[ok]] - Pc[ok, 2])
    dis.append(np.mean(diff < args.tol))
log("이웃 뷰와 깊이가 tol 안에서 맞는 비율: " + " ".join(f"{d:.2f}" for d in dis))

uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)


def unproject(i, z):
    K = Ks[i]
    Xc = np.dstack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z])
    return (Xc - ts[i]) @ Rs[i]


# 2) 소유권 심사: 가장 정면으로 본 뷰가 주인
owned = []
for i in range(NV):
    m = masks[i]
    P = unproject(i, Z[i])[m]
    Nw = Nc[i][m] @ Rs[i]                       # 카메라 → 월드
    def facing(C):
        d = C - P
        return (Nw * d).sum(-1) / (np.linalg.norm(d, axis=-1) + 1e-12)
    fi = facing(Cs[i])
    lose = np.zeros(len(P), bool)
    for j in range(NV):
        if j == i:
            continue
        Pc = P @ Rs[j].T + ts[j]
        u = Ks[j][0, 0] * Pc[:, 0] / Pc[:, 2] + Ks[j][0, 2]
        v = Ks[j][1, 1] * Pc[:, 1] / Pc[:, 2] + Ks[j][1, 2]
        x, y = np.floor(u).astype(int), np.floor(v).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        vis = np.zeros(len(P), bool)
        vis[ok] = masks[j][y[ok], x[ok]] & (np.abs(Z[j][y[ok], x[ok]] - Pc[ok, 2]) < args.tol)
        fj = facing(Cs[j])
        lose |= vis & ((fj > fi) | ((fj == fi) & (j < i)))
    om = np.zeros_like(m)
    om[m] = ~lose
    # 모자이크 정리: 작은 소유 조각 제거, 작은 구멍 메우기
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(om.astype(np.uint8), connectivity=4)
    small = np.isin(lab, np.nonzero(st[:, cv2.CC_STAT_AREA] < args.min_region)[0]) & om
    om &= ~small
    hole = m & ~om
    n_lab, lab, st, _ = cv2.connectedComponentsWithStats(hole.astype(np.uint8), connectivity=4)
    om |= np.isin(lab, np.nonzero(st[:, cv2.CC_STAT_AREA] < args.min_region)[0]) & hole & (lab > 0)
    owned.append(om)
    log(f"{views[i]['name']} 소유 {om.sum() / m.sum() * 100:5.1f}% ({om.sum()} 쿼드)")

# 3) 쿼드 건설 + 통합
Vs, Qs, G = [], [], []
off = 0
for i in range(NV):
    r = lift(owned[i], Z[i], Ks[i], Rs[i], ts[i])
    Vs.append(r["V"])
    Qs.append(r["Q"] + off)
    G.append(np.full(len(r["Q"]), i))
    off += len(r["V"])
V = np.concatenate(Vs)
Q = np.concatenate(Qs)
G = np.concatenate(G)
log(f"통합 메시: 꼭짓점 {len(V)}, 쿼드 {len(Q)} (삼각형 0)")

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "topo_mesh.obj"), "w") as fp:
    fp.write("# topo3d topology-ID quad mesh (group = owner view)\n")
    np.savetxt(fp, V, fmt="v %.6f %.6f %.6f")
    for i in range(NV):
        fp.write(f"g owner_{views[i]['name']}\n")
        np.savetxt(fp, Q[G == i] + 1, fmt="f %d %d %d %d")

# 4) 채점
fn = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
vn = np.zeros_like(V)
for k in range(4):
    np.add.at(vn, Q[:, k], fn)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
tris = quads_to_tris(Q)
tri_owner = np.concatenate([G, G])
PAL = np.array([[70, 200, 120], [60, 200, 240], [230, 120, 90], [150, 90, 230],
                [240, 180, 60], [90, 160, 250], [200, 90, 180], [120, 230, 200]], np.uint8)
print("view      IoU     구멍%    법선오차(중앙°)")
ious, nerrs, shaded, idviz = [], [], [], []
RL = np.array([0.8, -0.1, -0.6])
RL /= np.linalg.norm(RL)
for i in range(NV):
    r = render(V, tris, Ks[i], Rs[i], ts[i], H, W, attr=vn.astype(np.float32), ssaa=1)
    cov = r["face"] >= 0
    m = masks[i]
    iou = (cov & m).sum() / (cov | m).sum()
    holes = (m & ~cov).sum() / m.sum() * 100
    gt = np.load(f"data/lucy/turnaround_8v/{views[i]['name']}_normal.npy")
    ev = (cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & cov
    nn = r["attr"] / (np.linalg.norm(r["attr"], axis=-1, keepdims=True) + 1e-12)
    ok = ev & cov & (np.linalg.norm(gt, axis=-1) > 0.5)
    a = np.degrees(np.arccos(np.clip((nn[ok] * gt[ok]).sum(-1), -1, 1)))
    ious.append(iou)
    nerrs.append(np.median(a))
    print(f"{views[i]['name']}  {iou:.4f}  {holes:5.2f}    {np.median(a):6.2f}")
    s = np.clip(0.15 + 0.85 * np.clip((nn @ Rs[i].T) @ RL, 0, 1), 0, 1)
    img = np.where(cov, s * 235, 100).astype(np.uint8)
    shaded.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    col = np.full((H, W, 3), 25, np.uint8)
    col[cov] = (PAL[tri_owner[r["face"][cov]]] * (0.35 + 0.65 * s[cov])[:, None]).astype(np.uint8)
    idviz.append(col)
print(f"평균 IoU {np.mean(ious):.4f}, 평균 법선오차 {np.mean(nerrs):.2f}°")

# 정답 메시와 3D 거리 (Chamfer): 렌더 때 yaw 180 적용한 정답과 비교
gt_mesh = trimesh.load("data/lucy/lucy_4m.ply", process=False)
gv = gt_mesh.vertices * np.array([-1, 1, -1])
gt_pts = trimesh.Trimesh(gv, gt_mesh.faces, process=False).sample(300000)
our = trimesh.Trimesh(V, tris, process=False).sample(300000)
sk = trimesh.load("work/fused.ply").sample(300000)
tg = cKDTree(gt_pts)
for name, pts in (("뼈대(융합)", sk), ("토폴로지 메시", our)):
    d1 = tg.query(pts)[0]
    d2 = cKDTree(pts).query(gt_pts)[0]
    print(f"Chamfer {name}: 정확도(우리→정답) 중앙 {np.median(d1) * 100:.3f}%  완성도(정답→우리) 중앙 {np.median(d2) * 100:.3f}%  (키 대비)")

os.makedirs("results/topo", exist_ok=True)
sheet = lambda ims: np.vstack([np.hstack(ims[:4]), np.hstack(ims[4:])])
cv2.imwrite("results/topo/shaded_8v.png", cv2.resize(sheet(shaded), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA))
cv2.imwrite("results/topo/topology_id_8v.png", cv2.resize(sheet(idviz), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA))
m = masks[0]
ys, xs = np.nonzero(m)
y0, cx = ys.min(), (xs.min() + xs.max()) // 2
face = np.hstack([shaded[0][y0 + 40:y0 + 300, cx - 160:cx + 160], idviz[0][y0 + 40:y0 + 300, cx - 160:cx + 160]])
cv2.imwrite("results/topo/face_front.png", cv2.resize(face, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
log("done")
