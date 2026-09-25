"""8뷰 공동 법선 맞춤: 공유 메시의 꼭짓점을 법선 방향으로만 움직여 8뷰 목표 법선(MoGe 확대)에 동시에 맞춘다.

가시성(어느 픽셀이 어느 삼각형의 어디인지)은 numba 래스터로 구하고 주기적으로 갱신한다.
그 사이에는 고정된 픽셀-삼각형 대응 위에서 torch로 명암 손실을 미분한다.
사용: python scripts/sfs_multiview.py [--mesh work/fused.ply] [--iters 300]
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topo.raster import render  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mesh", default="work/fused.ply")
ap.add_argument("--targets", default="work/moge_zoom_fused.npz")
ap.add_argument("--key", default="zoom")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--rerast", type=int, default=100, help="몇 번마다 가시성 갱신")
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--lap", type=float, default=0.3, help="변위 라플라시안 매끄러움")
ap.add_argument("--nsmooth", type=float, default=0.05, help="이웃 법선 매끄러움 (줄무늬 억제)")
ap.add_argument("--subdiv", type=int, default=0, help="루프 분할 횟수 (디테일 해상도 ↑)")
ap.add_argument("--time", type=float, default=160)
ap.add_argument("--out", default="work/normal_fit")
args = ap.parse_args()
torch.set_num_threads(4)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:5.1f}s] {m}", flush=True)


cams = json.load(open("inputs/lucy_8v/cameras.json"))
views = cams["views"]
H, W = cams["height"], cams["width"]
Ks = [np.array(v["K"]) for v in views]
Rs = [np.array(v["R"]) for v in views]
ts = [np.array(v["t"]) for v in views]
Cs = [np.array(v["C"]) for v in views]
imgs, masks = [], []
for v in views:
    rgba = cv2.imread(f"inputs/lucy_8v/{v['name']}.png", cv2.IMREAD_UNCHANGED)
    imgs.append(rgba[..., 1] / 255.0 / (0.92 * 1.05))
    masks.append(cv2.erode((rgba[..., 3] >= 128).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)

TN = np.nan_to_num(np.load(args.targets)[args.key].astype(np.float64))
TN /= np.linalg.norm(TN, axis=-1, keepdims=True) + 1e-9
TNW = [TN[i] @ Rs[i] for i in range(len(views))]        # 월드 좌표 목표 법선

mesh = trimesh.load(args.mesh, process=True)
for _ in range(args.subdiv):
    mesh = mesh.subdivide()
V0 = mesh.vertices.astype(np.float64)
Fc = mesh.faces.astype(np.int64)
log(f"mesh {len(V0)} verts {len(Fc)} faces")

Ft = torch.tensor(Fc)
V0t = torch.tensor(V0)


def vnormals(V):
    fn = torch.cross(V[Ft[:, 1]] - V[Ft[:, 0]], V[Ft[:, 2]] - V[Ft[:, 0]], dim=-1)
    vn = torch.zeros_like(V)
    for k in range(3):
        vn = vn.index_add(0, Ft[:, k], fn)
    return vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)


# 바깥쪽 법선 확인: 정면 카메라에서 보이는 꼭짓점이 카메라를 향해야 함
N0 = vnormals(V0t)
r0 = render(V0, Fc, Ks[0], Rs[0], ts[0], H, W, ssaa=1)
vis_f = np.unique(r0["face"][r0["face"] >= 0])
fn0 = np.cross(V0[Fc[vis_f, 1]] - V0[Fc[vis_f, 0]], V0[Fc[vis_f, 2]] - V0[Fc[vis_f, 0]])
tow = (fn0 * (Cs[0] - V0[Fc[vis_f]].mean(1))).sum(-1)
if (tow > 0).mean() < 0.5:
    Fc = Fc[:, ::-1].copy()
    Ft = torch.tensor(Fc)
    N0 = vnormals(V0t)
    log("법선 뒤집음")
N0 = N0.detach()

# 균일 라플라시안 (이웃 평균과의 차)
edges = np.concatenate([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
edges = np.unique(np.sort(edges, 1), axis=0)
ei = torch.tensor(edges)
deg = torch.zeros(len(V0), dtype=torch.float64).index_add(0, ei.reshape(-1), torch.ones(ei.numel(), dtype=torch.float64))


def laplacian(x):
    s = torch.zeros_like(x).index_add(0, ei[:, 0], x[ei[:, 1]]).index_add(0, ei[:, 1], x[ei[:, 0]])
    return x - s / deg.clamp(min=1)[:, None] if x.dim() == 2 else x - s / deg.clamp(min=1)


# 조명 (렌더러와 같음: 카메라 좌표 고정 → 뷰마다 월드 방향이 다름)
def lights_world(R):
    out = []
    for d, k in [([-0.5, 0.7, 0.5], 0.75), ([0.6, 0.2, 0.6], 0.30), ([0.0, 0.5, -0.9], 0.25)]:
        c = np.array([d[0], -d[1], -d[2]], float)
        c /= np.linalg.norm(c)
        out.append((torch.tensor(R.T @ c), k))
    h = np.array([-0.5, -0.7, -1.0])
    h /= np.linalg.norm(h)
    return out, torch.tensor(R.T @ h)


LW = [lights_world(R) for R in Rs]


def shade(n, i):
    ls, hv = LW[i]
    c = torch.full(n.shape[:1], 0.18, dtype=n.dtype)
    for d, k in ls:
        c = c + k * F.softplus((n * d).sum(-1) * 40) / 40
    return c + 0.12 * torch.clamp((n * hv).sum(-1), min=0) ** 24


def rasterize(V):
    """뷰별 (픽셀 밝기, 삼각형 꼭짓점 3개, 무게중심좌표) 샘플."""
    Vu = V[Fc].reshape(-1, 3)                         # 삼각형마다 꼭짓점 복제 → 무게중심 좌표를 속성으로
    Tu = np.arange(len(Vu)).reshape(-1, 3)
    bary = np.tile(np.eye(3), (len(Fc), 1)).astype(np.float32)
    samples = []
    for i in range(len(views)):
        r = render(Vu, Tu, Ks[i], Rs[i], ts[i], H, W, attr=bary, ssaa=1)
        sel = masks[i] & (r["face"] >= 0) & (np.linalg.norm(TNW[i], axis=-1) > 0.5)
        fid = r["face"][sel]
        b = r["attr"][sel]
        samples.append((torch.tensor(TNW[i][sel]), torch.tensor(Fc[fid]), torch.tensor(b.astype(np.float64))))
    return samples


disp = torch.zeros(len(V0), dtype=torch.float64, requires_grad=True)
opt = torch.optim.Adam([disp], lr=args.lr)
samples = None
for it in range(args.iters):
    if it % args.rerast == 0:
        V = (V0t + disp.detach()[:, None] * N0).numpy()
        samples = rasterize(V)
        log(f"rasterize: 샘플 {sum(len(s[0]) for s in samples)}")
    opt.zero_grad()
    V = V0t + disp[:, None] * N0
    vn = vnormals(V)
    l_sh = 0
    for i, (I, tri, b) in enumerate(samples):
        n = (vn[tri] * b[..., None]).sum(1)
        n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
        l_sh = l_sh + (1 - (n * I).sum(-1)).mean()          # I = 목표 법선
    l_sh = l_sh / len(samples)
    l_lap = args.lap * (laplacian(disp) ** 2).mean() / 1e-6
    l_ns = args.nsmooth * ((vn[ei[:, 0]] - vn[ei[:, 1]]) ** 2).sum(-1).mean()
    loss = l_sh + l_lap + l_ns
    loss.backward()
    opt.step()
    if it % 25 == 0 or it == args.iters - 1:
        log(f"it {it} shading {l_sh.item():.5f} lap {l_lap.item():.5f} nsm {l_ns.item():.5f} |disp| {disp.abs().mean().item():.5f}")
    if time.time() - t0 > args.time:
        log("시간 제한 도달, 중단")
        break

Vf = (V0t + disp.detach()[:, None] * N0).numpy()
os.makedirs(args.out, exist_ok=True)
trimesh.Trimesh(Vf, Fc, process=False).export(os.path.join(args.out, "refined.ply"))

# ---------- 채점: 뷰별 법선 오차 (메시 꼭짓점 법선 보간) ----------
vn0 = vnormals(V0t).numpy()
vnf = vnormals(torch.tensor(Vf)).numpy()
RL = np.array([0.8, -0.1, -0.6])
RL /= np.linalg.norm(RL)
print("view      뼈대(°)  보정(°)   <11.25° 뼈대→보정")
tiles = []
for i, v in enumerate(views):
    gt_n = np.load(f"data/lucy/turnaround_8v/{v['name']}_normal.npy")
    res = []
    for Vx, vnx in ((V0, vn0), (Vf, vnf)):
        r = render(Vx, Fc, Ks[i], Rs[i], ts[i], H, W, attr=vnx.astype(np.float32), ssaa=1)
        n = r["attr"] / (np.linalg.norm(r["attr"], axis=-1, keepdims=True) + 1e-12)
        e = masks[i] & (r["face"] >= 0) & (np.linalg.norm(gt_n, axis=-1) > 0.5)
        a = np.degrees(np.arccos(np.clip((n[e] * gt_n[e]).sum(-1), -1, 1)))
        res.append((np.median(a), (a < 11.25).mean(), n, r["face"] >= 0))
    print(f"{v['name']}  {res[0][0]:6.2f}   {res[1][0]:6.2f}    {res[0][1]:.2f} → {res[1][1]:.2f}")
    if i == 0:
        Rc = Rs[0]
        for n, hit, lab in ((gt_n, np.linalg.norm(gt_n, axis=-1) > 0.5, "GT (side light)"),
                            (res[0][2], res[0][3], "skeleton"), (res[1][2], res[1][3], "8-view normal fit")):
            L = Rc.T @ RL
            s = 0.15 + 0.85 * np.clip(n @ L, 0, 1)
            s[~hit] = 0.4
            ys, xs = np.nonzero(masks[0])
            y0, cx = ys.min(), (xs.min() + xs.max()) // 2
            t = (np.clip(s, 0, 1) * 255).astype(np.uint8)[y0 + 58:y0 + 278, cx - 150:cx + 150].copy()
            cv2.putText(t, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
            tiles.append(t)
cv2.imwrite(os.path.join(args.out, "face.png"), cv2.resize(np.hstack(tiles), None, fx=2, fy=2,
                                                            interpolation=cv2.INTER_NEAREST))
log("done")
