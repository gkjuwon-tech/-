"""음영 보정 v2: (A) 뼈대에 붙잡힌 법선 추정 → (B) 스크린드 Poisson 적분으로 깊이 복원.

v1(깊이 직접 최적화)의 문제: 법선 뒤집힘(뭉치), 모호한 방향으로 흘러내림(줄무늬), 표면 접힘(깨짐).
v2는 픽셀별 법선을 뼈대 법선 근처로 묶어서 추정하고, 한 번의 선형 풀이로 연속 표면을 만든다.
사용: python scripts/sfs_normals.py --base work/fused_depth.npz
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="work/fused_depth.npz")
ap.add_argument("--view", type=int, default=0)
ap.add_argument("--smooth", type=float, default=3.0, help="뼈대 스무딩 σ(px)")
ap.add_argument("--prior", type=float, default=0.02, help="A: 뼈대 법선 쪽으로 묶는 힘")
ap.add_argument("--nsmooth", type=float, default=0.02, help="A: 이웃 법선 매끄러움")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--anchor", type=float, default=1e-3, help="B: 뼈대 깊이에 붙잡는 힘 (스크린드 Poisson)")
ap.add_argument("--out", default="work/sfs2")
args = ap.parse_args()
torch.set_num_threads(4)
t0 = time.time()


def log(m):
    print(f"[{time.time() - t0:5.1f}s] {m}", flush=True)


cams = json.load(open("inputs/lucy_8v/cameras.json"))
v = cams["views"][args.view]
name = v["name"]
K, R = np.array(v["K"]), np.array(v["R"])
f = K[0, 0]
rgba = cv2.imread(f"inputs/lucy_8v/{name}.png", cv2.IMREAD_UNCHANGED)
mask = rgba[..., 3] >= 128
I = rgba[..., 1] / 255.0 / (0.92 * 1.05)       # 렌더러 색 모델 역산 (초록 채널)

# 뼈대 깊이 (빈 곳은 가장 가까운 값으로 채움) + 스무딩
zb = np.load(args.base)["depth"][args.view].astype(np.float64)
if (mask & (zb <= 0)).any():
    _, idx = cv2.distanceTransformWithLabels((zb <= 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    yy, xx = np.nonzero(zb > 0)
    lut = np.zeros((idx.max() + 1, 2), int)
    lut[idx[zb > 0]] = np.stack([yy, xx], 1)
    src = lut[idx]
    zb = zb[src[..., 0], src[..., 1]]
if args.smooth > 0:
    zb = cv2.GaussianBlur(zb.astype(np.float32), (0, 0), args.smooth).astype(np.float64)

ys, xs = np.nonzero(mask)
y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
sl = (slice(y0, y1), slice(x0, x1))
Hc, Wc = y1 - y0, x1 - x0
m = mask[sl]
z_sk = zb[sl]
uu, vv = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
rx, ry = (uu - K[0, 2]) / f, (vv - K[1, 2]) / f


def normals_np(z):
    """중앙 차분 법선 (카메라 좌표, 카메라 쪽 = -z)."""
    P = np.dstack([rx * z, ry * z, z])
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 2
    dy[1:-1] = (P[2:] - P[:-2]) / 2
    n = np.cross(dy, dx)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


n_sk = normals_np(z_sk)

# ---------- A. 법선 추정 ----------
def Ld(d):
    d = np.array([d[0], -d[1], -d[2]], float)
    return torch.tensor(d / np.linalg.norm(d))


lights = [(Ld([-0.5, 0.7, 0.5]), 0.75), (Ld([0.6, 0.2, 0.6]), 0.30), (Ld([0.0, 0.5, -0.9]), 0.25)]
Hs = torch.tensor(np.array([-0.5, -0.7, -1.0]) / np.linalg.norm([-0.5, -0.7, -1.0]))


def shade(n, soft=True):
    c = torch.full(n.shape[:2], 0.18, dtype=n.dtype)
    for d, k in lights:
        x = (n * d).sum(-1)
        # 그림자 구간에서도 기울기가 살아 있도록 부드러운 max(0, x)
        c = c + k * (F.softplus(x * 40) / 40 if soft else torch.clamp(x, min=0))
    c = c + 0.12 * torch.clamp((n * Hs).sum(-1), min=0) ** 24
    return c


inner = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
It = torch.tensor(I[sl])
valid = torch.tensor(inner) & (It < 0.98)
nsk_t = torch.tensor(n_sk)
p = nsk_t.clone().requires_grad_(True)
opt = torch.optim.Adam([p], lr=0.01)
mt = torch.tensor(m)
for it in range(args.iters):
    opt.zero_grad()
    n = p / (p.norm(dim=-1, keepdim=True) + 1e-12)
    l_sh = ((shade(n) - It)[valid] ** 2).mean()
    l_pr = args.prior * ((n - nsk_t) ** 2).sum(-1)[mt].mean()
    dnx = ((n[:, 1:] - n[:, :-1]) ** 2).sum(-1)[mt[:, 1:] & mt[:, :-1]].mean()
    dny = ((n[1:] - n[:-1]) ** 2).sum(-1)[mt[1:] & mt[:-1]].mean()
    l_sm = args.nsmooth * (dnx + dny)
    loss = l_sh + l_pr + l_sm
    loss.backward()
    opt.step()
    with torch.no_grad():                     # 카메라 반대쪽으로 뒤집히지 않게
        p[..., 2].clamp_(max=-0.05)
    if it % 100 == 0 or it == args.iters - 1:
        log(f"A it {it} shading {l_sh.item():.5f} prior {l_pr.item():.5f} smooth {l_sm.item():.5f}")
n_est = (p / p.norm(dim=-1, keepdim=True)).detach().numpy()

# ---------- B. 스크린드 Poisson 적분 ----------
# 원근 대신 국소 정사영 근사: dz/du = -(z/f) nx/nz, dz/dv = -(z/f) ny/nz (z는 뼈대 값)
nz = np.minimum(n_est[..., 2], -0.1)          # 거의 옆을 보는 면의 기울기 폭주 방지
gu = -(z_sk / f) * n_est[..., 0] / nz
gv = -(z_sk / f) * n_est[..., 1] / nz
# 뼈대에서 깊이가 크게 끊기는 곳은 적분 제약에서 뺀다 (날개-몸통 경계 등)
jump_u = np.abs(z_sk[:, 1:] - z_sk[:, :-1]) > 0.01 * z_sk[:, 1:]
jump_v = np.abs(z_sk[1:] - z_sk[:-1]) > 0.01 * z_sk[1:]
idx = -np.ones((Hc, Wc), int)
idx[m] = np.arange(m.sum())
N = int(m.sum())
rows, cols, vals, rhs = [], [], [], []
r = 0
eu = m[:, 1:] & m[:, :-1] & ~jump_u
a, b = idx[:, :-1][eu], idx[:, 1:][eu]
g = ((gu[:, :-1] + gu[:, 1:]) / 2)[eu]
k = len(a)
rows += [np.arange(r, r + k)] * 2
cols += [a, b]
vals += [-np.ones(k), np.ones(k)]
rhs.append(g)
r += k
ev = m[1:] & m[:-1] & ~jump_v
a, b = idx[:-1][ev], idx[1:][ev]
g = ((gv[:-1] + gv[1:]) / 2)[ev]
k = len(a)
rows += [np.arange(r, r + k)] * 2
cols += [a, b]
vals += [-np.ones(k), np.ones(k)]
rhs.append(g)
r += k
lam = np.sqrt(args.anchor)
rows.append(np.arange(r, r + N))
cols.append(np.arange(N))
vals.append(np.full(N, lam))
rhs.append(lam * z_sk[m])
r += N
A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, N))
bvec = np.concatenate(rhs)
AtA = (A.T @ A).tocsc()
Atb = A.T @ bvec
log(f"B solve {N} unknowns")
z_sol, info = spla.cg(AtA, Atb, x0=z_sk[m], rtol=1e-8, maxiter=3000)
log(f"B done (cg info {info})")
z_new = z_sk.copy()
z_new[m] = z_sol

# ---------- 평가 ----------
gt_n = np.load(f"data/lucy/turnaround_8v/{name}_normal.npy")[sl] @ R.T
gt_d = np.load(f"data/lucy/turnaround_8v/{name}_depth.npy")[sl]
ev_m = inner & (np.linalg.norm(gt_n, axis=-1) > 0.5)


def err(n):
    a_ = np.degrees(np.arccos(np.clip((n[ev_m] * gt_n[ev_m]).sum(-1), -1, 1)))
    return round(float(np.median(a_)), 2), round(float((a_ < 11.25).mean()), 3), round(float((a_ < 30).mean()), 3)


print("법선 오차 (중앙값°, <11.25° 비율, <30° 비율)")
print("  정답 깊이        :", err(normals_np(np.where(gt_d > 0, gt_d, z_sk))))
print("  뼈대             :", err(n_sk))
print("  A 추정 법선       :", err(n_est))
print("  B 적분 깊이 법선  :", err(normals_np(z_new)))

os.makedirs(args.out, exist_ok=True)
full = zb.copy()
full[sl] = z_new
np.savez(os.path.join(args.out, f"{name}_depth.npz"), depth=full[None].astype(np.float32))

RL = np.array([0.8, -0.1, -0.6])
RL /= np.linalg.norm(RL)


def vis(n):
    s = 0.15 + 0.85 * np.clip((n * RL).sum(-1), 0, 1)
    s[~m] = 0.4
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)


crop = (slice(60, 280), slice(Wc // 2 - 150, Wc // 2 + 150))
tiles = [vis(normals_np(np.where(gt_d > 0, gt_d, z_sk))), vis(n_sk), vis(n_est), vis(normals_np(z_new))]
tiles = [t[crop].copy() for t in tiles]
for tl, txt in zip(tiles, ["GT (side light)", "skeleton", "A normals", "B integrated"]):
    cv2.putText(tl, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
cv2.imwrite(os.path.join(args.out, f"{name}_face.png"),
            cv2.resize(np.hstack(tiles), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
log("done")
