"""실험: DA3 깊이 + 음영(Shape from Shading) 보정, 정면 1장.

루시 렌더의 조명을 알고 있다는 가정의 상한선 실험. 이미지 명암이 렌더 모델과 맞도록
DA3 깊이에 잔차 δ를 더해 최적화한다. 결과 법선을 정답과 비교한다.
사용: python scripts/sfs_front.py [--iters 300]
"""
import argparse
import json
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--anchor", type=float, default=0.05, help="δ 크기 벌점 (DA3에 붙잡는 힘)")
ap.add_argument("--base", default=None, help="1024px 뼈대 깊이 npz (예: work/fused_depth.npz). 없으면 DA3")
ap.add_argument("--smooth", type=float, default=0.0, help="뼈대 가우시안 스무딩 σ(px), 줄무늬 제거용")
ap.add_argument("--finest", type=int, default=1, help="다해상도 잔차의 가장 촘촘한 격자 간격(px)")
ap.add_argument("--curv", type=float, default=0.0, help="잔차 라플라시안 벌점 (긁힘 줄무늬 억제)")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--out", default="work/sfs_view_000.npz")
args = ap.parse_args()
torch.set_num_threads(4)
t0 = time.time()

cams = json.load(open("inputs/lucy_8v/cameras.json"))
v = cams["views"][0]
K, R = np.array(v["K"]), np.array(v["R"])
rgba = cv2.imread("inputs/lucy_8v/view_000.png", cv2.IMREAD_UNCHANGED)
alpha = rgba[..., 3] / 255.0
mask = alpha >= 0.5
# 렌더러의 색 모델 역산: rgb = clip(col * [0.95,0.92,0.86] * 1.05), 초록 채널 사용
I = rgba[..., 1] / 255.0 / (0.92 * 1.05)

from q1_front import upsample_depth  # noqa: E402
if args.base:
    zb = np.load(args.base)["depth"][0]
    z0 = zb.astype(np.float64)
    hole = mask & (z0 <= 0)                        # 뼈대가 못 덮은 마스크 픽셀은 가장 가까운 값으로
    if hole.any():
        _, idx = cv2.distanceTransformWithLabels((z0 <= 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
        yy, xx = np.nonzero(z0 > 0)
        lut = np.zeros((idx.max() + 1, 2), int)
        lut[idx[z0 > 0]] = np.stack([yy, xx], 1)
        src = lut[idx]
        z0 = z0[src[..., 0], src[..., 1]]
else:
    z0 = upsample_depth(np.load("work/da3_mv504_pose.npz")["depth"][0], mask)
if args.smooth > 0:
    z0 = cv2.GaussianBlur(z0.astype(np.float32), (0, 0), args.smooth).astype(np.float64)

# 물체 bbox로 자르기
ys, xs = np.nonzero(mask)
y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
sl = (slice(y0, y1), slice(x0, x1))
Hc, Wc = y1 - y0, x1 - x0
u = torch.arange(x0, x1, dtype=torch.float64) + 0.5
w = torch.arange(y0, y1, dtype=torch.float64) + 0.5
vv, uu = torch.meshgrid(w, u, indexing="ij")
rx = (uu - K[0, 2]) / K[0, 0]
ry = (vv - K[1, 2]) / K[1, 1]

m = torch.tensor(mask[sl])
# 안쪽 픽셀만 (경계·안티앨리어싱 제외), 명암이 포화된 곳 제외
inner = torch.tensor(cv2.erode(mask[sl].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
It = torch.tensor(I[sl])
valid = inner & (It < 0.98)
z0t = torch.tensor(z0[sl], dtype=torch.float64)

# 조명 (카메라 좌표; 렌더러 shade()와 동일)
def L(d):
    d = np.array([d[0], -d[1], -d[2]], float)
    return torch.tensor(d / np.linalg.norm(d))
lights = [(L([-0.5, 0.7, 0.5]), 0.75), (L([0.6, 0.2, 0.6]), 0.30), (L([0.0, 0.5, -0.9]), 0.25)]
Hs = torch.tensor(np.array([-0.5, -0.7, -1.0]) / np.linalg.norm([-0.5, -0.7, -1.0]))


def normals(z):
    P = torch.stack([rx * z, ry * z, z], -1)
    dx = P[:, 1:, :] - P[:, :-1, :]
    dy = P[1:, :, :] - P[:-1, :, :]
    dx = F.pad(dx.permute(2, 0, 1), (0, 1, 0, 0), mode="replicate").permute(1, 2, 0)
    dy = F.pad(dy.permute(2, 0, 1), (0, 0, 0, 1), mode="replicate").permute(1, 2, 0)
    n = torch.cross(dy, dx, dim=-1)          # 카메라 쪽(-z)을 향하도록
    return n / (n.norm(dim=-1, keepdim=True) + 1e-12)


def shade(n):
    c = torch.full(n.shape[:2], 0.18, dtype=n.dtype)
    for d, k in lights:
        c = c + k * torch.clamp((n * d).sum(-1), min=0)
    c = c + 0.12 * torch.clamp((n * Hs).sum(-1), min=0) ** 24
    return c


# 깊이 불연속(큰 점프)에서는 음영 항을 끄기
jump = torch.zeros_like(m)
gz = torch.zeros_like(z0t)
gz[:, :-1] = (z0t[:, 1:] - z0t[:, :-1]).abs()
gz[:-1, :] = torch.maximum(gz[:-1, :], (z0t[1:, :] - z0t[:-1, :]).abs())
jump = gz > 0.02 * z0t
valid = valid & ~jump

# 다해상도 잔차: δ = Σ 업샘플(거친 격자들). 큰 기울기를 빠르게 고칠 수 있게
levels = [torch.zeros((max(Hc // f, 2), max(Wc // f, 2)), dtype=torch.float64, requires_grad=True)
          for f in (32, 16, 8, 4, 2, 1) if f >= args.finest]
opt = torch.optim.Adam(levels, lr=args.lr)


def make_delta():
    return sum(F.interpolate(g[None, None], size=(Hc, Wc), mode="bilinear", align_corners=False)[0, 0]
               for g in levels)
scale = z0t[m].mean()
for it in range(args.iters):
    opt.zero_grad()
    delta = make_delta()
    z = z0t + delta
    s = shade(normals(z))
    loss_sh = ((s - It)[valid] ** 2).mean()
    loss_an = args.anchor * ((delta[m] / scale) ** 2).mean() * 1e4
    lap = (delta[1:-1, 1:-1] * 4 - delta[:-2, 1:-1] - delta[2:, 1:-1] - delta[1:-1, :-2] - delta[1:-1, 2:])
    loss_cv = args.curv * ((lap[m[1:-1, 1:-1]] / scale) ** 2).mean() * 1e8
    loss = loss_sh + loss_an + loss_cv
    loss.backward()
    opt.step()
    if it % 50 == 0 or it == args.iters - 1:
        print(f"it {it:4d} shading {loss_sh.item():.5f} anchor {loss_an.item():.5f}  t={time.time()-t0:.0f}s", flush=True)
    if time.time() - t0 > 55:
        print("시간 제한 도달, 중단", flush=True)
        break

# 평가: 정답 법선과 비교
gt_n = np.load("data/lucy/turnaround_8v/view_000_normal.npy")[sl] @ R.T
ev = inner.numpy() & (np.linalg.norm(gt_n, axis=-1) > 0.5)


def err(n):
    n = n.detach().numpy()
    a = np.degrees(np.arccos(np.clip((n[ev] * gt_n[ev]).sum(-1), -1, 1)))
    return round(float(np.median(a)), 2), round(float((a < 11.25).mean()), 3), round(float((a < 30).mean()), 3)


gt_d = np.load("data/lucy/turnaround_8v/view_000_depth.npy")[sl]
delta = make_delta().detach()
n_da3, n_sfs = normals(z0t), normals(z0t + delta)
n_da3_raw = normals(torch.tensor(upsample_depth(np.load("work/da3_mv504_pose.npz")["depth"][0], mask)[sl], dtype=torch.float64))
n_gtd = normals(torch.tensor(np.where(gt_d > 0, gt_d, z0[sl]), dtype=torch.float64))
print("법선 오차 (중앙값°, <11.25° 비율, <30° 비율)")
print("  정답 깊이에서 계산 :", err(n_gtd))
print("  DA3 원본           :", err(n_da3_raw))
print("  뼈대(보정 전)       :", err(n_da3))
print("  뼈대 + 음영 보정    :", err(n_sfs))
full = z0.copy()
full[sl] = (z0t + delta).detach().numpy()
np.savez(args.out, depth=full[None].astype(np.float32))


RELIGHT = torch.tensor(np.array([0.8, -0.1, -0.6]) / np.linalg.norm([0.8, -0.1, -0.6]))


def vis(n):
    s = (0.15 + 0.85 * torch.clamp((n * RELIGHT).sum(-1), min=0)).detach().numpy()
    s[~m.numpy()] = 0.4
    return (np.clip(s, 0, 1) * 255).astype(np.uint8)


fy0 = 60
crop = (slice(fy0, fy0 + 220), slice(Wc // 2 - 150, Wc // 2 + 150))
tiles = [vis(n_gtd)[crop], vis(n_da3_raw)[crop], vis(n_da3)[crop], vis(n_sfs)[crop]]
for tl, txt in zip(tiles, ["GT (side light)", "DA3 raw", "skeleton", "skeleton+SfS"]):
    cv2.putText(tl, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
cv2.imwrite(args.out.replace(".npz", "_face.png"), cv2.resize(np.hstack(tiles), None, fx=2, fy=2,
                                                              interpolation=cv2.INTER_NEAREST))
print(f"총 {time.time()-t0:.0f}s")
