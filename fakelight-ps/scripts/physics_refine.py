"""물리 법칙 검문소: 가짜 조명 이미지들이 '같은 노멀 + 이미지별 방향광 + 환경광'으로 설명되도록 노멀을 다듬는다.

SDM-UniPS 노멀에서 출발해 교대로 푼다 (정답은 채점에만 쓴다):
  1) 조명: 이미지 k마다 I_k ≈ ρ·max(0, n·l_k)·s_k + a_k 를 현재 노멀로 최소제곱 (빛 받는 픽셀만)
  2) 노멀: 픽셀마다 b = ρn 을 가중 최소제곱으로. 그림자(n·l<0) 관측은 빼고, 잔차가 큰 관측
     (IC-Light가 이미지마다 다르게 지어낸 음영)은 Huber 가중치로 눌러준다. 초기 노멀 쪽으로 약하게 당긴다.
  3) 반복

입력 이미지는 화면용 sRGB라서 선형 밝기로 바꿔서 쓴다.

사용법:
    python scripts/physics_refine.py --images "results/relight_ps_v2/jugg_s1.data/L_*.png" \
        --init results/relight_ps_v2/sdm_results/jugg_s1_x2.data/normal.png \
        --mask results/relight_ps_v2/jugg_s1.data/mask.png --gt renders/bunny_front_aligned_768
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image

from ps_eval import load_mask, load_normal_png, mean_angular_error


def srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def load_images(pattern, shape, linear=True):
    out = []
    for p in sorted(glob.glob(pattern)):
        im = Image.open(p).convert("RGB")
        if im.size != (shape[1], shape[0]):
            im = im.resize((shape[1], shape[0]), Image.BICUBIC)
        x = np.asarray(im, dtype=np.float64).mean(2) / 255.0
        out.append(srgb2lin(x) if linear else x)
    return np.stack(out)


def fit_lights(I, B, XY=None):
    """I: (K,P), B: (P,3) = ρn. 반환 G: (K,3) (세기 포함 방향), a: (K,P) 환경광, s: (K,P) 빛 세기 배율.
    XY가 있으면 빛 세기와 환경광이 화면 위치에 따라 선형으로 변한다고 본다 (IC-Light는 배경 그라데이션 쪽이
    더 밝게 비춰져서 먼 곳의 균일한 빛이 아니라 가까운 조명처럼 행동한다)."""
    K, P = I.shape
    G, a, s = np.zeros((K, 3)), np.zeros((K, P)), np.ones((K, P))
    for k in range(K):
        lit = np.ones(P, bool)
        for _ in range(4):
            if XY is None:
                A = np.c_[B, np.ones(P)]
            else:
                X, Y = XY
                A = np.c_[B, B * X[:, None], B * Y[:, None], np.ones(P), X, Y]
            Al = A.copy()
            Al[~lit, : (3 if XY is None else 9)] = 0
            x, *_ = np.linalg.lstsq(Al, I[k], rcond=None)
            G[k] = x[:3]
            lit = (B @ G[k]) > 0
        if XY is None:
            a[k] = x[3]
        else:
            X, Y = XY
            gx, gy = x[3:6], x[6:9]
            # 세기 배율 s(x,y) = 1 + αX + βY 로 근사: gx ≈ αG, gy ≈ βG
            gn = G[k] @ G[k] + 1e-12
            alpha, beta = gx @ G[k] / gn, gy @ G[k] / gn
            s[k] = np.clip(1 + alpha * X + beta * Y, 0.2, 5)
            a[k] = x[9] + x[10] * X + x[11] * Y
    return G, a, s


def solve_normals(I, G, a, s, B_prev, B_init, lam, huber):
    """픽셀별 가중 최소제곱: Σ_k w_kp ((I_kp - a_kp)/s_kp - b_p·g_k)² + lam·|b_p - B_init_p|²"""
    R = (I - a) / s                                     # (K,P) 환경광 빼고 세기 배율 나눔 = 균일광으로 보정
    shade = G @ B_prev.T                                # (K,P) 현재 예측 음영
    lit = shade > 0
    res = np.abs(R - shade)
    scale = np.median(res[lit]) + 1e-8
    w = np.where(res <= huber * scale, 1.0, huber * scale / (res + 1e-12)) * lit
    AtA = np.einsum("kp,ki,kj->pij", w, G, G) + lam * np.eye(3)
    Atb = np.einsum("kp,ki,kp->pi", w, G, R) + lam * B_init
    return np.linalg.solve(AtA, Atb[..., None])[..., 0], w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob 패턴")
    ap.add_argument("--init", required=True, help="초기 노멀 PNG (SDM-UniPS 결과)")
    ap.add_argument("--mask", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.05, help="초기 노멀 쪽으로 당기는 세기 (알베도 스케일 기준)")
    ap.add_argument("--huber", type=float, default=2.0)
    ap.add_argument("--no-linear", action="store_true")
    ap.add_argument("--ramp", action="store_true", help="위치에 따라 변하는 빛 세기/환경광 모델")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gt = np.load(os.path.join(args.gt, "normal_gt.npy"))
    gmask = load_mask(os.path.join(args.gt, "mask.png"), gt.shape[:2])
    shape = gt.shape[:2]
    m = load_mask(args.mask, shape)
    n0 = load_normal_png(args.init, shape)
    I = load_images(args.images, shape, linear=not args.no_linear)[:, m]   # (K,P)
    N0 = n0[m]
    yy, xx = np.nonzero(m)
    XY = ((xx / shape[1] - 0.5) * 2, -(yy / shape[0] - 0.5) * 2) if args.ramp else None

    # 알베도 초기값: 조명 한 번 맞춘 뒤 픽셀별 스케일
    rho = np.full(N0.shape[0], np.median(I))
    B_init = N0 * rho[:, None]
    B = B_init.copy()
    score = lambda Bm: mean_angular_error(_to_img(Bm, m, shape), gt, gmask)[0]
    print(f"iter 0 (SDM-UniPS): {score(B):.2f}°")
    for it in range(1, args.iters + 1):
        G, a, sc = fit_lights(I, B, XY)
        B, w = solve_normals(I, G, a, sc, B, B_init * (np.linalg.norm(B, axis=1, keepdims=True) /
                                                   (np.linalg.norm(B_init, axis=1, keepdims=True) + 1e-12)),
                             args.lam * np.median(np.linalg.norm(G, axis=1)) ** 2, args.huber)
        print(f"iter {it}: {score(B):.2f}°  (가중치 평균 {w[w > 0].mean():.2f})")
    if args.out:
        n = _to_img(B, m, shape)
        Image.fromarray(((n * 0.5 + 0.5) * 255 * m[..., None]).astype(np.uint8)).save(args.out)


def _to_img(Bm, m, shape):
    n = np.zeros(shape + (3,))
    n[m] = Bm / (np.linalg.norm(Bm, axis=1, keepdims=True) + 1e-12)
    return n


if __name__ == "__main__":
    main()
