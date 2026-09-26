"""음영 기반 디테일 보정: 입력 사진 한 장의 미세한 명암으로 노멀의 고주파를 되살린다.

생성된 조명 이미지(IC-Light 등)는 주름을 새로 지어내서 디테일이 뭉개진다. 반면 입력 사진의 명암은
실제 주름이 원래 조명을 받아 생긴 것이다. 그래서 큰 모양이 좋은 노멀(합체 결과)을 출발점으로
  1) 입력 사진의 조명을 추정한다: I ≈ c0 + c·n  (1차 구면조화 = 환경광 + 방향광, 알베도 상수 가정)
  2) 노멀을 푼다: min Σ_p (c0 + c·n_p - I_p)² + λ|n_p - n0_p|² + μ Σ_이웃 |n_p - n_q|²  (반복 후 정규화)
     첫 항이 입력 명암의 디테일을, 둘째 항이 큰 모양 유지를, 셋째 항이 노이즈 억제를 맡는다.
단일 이미지라 조명 방향과 수직인 성분의 디테일은 복원되지 않는다 (고전적인 한계).

사용법:
    python scripts/shading_refine.py --image results/audition/Juggernaut-XL-v9/view_az000.png \
        --init results/fusion/fused_normal.png --mask results/relight_ps_v2/jugg_s1.data/mask.png \
        --gt renders/bunny_front_aligned_768 --out results/fusion/refined_normal.png
"""
import argparse
import os

import numpy as np
from PIL import Image

from detail_metrics import detail_scores
from ps_eval import load_mask, load_normal_png, mean_angular_error


def refine(I, n0, m, lam=1.0, mu=0.5, iters=200, step=0.4):
    """I: HxW 밝기, n0: HxWx3 초기 노멀, m: 마스크. 경사하강으로 푼다."""
    N = n0[m]
    A = np.c_[np.ones(len(N)), N]
    coef, *_ = np.linalg.lstsq(A, I[m], rcond=None)   # 조명 추정 (c0, c)
    c0, c = coef[0], coef[1:]
    n = n0.copy()
    for _ in range(iters):
        r = (c0 + n @ c - I) * m                        # 음영 잔차
        grad = r[..., None] * c[None, None, :] + lam * (n - n0)
        lap = (np.roll(n, 1, 0) + np.roll(n, -1, 0) + np.roll(n, 1, 1) + np.roll(n, -1, 1) - 4 * n)
        grad -= mu * lap
        n = n - step * grad / (c @ c + lam + 4 * mu)
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
        n[~m] = n0[~m]
    return n, (c0, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0])
    ap.add_argument("--mu", type=float, nargs="+", default=[0.5])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n0 = load_normal_png(args.init, None)
    shape = n0.shape[:2]
    m = load_mask(args.mask, shape)
    I = np.asarray(Image.open(args.image).convert("L").resize(shape[::-1], Image.BICUBIC), dtype=np.float64) / 255.0
    if args.gt:
        gt = np.load(os.path.join(args.gt, "normal_gt.npy"))
        gm = load_mask(os.path.join(args.gt, "mask.png"), gt.shape[:2])
        c, e = detail_scores(n0, gt, gm)
        print(f"초기: MAE {mean_angular_error(n0, gt, gm)[0]:.2f}  디테일상관 {c:.3f} 세기 {e:.2f}")
    best = None
    for lam in args.lam:
        for mu in args.mu:
            n, (c0, c) = refine(I, n0, m & (np.abs(n0).sum(2) > 0), lam, mu)
            if args.gt:
                mae = mean_angular_error(n, gt, gm)[0]
                dc, de = detail_scores(n, gt, gm)
                print(f"λ={lam:<5} μ={mu:<5} MAE {mae:.2f}  디테일상관 {dc:.3f} 세기 {de:.2f}  (조명 {np.round(c / np.linalg.norm(c), 2)})")
            best = n
    if args.out:
        valid = np.linalg.norm(best, axis=2, keepdims=True) > 0.5   # 보정 마스크 밖은 초기 노멀 그대로 유지
        Image.fromarray(((best * 0.5 + 0.5) * 255 * valid).astype(np.uint8)).save(args.out)


if __name__ == "__main__":
    main()
