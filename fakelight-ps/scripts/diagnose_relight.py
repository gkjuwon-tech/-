"""리라이팅 병목 진단: 정답 노멀로 각 가짜 조명 이미지의 '실제' 조명 방향과 물리적 일관성을 역산한다.

램버트 모델 I = max(0, n·g), g = 알베도 × 조명방향. 정답 노멀 n을 알면 이미지마다 g를 최소제곱으로 풀 수 있다
(빛을 받는 픽셀만 써서 몇 번 반복). 결과:
  - 방위각/고도: IC-Light가 실제로 비춘 빛의 방향 (요청한 그라데이션 방향과 비교)
  - R²: 이 이미지 음영이 '진짜 방향광 하나'로 얼마나 설명되나. 낮으면 IC-Light가 음영을 지어낸 것
  - 커버리지: 조명 방향들이 반구를 얼마나 고르게 덮나 (정답 노멀별로 빛을 받는 이미지 수)

사용법:
    python scripts/diagnose_relight.py results/relight_ps_v2/jugg_s1.data --gt renders/bunny_front_aligned_768
    python scripts/diagnose_relight.py renders/bunny_real_lights.data --gt renders/bunny_real_lights.data
"""
import argparse
import glob
import json
import os

import numpy as np

from ps_eval import load_gray, load_mask


def fit_light(I, N, iters=5):
    """I: (P,), N: (P,3). 빛 받는 픽셀로 g를 풀고 R²를 돌려준다."""
    lit = I > 0.02
    g = np.zeros(3)
    for _ in range(iters):
        g, *_ = np.linalg.lstsq(N[lit], I[lit], rcond=None)
        lit = (N @ g) > 0.02
    pred = np.clip(N @ g, 0, None)
    ss_res = ((I - pred) ** 2).sum()
    ss_tot = ((I - I.mean()) ** 2).sum()
    return g, 1 - ss_res / ss_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--gt", required=True, help="normal_gt.npy와 mask.png가 있는 폴더")
    args = ap.parse_args()

    N_img = np.load(os.path.join(args.gt, "normal_gt.npy"))
    mask = load_mask(os.path.join(args.gt, "mask.png"), N_img.shape[:2])
    N = N_img[mask]
    paths = sorted(glob.glob(os.path.join(args.folder, "L_*.png")))
    meta = os.path.join(args.folder, "relight.json")
    req = json.load(open(meta))["angles"] if os.path.exists(meta) else [None] * len(paths)

    dirs, rows = [], []
    for p, a in zip(paths, req):
        I = load_gray(p)
        if I.shape != N_img.shape[:2]:
            from PIL import Image
            I = np.asarray(Image.fromarray(I.astype(np.float32), "F").resize(N_img.shape[1::-1], Image.BILINEAR))
        g, r2 = fit_light(I[mask], N)
        l = g / (np.linalg.norm(g) + 1e-12)
        dirs.append(l)
        az = np.degrees(np.arctan2(l[1], l[0])) % 360
        el = np.degrees(np.arcsin(np.clip(l[2], -1, 1)))  # 카메라 축 기준 고도 (90 = 정면광)
        rows.append((os.path.basename(p), a, az, el, np.linalg.norm(g), r2))

    print(f"{'image':10s} {'요청방향':>8s} {'실제방위':>8s} {'정면쪽고도':>10s} {'세기':>6s} {'R²':>6s}")
    for name, a, az, el, s, r2 in rows:
        print(f"{name:10s} {('-' if a is None else f'{a:.0f}'):>8s} {az:8.0f} {el:10.0f} {s:6.2f} {r2:6.3f}")

    D = np.array(dirs)
    # 커버리지: 각 정답 노멀이 몇 장에서 '충분히' 빛을 받나 (n·l > 0.3)
    lit_count = ((N @ D.T) > 0.3).sum(1)
    down = N[:, 1] < -0.3
    print(f"\n평균 R² {np.mean([r[5] for r in rows]):.3f} | 실제 방향 고도 평균 {np.mean([r[3] for r in rows]):.0f}° "
          f"| 방위각 범위 {min(r[2] for r in rows):.0f}~{max(r[2] for r in rows):.0f}°")
    print(f"노멀당 충분히 빛받은 이미지 수: 전체 평균 {lit_count.mean():.1f}, "
          f"아래 향한 면 {lit_count[down].mean():.1f}, 3장 미만인 픽셀 {(lit_count < 3).mean():.1%}")
    # 조명 방향 행렬의 조건수: 클수록 방향들이 한쪽에 몰려 풀기 불안정
    print(f"조명 방향 행렬 특이값 {np.round(np.linalg.svd(D, compute_uv=False), 2)}")


if __name__ == "__main__":
    main()
