"""무늬 오인 시험 세트: 기하는 그대로 두고 버니 표면에 무늬(알베도)만 입힌다.

같은 시점이라 무늬는 화면 좌표로 정의해도 모든 조명 이미지에서 같은 자리에 붙는다.
  - input.png      : 무늬 × 스튜디오 조명 (render_front.shade와 같은 조명) → 디테일 보정의 입력
  - relights/L_XX  : 무늬 × 방향광 K개 (+ 환경광, 생성 모델 흉내로 이미지마다 세기/감마를 조금씩 흔듦)
  - albedo_gt.npy  : 정답 무늬 (진단용)
무늬 없는 버전(--plain)도 같은 방식으로 만들어서, 개선판이 무늬 없는 경우를 망가뜨리지 않는지 확인한다.

사용법:
    python scripts/make_textured_test.py --out renders/texture_test
    python scripts/make_textured_test.py --out renders/plain_test --plain
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from render_front import shade
from render_lights import light_dirs


def texture(shape, seed=0):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W] / H
    rng = np.random.default_rng(seed)
    stripes = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * (xx * 0.8 + yy * 0.5) * 9))     # 대각선 줄무늬
    spots = np.zeros(shape)
    for _ in range(60):                                                               # 점박이
        cy, cx, r = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.01, 0.035)
        spots[(yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2] = 1
    rho = 0.85 - 0.35 * stripes * (yy < 0.55) - 0.45 * spots * (yy >= 0.4)
    return gaussian_filter(np.clip(rho, 0.2, 1), 0.8)                                   # 경계 안티에일리어싱


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", default="renders/bunny_front_normal_gt.npy")
    ap.add_argument("--mask", default="renders/bunny_front_mask.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--num", type=int, default=16)
    args = ap.parse_args()

    N = np.load(args.normal)
    m = np.asarray(Image.open(args.mask).convert("L")) > 127
    rho = np.full(m.shape, 0.8) if args.plain else texture(m.shape)
    rho = rho * m
    os.makedirs(os.path.join(args.out, "relights"), exist_ok=True)

    lum = shade(N, m, albedo=(1.0, 1.0, 1.0))[..., 0]          # 무늬 없는 스튜디오 조명 밝기
    img = np.where(m, rho * lum, 1.0)
    Image.fromarray((np.repeat(img[..., None], 3, 2) * 255).clip(0, 255).astype(np.uint8)).save(
        os.path.join(args.out, "input.png"))

    rng = np.random.default_rng(1)
    L = light_dirs(args.num, 45.0)
    for k, l in enumerate(L):
        s = 0.1 + 0.8 * np.clip(N @ l, 0, 1)                    # 환경광 + 방향광
        s = s * rng.uniform(0.85, 1.15)                         # 이미지마다 세기 흔들림
        x = np.clip(rho * s, 0, 1) ** rng.uniform(0.9, 1.1)     # 약한 톤 변화
        x[~m] = 0
        Image.fromarray((np.repeat(x[..., None], 3, 2) * 255).astype(np.uint8)).save(
            os.path.join(args.out, "relights", f"L_{k:02d}.png"))
    Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(args.out, "mask.png"))
    Image.fromarray((m * 255).astype(np.uint8)).save(os.path.join(args.out, "relights", "mask.png"))
    np.save(os.path.join(args.out, "albedo_gt.npy"), rho.astype(np.float32))
    np.save(os.path.join(args.out, "normal_gt.npy"), N.astype(np.float32))
    json.dump({"plain": args.plain, "lights": L.tolist()}, open(os.path.join(args.out, "info.json"), "w"))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
