"""멀티뷰 프레임 누끼 따기 (rembg, BiRefNet 일반 모델). RGBA PNG와 마스크를 저장한다.

사용법:
    python scripts/matte_views.py --src build/dl/seva_dragon/out/dragon/result/samples-rgb --out results/seva_basic_dragon/matte
"""
import argparse
import glob
import os

import numpy as np
from PIL import Image
from rembg import new_session, remove


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="birefnet-general")
    ap.add_argument("--one", default=None, help="이 파일 하나만 (BiRefNet은 CPU에서 한 프로세스로 여러 장 돌리면 메모리가 계속 불어남)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "rgba"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "mask"), exist_ok=True)
    sess = new_session(args.model)
    files = [os.path.join(args.src, args.one)] if args.one else sorted(glob.glob(os.path.join(args.src, "*.png")))
    for p in files:
        im = Image.open(p).convert("RGB")
        rgba = remove(im, session=sess)
        a = np.asarray(rgba)
        name = os.path.basename(p)
        rgba.save(os.path.join(args.out, "rgba", name))
        Image.fromarray(((a[..., 3] > 127) * 255).astype(np.uint8)).save(os.path.join(args.out, "mask", name))
        print(name, f"fg {(a[..., 3] > 127).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
