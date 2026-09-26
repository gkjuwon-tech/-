"""궤도 영상(Veo, Wan, SEVA 등) → 멀티뷰 프레임 세트.

1) 영상에서 프레임을 고르게 N장 뽑는다 (첫 프레임 = 입력 사진 시점)
2) 프레임마다 누끼 (rembg BiRefNet, CPU에서는 한 프로세스에 여러 장 돌리면 메모리가 불어나서 장마다 새 프로세스)
3) out/frames/*.png, out/rgba/*.png, out/mask/*.png, out/frames.json (원래 프레임 번호, 시간)

카메라는 이 단계에서 모른다 (Veo는 카메라 값을 주지 않음). 다음 단계에서 VGGT로 프레임끼리 카메라를 추정한다.

사용법:
    python scripts/video_to_views.py --video orbit.mp4 --out build/views/knight --n 24
"""
import argparse
import json
import os
import subprocess
import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24, help="뽑을 프레임 수")
    ap.add_argument("--skip-last", type=int, default=0, help="끝에서 버릴 프레임 수 (마지막이 첫 프레임과 겹치는 한 바퀴 영상)")
    ap.add_argument("--no-matte", action="store_true")
    args = ap.parse_args()

    frames = iio.imread(args.video, plugin="pyav")                  # (T, H, W, 3)
    T = len(frames) - args.skip_last
    idx = np.linspace(0, T - 1, args.n).round().astype(int)
    meta = iio.immeta(args.video, plugin="pyav")
    fps = meta.get("fps", 24)
    for d in ["frames", "rgba", "mask"]:
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    names = []
    for j, i in enumerate(idx):
        name = f"f{j:03d}.png"
        Image.fromarray(frames[i]).save(os.path.join(args.out, "frames", name))
        names.append({"name": name, "frame": int(i), "time": float(i / fps)})
    json.dump({"video": os.path.basename(args.video), "total_frames": int(len(frames)), "fps": fps, "views": names},
              open(os.path.join(args.out, "frames.json"), "w"), indent=1)
    print(f"{len(names)} frames from {len(frames)} ({frames.shape[2]}x{frames.shape[1]})", flush=True)
    if args.no_matte:
        return
    for v in names:
        if os.path.exists(os.path.join(args.out, "rgba", v["name"])):
            continue
        subprocess.run([sys.executable, os.path.join(HERE, "matte_views.py"), "--src", os.path.join(args.out, "frames"),
                        "--out", args.out, "--one", v["name"]], check=True)


if __name__ == "__main__":
    main()
