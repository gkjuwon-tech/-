"""Stable Virtual Camera(SEVA) 입력 장면 만들기: 사진 1장 + 물체 중심을 도는 목표 카메라들.

SEVA의 ReconfusionParser 형식으로 쓴다: images/ (입력 1장), transforms.json (OpenGL c2w, 초점거리),
train_test_split_1.json (입력 0번, 나머지는 목표). 목표 이미지는 file_path=None → SEVA가 검은 자리표시자로 채운다.
같은 카메라를 우리 형식(views.json, OpenCV c2w + 가로 FOV)으로도 저장해서 LightSwitch/stage2에 그대로 넘긴다.

카메라: 물체 중심 = 원점, 입력 카메라 = 방위각 0°, 고도 0°, 거리 D, FOV 30°.
  고도 0° 링 30° 간격 11개 + 고도 +30° 링 60° 간격 6개 + 고도 -20° 3개 = 목표 20개 (입력 포함 21 = SEVA 기본 T 한 번에)

사용법:
    python scripts/seva_scene.py --image renders/bunny_front_rgba.png --out build/seva_scene/bunny
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

D, FOV = 3.2, 30.0
RINGS = [(0, list(range(30, 360, 30))), (30, list(range(0, 360, 60))), (-20, [45, 165, 285])]


def c2w_opencv(az_deg, el_deg, dist=D):
    """z 위 세계. 방위각 0° 카메라는 -y 쪽에서 +y를 바라본다 (MV-Adapter 규약과 같은 방향)."""
    az, el = np.radians(az_deg - 90), np.radians(el_deg)
    c = dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    f = -c / np.linalg.norm(c)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = r, d, f, c
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGBA 또는 RGB 사진")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=576)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    im = Image.open(args.image)
    if im.mode == "RGBA":
        a = np.asarray(im).astype(np.float32) / 255
        rgb = a[..., :3] * a[..., 3:] + (1 - a[..., 3:])                       # 흰 배경
        im = Image.fromarray((rgb * 255).round().astype(np.uint8))
    im.convert("RGB").resize((args.res, args.res), Image.LANCZOS).save(os.path.join(args.out, "images", "input.png"))

    cams = [(0, 0)] + [(az, el) for el, azs in RINGS for az in azs]
    fl = args.res / 2 / np.tan(np.radians(FOV) / 2)
    frames, views = [], []
    for i, (az, el) in enumerate(cams):
        m = c2w_opencv(az, el)
        gl = m.copy()
        gl[:3, 1:3] *= -1                                                        # OpenCV → OpenGL (파서가 되돌림)
        frames.append({"file_path": "images/input.png" if i == 0 else None, "transform_matrix": gl.tolist()})
        views.append({"name": f"v{i:02d}_az{az:03d}_el{el:+03d}.png", "c2w_opencv": m.tolist(), "fovx_deg": FOV,
                      "azimuth": az, "elevation": el})
    meta = {"fl_x": fl, "fl_y": fl, "cx": args.res / 2, "cy": args.res / 2, "w": args.res, "h": args.res,
            "frames": frames}
    json.dump(meta, open(os.path.join(args.out, "transforms.json"), "w"), indent=1)
    json.dump({"train_ids": [0], "test_ids": list(range(1, len(cams)))},
              open(os.path.join(args.out, "train_test_split_1.json"), "w"))
    json.dump({"views": views}, open(os.path.join(args.out, "views.json"), "w"), indent=1)
    print(f"{len(cams)} cameras -> {args.out}")


if __name__ == "__main__":
    main()
