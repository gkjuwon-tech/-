"""진짜 조명 시퀀스 렌더러: 정면 버니를 방향광 K개로 비춘 이미지를 만든다.

광도 스테레오 풀이기의 기준 실험용이다. 진짜 조명으로 정답에 가깝게 복원되면 풀이기는 정상이고,
나중에 가짜(생성) 조명에서 오차가 커지면 그건 생성기 탓이라고 분리해서 볼 수 있다.

램버트 반사 + 붙은 그림자(max(0, n·l))만 넣는다. 드리운 그림자는 없다.
출력 폴더는 SDM-UniPS 입력 형식(L_XX.png + mask.png)이고, 조명 방향은 lights.json에 저장한다.

사용법:
    python scripts/render_lights.py --out renders/bunny_real_lights.data --num 8 --polar 45
"""
import argparse
import json
import os

import numpy as np
import trimesh
from PIL import Image

from render_front import look_rotation, rasterize


def light_dirs(num, polar_deg):
    """카메라 축에서 polar_deg만큼 기울어진 원뿔 위에 고르게 놓인 방향광 (x 오른쪽, y 위, z 카메라 쪽)."""
    a = np.deg2rad(polar_deg)
    phi = np.arange(num) * 2 * np.pi / num
    return np.stack([np.sin(a) * np.cos(phi), np.sin(a) * np.sin(phi), np.full(num, np.cos(a))], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="assets/stanford-bunny.obj")
    ap.add_argument("--out", default="renders/bunny_real_lights.data")
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--yaw", type=float, default=80.0)
    ap.add_argument("--fill", type=float, default=0.85)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--polar", type=float, default=45.0)
    ap.add_argument("--albedo", type=float, default=0.8)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=True)
    v = mesh.vertices - mesh.bounding_box.centroid
    v = v @ look_rotation(args.yaw).T
    v = v / (np.abs(v[:, :2]).max() / args.fill)
    m = trimesh.Trimesh(v, mesh.faces, process=False)
    normals, mask = rasterize(v, m.faces, m.vertex_normals, args.res)

    os.makedirs(args.out, exist_ok=True)
    L = light_dirs(args.num, args.polar)
    for i, l in enumerate(L):
        img = args.albedo * np.clip(normals @ l, 0, 1)
        img[~mask] = 0
        Image.fromarray((np.repeat(img[..., None], 3, 2) * 255).round().astype(np.uint8)).save(
            os.path.join(args.out, f"L_{i:02d}.png"))
    Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out, "mask.png"))
    np.save(os.path.join(args.out, "normal_gt.npy"), normals.astype(np.float32))
    json.dump({"lights": L.tolist(), "polar_deg": args.polar, "yaw": args.yaw},
              open(os.path.join(args.out, "lights.json"), "w"), indent=2)
    print(f"saved {args.num} images to {args.out}")


if __name__ == "__main__":
    main()
