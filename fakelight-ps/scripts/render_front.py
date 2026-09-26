"""스탠포드 버니 앞면 렌더러 (순수 numpy, GPU/OpenGL 필요 없음).

정사영 카메라로 정면 한 장만 찍는다. MV-Adapter 입력용 RGB(흰 배경)와
나중에 채점할 때 쓸 정답 노멀맵, 마스크도 같이 저장한다.

사용법:
    python scripts/render_front.py --mesh assets/stanford-bunny.obj --out renders
"""
import argparse
import os

import numpy as np
import trimesh
from PIL import Image


def look_rotation(yaw_deg: float) -> np.ndarray:
    """y축 기준 회전. 버니 얼굴이 카메라(+z)를 보게 돌리는 용도."""
    t = np.deg2rad(yaw_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rasterize(verts, faces, vnormals, res):
    """정사영 z-버퍼 래스터라이저. 카메라는 +z에서 -z 방향을 본다.

    반환: 픽셀별 보간 노멀 (res, res, 3), 마스크 (res, res)
    """
    zbuf = np.full((res, res), -np.inf)
    nbuf = np.zeros((res, res, 3))

    # 화면 좌표: x → 열, y(위) → 행(아래로 증가)
    px = (verts[:, 0] * 0.5 + 0.5) * (res - 1)
    py = (0.5 - verts[:, 1] * 0.5) * (res - 1)
    pz = verts[:, 2]

    for f in faces:
        x, y, z = px[f], py[f], pz[f]
        x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), res - 1))
        y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), res - 1))
        if x0 > x1 or y0 > y1:
            continue
        area = (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
        if abs(area) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        # 무게중심 좌표
        w0 = ((x[1] - gx) * (y[2] - gy) - (x[2] - gx) * (y[1] - gy)) / area
        w1 = ((x[2] - gx) * (y[0] - gy) - (x[0] - gx) * (y[2] - gy)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        closer = inside & (zz > sub)
        if not closer.any():
            continue
        sub[closer] = zz[closer]
        n = (w0[..., None] * vnormals[f[0]] + w1[..., None] * vnormals[f[1]]
             + w2[..., None] * vnormals[f[2]])
        nbuf[y0:y1 + 1, x0:x1 + 1][closer] = n[closer]

    mask = np.isfinite(zbuf)
    nbuf[mask] /= np.linalg.norm(nbuf[mask], axis=1, keepdims=True) + 1e-12
    return nbuf, mask


def shade(normals, mask, albedo=(0.78, 0.76, 0.74)):
    """클레이 느낌 음영: 부드러운 키 라이트 + 필 라이트 + 환경광.

    MV-Adapter가 형태를 잘 읽게 하려고 그림자 없는 무난한 스튜디오 조명을 쓴다.
    """
    key = np.array([0.35, 0.45, 0.82]); key /= np.linalg.norm(key)
    fill = np.array([-0.6, 0.1, 0.8]); fill /= np.linalg.norm(fill)
    ndk = np.clip(normals @ key, 0, 1)
    ndf = np.clip(normals @ fill, 0, 1)
    intensity = 0.25 + 0.65 * ndk + 0.25 * ndf
    rgb = np.clip(intensity[..., None] * np.array(albedo), 0, 1)
    rgb[~mask] = 1.0  # 흰 배경
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="assets/stanford-bunny.obj")
    ap.add_argument("--out", default="renders")
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--yaw", type=float, default=0.0, help="버니를 y축으로 돌리는 각도")
    ap.add_argument("--fill", type=float, default=0.85, help="화면에서 물체가 차지하는 비율")
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh", process=True)
    v = mesh.vertices - mesh.bounding_box.centroid
    v = v @ look_rotation(args.yaw).T
    v = v / (np.abs(v[:, :2]).max() / args.fill)  # x,y를 [-fill, fill]에 맞춤
    m = trimesh.Trimesh(v, mesh.faces, process=False)
    vn = m.vertex_normals

    normals, mask = rasterize(v, m.faces, vn, args.res)
    rgb = shade(normals, mask)

    os.makedirs(args.out, exist_ok=True)
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(
        os.path.join(args.out, "bunny_front.png"))
    rgba = np.dstack([rgb, mask.astype(float)])
    Image.fromarray((rgba * 255).round().astype(np.uint8), "RGBA").save(
        os.path.join(args.out, "bunny_front_rgba.png"))
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        os.path.join(args.out, "bunny_front_mask.png"))
    # 정답 노멀 (카메라 좌표, [-1,1] → [0,255]); 16비트 npy도 같이 저장
    nvis = normals * 0.5 + 0.5
    nvis[~mask] = 0
    Image.fromarray((nvis * 255).round().astype(np.uint8)).save(
        os.path.join(args.out, "bunny_front_normal_gt.png"))
    np.save(os.path.join(args.out, "bunny_front_normal_gt.npy"), normals.astype(np.float32))
    print(f"saved to {args.out}  (mask pixels: {mask.sum()})")


if __name__ == "__main__":
    main()
