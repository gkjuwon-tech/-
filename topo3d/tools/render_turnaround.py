"""8뷰 턴어라운드 렌더러 (CPU, numba 래스터라이저).

메시를 수평 45° 간격 원근 카메라로 렌더링하고, 채점용 정답 데이터를 함께 저장한다.

출력 (out_dir):
  view_XXX.png        RGBA, 알파 = 안티앨리어싱된 커버리지
  view_XXX_gray.png   회색 배경 합성본 (턴어라운드 시트 흉내)
  view_XXX_depth.npy  카메라 z 깊이 (float32, 배경 0)
  view_XXX_normal.npy 월드 법선 (float32 HxWx3, 배경 0)
  cameras.json        OpenCV 규약 K, world→camera [R|t], 카메라 위치
  sheet.png           8뷰 컨택트 시트

좌표계: 월드 Y-up. 각도 a의 카메라 위치 = (d·sin a, cy, d·cos a), 원점 축을 바라봄.
즉 0°는 +Z 쪽에서, 90°는 +X 쪽에서 본다 (위에서 볼 때 카메라가 반시계로 돈다).
"""
import argparse
import json
import os

import cv2
import numpy as np
import trimesh
from numba import njit


@njit(cache=True)
def _raster(sx, sy, invz, nrm, faces, H, W):
    zbuf = np.zeros((H, W), np.float32)          # 1/z, 클수록 가까움
    nbuf = np.zeros((H, W, 3), np.float32)
    for f in range(faces.shape[0]):
        a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
        if invz[a] <= 0 or invz[b] <= 0 or invz[c] <= 0:
            continue
        x0, y0, x1, y1, x2, y2 = sx[a], sy[a], sx[b], sy[b], sx[c], sy[c]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xmin = max(int(np.ceil(min(x0, x1, x2) - 0.5)), 0)
        xmax = min(int(np.floor(max(x0, x1, x2) - 0.5)), W - 1)
        ymin = max(int(np.ceil(min(y0, y1, y2) - 0.5)), 0)
        ymax = min(int(np.floor(max(y0, y1, y2) - 0.5)), H - 1)
        for y in range(ymin, ymax + 1):
            py = y + 0.5
            for x in range(xmin, xmax + 1):
                px = x + 0.5
                w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                iz = w0 * invz[a] + w1 * invz[b] + w2 * invz[c]
                if iz > zbuf[y, x]:
                    zbuf[y, x] = iz
                    # 원근 보정 보간
                    p0 = w0 * invz[a] / iz
                    p1 = w1 * invz[b] / iz
                    p2 = 1.0 - p0 - p1
                    for k in range(3):
                        nbuf[y, x, k] = p0 * nrm[a, k] + p1 * nrm[b, k] + p2 * nrm[c, k]
    return zbuf, nbuf


def look_at(C, target):
    """OpenCV 규약 world→camera 회전과 평행이동 (x 오른쪽, y 아래, z 앞)."""
    fwd = target - C
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ C


def shade(n_world, view_dir_world):
    """회색 대리석 느낌 셰이딩: 키 + 필 + 림 라이트, 약한 스페큘러."""
    n = n_world / (np.linalg.norm(n_world, axis=-1, keepdims=True) + 1e-9)
    lights = [  # (월드 방향은 카메라 기준으로 돌려서 모든 뷰가 같은 조명을 받게 함)
        (np.array([-0.5, 0.7, 0.5]), 0.75),
        (np.array([0.6, 0.2, 0.6]), 0.30),
        (np.array([0.0, 0.5, -0.9]), 0.25),
    ]
    right, down, fwd = view_dir_world
    col = np.full(n.shape[:2], 0.18, np.float32)
    for d, k in lights:
        L = d[0] * right - d[1] * down - d[2] * fwd
        L /= np.linalg.norm(L)
        col += k * np.clip(n @ L, 0, 1)
    H = -fwd + (-0.5 * right - 0.7 * down)
    H /= np.linalg.norm(H)
    col += 0.12 * np.clip(n @ H, 0, 1) ** 24
    return np.clip(col, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("out_dir")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--fov", type=float, default=30.0, help="세로 FOV (도)")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--margin", type=float, default=1.12)
    ap.add_argument("--yaw", type=float, default=0.0, help="렌더 전 메시를 Y축으로 회전 (도), 정면을 0°로 맞출 때")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mesh = trimesh.load(args.mesh, process=False)
    V = mesh.vertices.astype(np.float64)
    F = mesh.faces.astype(np.int64)
    N = mesh.vertex_normals.astype(np.float64)
    if args.yaw:
        y = np.deg2rad(args.yaw)
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        V, N = V @ Ry.T, N @ Ry.T
    N = N.astype(np.float32)
    lo, hi = V.min(0), V.max(0)
    cy = (lo[1] + hi[1]) / 2
    target = np.array([0.0, cy, 0.0])
    radius_xz = np.sqrt((V[:, 0] ** 2 + V[:, 2] ** 2).max())
    half_h = (hi[1] - lo[1]) / 2 * args.margin
    t = np.tan(np.deg2rad(args.fov) / 2)
    dist = half_h / t + radius_xz  # 가장 가까운 점까지 고려

    S = args.size * args.ssaa
    f_px = (S / 2) / t
    K_hi = np.array([[f_px, 0, S / 2], [0, f_px, S / 2], [0, 0, 1]])
    K = K_hi.copy()
    K[:2] /= args.ssaa

    cams = {"convention": "opencv, world Y-up, extrinsic = world->camera [R|t]",
            "width": args.size, "height": args.size, "fov_y_deg": args.fov,
            "mesh": os.path.basename(args.mesh), "mesh_yaw_deg": args.yaw, "views": []}
    thumbs = []
    for i in range(args.views):
        ang = 360.0 * i / args.views
        a = np.deg2rad(ang)
        C = np.array([dist * np.sin(a), cy, dist * np.cos(a)])
        R, tvec = look_at(C, target)
        Pc = V @ R.T + tvec
        z = Pc[:, 2]
        invz = np.where(z > 1e-6, 1.0 / z, -1.0)
        sx = (K_hi[0, 0] * Pc[:, 0] * invz + K_hi[0, 2]).astype(np.float64)
        sy = (K_hi[1, 1] * Pc[:, 1] * invz + K_hi[1, 2]).astype(np.float64)
        zb, nb = _raster(sx, sy, invz.astype(np.float64), N, F, S, S)

        cov_hi = zb > 0
        col_hi = shade(nb, (R[0], R[1], R[2])) * cov_hi
        # SSAA 다운샘플: 색·알파는 평균, 깊이·법선은 가장 가까운 샘플
        r = args.ssaa
        alpha = cov_hi.reshape(args.size, r, args.size, r).mean((1, 3))
        csum = col_hi.reshape(args.size, r, args.size, r).sum((1, 3))
        cnt = cov_hi.reshape(args.size, r, args.size, r).sum((1, 3))
        color = np.where(cnt > 0, csum / np.maximum(cnt, 1), 0)
        zb4 = zb.reshape(args.size, r, args.size, r).transpose(0, 2, 1, 3).reshape(args.size, args.size, r * r)
        nb4 = nb.reshape(args.size, r, args.size, r, 3).transpose(0, 2, 1, 3, 4).reshape(args.size, args.size, r * r, 3)
        best = zb4.argmax(-1)
        iz = np.take_along_axis(zb4, best[..., None], -1)[..., 0]
        depth = np.where(iz > 0, 1.0 / np.maximum(iz, 1e-12), 0).astype(np.float32)
        nrm = np.take_along_axis(nb4, best[..., None, None], 2)[:, :, 0]
        nrm = nrm / (np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9) * (iz[..., None] > 0)

        tag = f"view_{int(round(ang)):03d}"
        rgb = np.clip(color[..., None] * np.array([0.95, 0.92, 0.86]) * 1.05, 0, 1)
        rgba = np.dstack([rgb, alpha]).astype(np.float32)
        cv2.imwrite(os.path.join(args.out_dir, f"{tag}.png"),
                    (rgba[..., [2, 1, 0, 3]] * 255 + 0.5).astype(np.uint8))
        bg = np.array([0.52, 0.52, 0.52])
        gray = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
        gray8 = (gray[..., ::-1] * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_dir, f"{tag}_gray.png"), gray8)
        np.save(os.path.join(args.out_dir, f"{tag}_depth.npy"), depth)
        np.save(os.path.join(args.out_dir, f"{tag}_normal.npy"), nrm.astype(np.float32))
        cams["views"].append({"name": tag, "angle_deg": ang, "K": K.tolist(),
                              "R": R.tolist(), "t": tvec.tolist(), "C": C.tolist()})
        th = cv2.resize(gray8, (320, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(th, f"{int(round(ang))}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        thumbs.append(th)
        print(tag, "coverage px", int((alpha > 0).sum()), "depth range",
              float(depth[depth > 0].min()), float(depth.max()))

    with open(os.path.join(args.out_dir, "cameras.json"), "w") as fp:
        json.dump(cams, fp, indent=1)
    cols = 4
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(os.path.join(args.out_dir, "sheet.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
