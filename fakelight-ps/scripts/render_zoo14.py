"""정답 메쉬를 3d 레포 stage2의 zoo14 카메라로 렌더 (정사영, ortho_scale 1.1, 거리 2).

stage2 입력 형식 그대로 저장한다:
  <out>/views/cameras.json  {"ortho_scale": 1.1, "resolution": [R, R], "views": {이름: {"matrix_world": 4x4}}}
  <out>/views/rgb/<이름>.png   점토색 + 머리 위 방향광 + 환경광 (재조명 모델 입력용)
  <out>/views/mask/<이름>.png
  <out>/normals_gt/<이름>.npy  카메라 좌표 단위 노멀 (x 오른쪽, y 위, z 카메라 쪽), 없는 곳 0
  <out>/gt_mesh.ply           정규화된 정답 메쉬 (채점용)
카메라 규약은 tools/mvgen_views.py와 같다: 방위각 a(MV-Adapter 기준, 0 = 정면) → 우리 방위각 270 + a,
matrix_world 열 = (right, up, d, 2d), d = 물체에서 카메라 쪽 방향 (Blender/OpenGL 카메라는 -Z를 봄).

사용법 (pyrender + OSMesa):
    PYOPENGL_PLATFORM=osmesa python scripts/render_zoo14.py --mesh /home/user/lucy/lucy_1m.ply --out build/lucy_zoo14
"""
import argparse
import json
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import numpy as np
import pyrender
import trimesh
from PIL import Image

ZOO14 = [("01_front", 0, 0), ("02_right", 90, 0), ("03_back", 180, 0), ("04_left", 270, 0),
         ("05_top", 0, 89.99), ("06_bottom", 0, -89.99),
         ("07_az45_up", 45, 45), ("08_az135_up", 135, 45), ("09_az225_up", 225, 45), ("10_az315_up", 315, 45),
         ("11_az45_dn", 45, -45), ("12_az135_dn", 135, -45), ("13_az225_dn", 225, -45), ("14_az315_dn", 315, -45)]


def camera(az, el):
    """tools/mvgen_views.py camera()와 동일 (az는 우리 방위각)"""
    d = np.array([np.cos(np.radians(el)) * np.cos(np.radians(az)),
                  np.cos(np.radians(el)) * np.sin(np.radians(az)), np.sin(np.radians(el))])
    right = np.cross([0, 0, 1], d)
    right /= np.linalg.norm(right)
    up = np.cross(d, right)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, d, 2 * d
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=int, default=768)
    ap.add_argument("--height", type=float, default=1.0, help="정규화 후 물체 높이 (프레임은 1.1)")
    ap.add_argument("--yaw", type=float, default=0.0, help="물체를 z축으로 돌려 정면을 맞출 각도")
    args = ap.parse_args()

    m = trimesh.load(args.mesh, process=False)
    v = m.vertices - m.bounding_box.centroid
    v = v / m.extents[2] * args.height
    c, s = np.cos(np.radians(args.yaw)), np.sin(np.radians(args.yaw))
    v = v @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    mesh = trimesh.Trimesh(v, m.faces, process=False)
    os.makedirs(args.out, exist_ok=True)
    mesh.export(os.path.join(args.out, "gt_mesh.ply"))
    vn = mesh.vertex_normals
    for d in ["views/rgb", "views/mask", "normals_gt"]:
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    r = pyrender.OffscreenRenderer(args.res, args.res)
    cam = pyrender.OrthographicCamera(xmag=0.55, ymag=0.55, znear=0.01, zfar=10.0)
    mats = {}
    for name, az_mv, el in ZOO14:
        M = camera(270 + az_mv, el)
        mats[name] = M
        R_wc = M[:3, :3].T                                        # 세계 → 카메라 회전
        # 1) 노멀: 카메라 좌표 노멀을 꼭짓점 색으로 넣고 조명 없이 평면 렌더 → 보간 후 다시 정규화
        ncam = vn @ R_wc.T
        col = np.clip((ncam * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        sc = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1, 1, 1])
        sc.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v, m.faces, vertex_colors=col, process=False), smooth=True))
        sc.add(cam, pose=M)
        rgb, depth = r.render(sc, flags=pyrender.RenderFlags.FLAT)
        mask = depth > 0
        n = rgb.astype(np.float64) / 255 * 2 - 1
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
        n[~mask] = 0
        np.save(os.path.join(args.out, "normals_gt", f"{name}.npy"), n.astype(np.float32))
        Image.fromarray((mask * 255).astype(np.uint8)).save(os.path.join(args.out, "views/mask", f"{name}.png"))
        # 2) 점토 렌더: 회색 알베도, 카메라 왼쪽 위에서 오는 방향광 + 약한 환경광
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.25, 0.25, 0.25])
        sc.add(pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.72, 0.70, 0.68, 1.0], metallicFactor=0.0, roughnessFactor=0.9), smooth=True))
        sc.add(cam, pose=M)
        L = np.eye(4)
        ldir = M[:3, :3] @ np.array([-0.4, 0.5, 1.0])              # 카메라 기준 왼쪽 위 앞
        ldir /= np.linalg.norm(ldir)
        z = ldir
        x = np.cross([0, 0, 1], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        L[:3, 0], L[:3, 1], L[:3, 2] = x, np.cross(z, x), z
        sc.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=L)
        img, _ = r.render(sc)
        img = np.where(mask[..., None], img[..., :3], 255).astype(np.uint8)      # 배경은 흰색으로 고정
        Image.fromarray(img).save(os.path.join(args.out, "views/rgb", f"{name}.png"))
        print(name, "mask", round(mask.mean(), 3), flush=True)
    r.delete()
    json.dump({"ortho_scale": 1.1, "resolution": [args.res, args.res],
               "views": {k: {"matrix_world": M.tolist()} for k, M in mats.items()}},
              open(os.path.join(args.out, "views", "cameras.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
