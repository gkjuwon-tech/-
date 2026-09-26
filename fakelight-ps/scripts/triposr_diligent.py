"""TripoSR을 DiLiGenT 단일 사진 벤치마크로 채점: 우리와 같은 입력 크롭 1장 → TripoSR 메쉬 → 입력 시점 노멀 → 정답과 비교.

3D 생성 모델이 사진에 보이는 면을 얼마나 정확하게 만드는지 재는 용도 (우리 파이프라인에는 쓰지 않음).
  1) results/diligent/{obj}PNG_025의 input_crop.png + mask_crop.png → TripoSR 공식 전처리 (전경 85%, 회색 배경)
  2) TripoSR 메쉬를 입력 카메라 좌표계로 돌림 (버니로 구한 축 회전, TripoSR 좌표계는 모델 고정)
  3) 정사영으로 노멀을 그리되, 투영 실루엣의 bbox를 크롭 마스크 bbox에 맞춤 (크기/위치만 맞추고 형태는 안 건드림)
  4) 크롭 좌표의 정답 노멀과 마스크 안 평균 각도 오차. 메쉬가 안 덮은 픽셀은 평평한 노멀(0,0,1)로 채점

사용법 (TripoSR 가상환경):
    /home/user/tsr_venv/bin/python scripts/triposr_diligent.py --objs harvest cat ball bear reading
"""
import argparse
import json
import os
import sys

import numpy as np
import scipy.io as sio
import torch
import trimesh
from PIL import Image

sys.path.insert(0, "/home/user/triposr")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsr.system import TSR  # noqa: E402
from tsr.utils import resize_foreground  # noqa: E402

R_TSR = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)   # TripoSR 좌표 → 입력 카메라 (x 오른쪽, y 위, z 카메라 쪽)


def gt_crop(crop, gt, F):
    y0, x0, s = crop["y0"], crop["x0"], crop["size"]
    H, W = gt.shape[:2]
    g = np.zeros((s, s, 3))
    ys, xs, ye, xe = max(0, y0), max(0, x0), min(H, y0 + s), min(W, x0 + s)
    g[ys - y0:ye - y0, xs - x0:xe - x0] = gt[ys:ye, xs:xe]
    out = np.stack([np.asarray(Image.fromarray(g[..., c].astype(np.float32), "F").resize((F, F), Image.BILINEAR))
                    for c in range(3)], 2)
    return out / (np.linalg.norm(out, axis=2, keepdims=True) + 1e-12)


def render_normals(v, f, cm):
    """v: 카메라 좌표 꼭짓점. 투영 bbox를 크롭 마스크 bbox에 맞춘 정사영 z-버퍼 노멀."""
    F = cm.shape[0]
    ys, xs = np.nonzero(cm)
    px_lo, px_hi, py_lo, py_hi = xs.min(), xs.max(), ys.min(), ys.max()
    sx = (px_hi - px_lo) / (v[:, 0].max() - v[:, 0].min())
    sy = (py_hi - py_lo) / (v[:, 1].max() - v[:, 1].min())
    s = (sx + sy) / 2
    px = (v[:, 0] - v[:, 0].min()) * s + px_lo
    py = py_hi - (v[:, 1] - v[:, 1].min()) * s
    pz = v[:, 2]
    m = trimesh.Trimesh(v, f, process=False)
    vn = m.vertex_normals
    zb = np.full((F, F), -np.inf)
    nb = np.zeros((F, F, 3))
    for tri in f:
        x, y, z = px[tri], py[tri], pz[tri]
        x0, x1 = int(max(np.floor(x.min()), 0)), int(min(np.ceil(x.max()), F - 1))
        y0, y1 = int(max(np.floor(y.min()), 0)), int(min(np.ceil(y.max()), F - 1))
        if x1 < x0 or y1 < y0:
            continue
        den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
        if abs(den) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
        w2 = 1 - w0 - w1
        ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not ins.any():
            continue
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        sub = zb[y0:y1 + 1, x0:x1 + 1]
        upd = ins & (zz > sub)
        sub[upd] = zz[upd]
        n = w0[..., None] * vn[tri[0]] + w1[..., None] * vn[tri[1]] + w2[..., None] * vn[tri[2]]
        nb[y0:y1 + 1, x0:x1 + 1][upd] = n[upd]
    nb /= np.linalg.norm(nb, axis=2, keepdims=True) + 1e-12
    return nb, np.isfinite(zb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", nargs="+", default=["harvest", "cat", "ball", "bear", "reading"])
    ap.add_argument("--data", default="data/DiLiGenT/pmsData")
    ap.add_argument("--out", default="results/triposr_diligent")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    ang = lambda a, b: np.degrees(np.arccos(np.clip((a * b).sum(2), -1, 1)))
    rows = {}
    for obj in args.objs:
        f = f"results/diligent/{obj}PNG_025"
        crop = json.load(open(f"{f}/crop.json"))
        F = crop["frame"]
        img = np.asarray(Image.open(f"{f}/input_crop.png").convert("RGB"))
        cm = np.asarray(Image.open(f"{f}/mask_crop.png").convert("L")) > 127
        rgba = Image.fromarray(np.dstack([img, (cm * 255).astype(np.uint8)]))
        pre = resize_foreground(rgba, 0.85)
        a = np.asarray(pre).astype(np.float32) / 255
        rgb = a[..., :3] * a[..., 3:4] + (1 - a[..., 3:4]) * 0.5
        pre = Image.fromarray((rgb * 255).astype(np.uint8))
        with torch.no_grad():
            codes = model([pre], device="cpu")
        mesh = model.extract_mesh(codes, True, resolution=256)[0]
        v = (mesh.vertices - mesh.bounding_box.centroid) @ R_TSR.T
        fcs = mesh.faces[:, ::-1]                                   # skimage 마칭큐브 → 바깥 방향 면
        mesh.export(f"{args.out}/{obj}_mesh.obj")
        n, cov = render_normals(v, fcs, cm)
        od = f"{args.data}/{crop['obj']}"
        gt = gt_crop(crop, sio.loadmat(f"{od}/Normal_gt.mat")["Normal_gt"].astype(np.float64), F)
        valid = cm & (np.linalg.norm(gt, axis=2) > 0.5)
        n_eval = np.where(cov[..., None], n, np.array([0, 0, 1.0]))
        mae = float(ang(n_eval, gt)[valid].mean())
        flat = float(ang(np.dstack([0 * cm, 0 * cm, 1 + 0 * cm]).astype(float), gt)[valid].mean())
        ours = json.load(open(f"{f}/scores.json"))
        rows[obj] = dict(triposr=mae, coverage=float((cov & valid).sum() / valid.sum()), flat=flat,
                         ours=ours.get("합체 + 디테일 보정 (최종)"), rose=ours.get("RoSE (같은 사진)"))
        vis = ((n_eval * 0.5 + 0.5) * 255 * valid[..., None]).astype(np.uint8)
        gvis = ((gt * 0.5 + 0.5) * 255 * valid[..., None]).astype(np.uint8)
        Image.fromarray(np.concatenate([img, gvis, vis], 1)).save(f"{args.out}/{obj}_compare.png")
        print(obj, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in rows[obj].items()}, flush=True)
    json.dump(rows, open(f"{args.out}/scores.json", "w"), indent=2)


if __name__ == "__main__":
    main()
