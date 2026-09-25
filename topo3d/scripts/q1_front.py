"""Q1: 정면 한 장 → 쿼드 메시, 루시 정답으로 채점.

사용: python scripts/q1_front.py [--depth work/da3_mv504_pose.npz] [--view 0] [--out work/q1]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topo.lift import lift  # noqa: E402
from topo.raster import quads_to_tris, render  # noqa: E402

GT_DIR = "data/lucy/turnaround_8v"
IN_DIR = "inputs/lucy_8v"

# 쿼드 방향 분류 색 (BGR): 앞 / 위 / 아래 / 옆 / 절벽
CLASS_COL = np.array([[120, 200, 70], [60, 200, 240], [150, 90, 230], [60, 140, 250], [140, 60, 120]], np.uint8)
CLASS_NAME = ["front", "up", "down", "side", "grazing"]


def load_cam(cams, i):
    v = cams["views"][i]
    return np.array(v["K"]), np.array(v["R"]), np.array(v["t"])


def upsample_depth(d_lo, mask_hi):
    """저해상도 깊이를 마스크 해상도로 올린다. 배경 쪽 값을 가장 가까운 전경 값으로 채워서 경계 번짐을 막는다."""
    H, W = mask_hi.shape
    m_lo = cv2.resize(mask_hi.astype(np.uint8), d_lo.shape[::-1], interpolation=cv2.INTER_AREA) > 0
    _, idx = cv2.distanceTransformWithLabels((~m_lo).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(m_lo)
    lut = np.zeros((idx.max() + 1, 2), int)
    lut[idx[m_lo]] = np.stack([ys, xs], 1)
    src = lut[idx]
    filled = d_lo[src[..., 0], src[..., 1]]
    return cv2.resize(filled, (W, H), interpolation=cv2.INTER_CUBIC)


def quad_normals(V, Q):
    n = np.cross(V[Q[:, 2]] - V[Q[:, 0]], V[Q[:, 3]] - V[Q[:, 1]])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def vertex_normals(V, Q, fn):
    vn = np.zeros_like(V)
    for k in range(4):
        np.add.at(vn, Q[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def classify(fn, R):
    """카메라 기준 쿼드 방향 분류."""
    nc = fn @ R.T                      # 카메라 좌표 (x 오른쪽, y 아래, z 앞)
    toward = -nc[:, 2]
    cls = np.full(len(fn), 3)
    cls[toward > np.cos(np.deg2rad(40))] = 0
    rest = cls != 0
    cls[rest & (-nc[:, 1] > np.abs(nc[:, 0]))] = 1       # 위를 봄
    cls[rest & (nc[:, 1] > np.abs(nc[:, 0]))] = 2        # 아래를 봄
    cls[toward < np.cos(np.deg2rad(80))] = 4             # 절벽 (시선과 거의 평행)
    return cls


def shade(n_world, R):
    nc = n_world @ R.T
    L = np.array([-0.45, -0.6, -0.66])
    L /= np.linalg.norm(L)
    return np.clip(0.2 + 0.8 * np.clip(nc @ L, 0, 1), 0, 1)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def soft_iou(a, b):
    return np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", default="work/da3_mv504_pose.npz")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--out", default="work/q1")
    ap.add_argument("--gt-depth", action="store_true", help="DA3 대신 정답 깊이로 엔진만 검증")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cams = json.load(open(os.path.join(IN_DIR, "cameras.json")))
    name = cams["views"][args.view]["name"]
    K, R, t = load_cam(cams, args.view)
    H, W = cams["height"], cams["width"]
    rgba = cv2.imread(os.path.join(IN_DIR, f"{name}.png"), cv2.IMREAD_UNCHANGED)
    alpha = rgba[..., 3].astype(np.float32) / 255
    mask = alpha >= 0.5
    gt_depth = np.load(os.path.join(GT_DIR, f"{name}_depth.npy"))
    gt_norm = np.load(os.path.join(GT_DIR, f"{name}_normal.npy"))

    if args.gt_depth:
        depth = gt_depth.copy()
        depth[~mask] = 0
    else:
        d_lo = np.load(args.depth)["depth"][args.view]
        depth = upsample_depth(d_lo, mask)

    m = lift(mask, depth, K, R, t)
    V, Q = m["V"], m["Q"]
    fn = quad_normals(V, Q)
    vn = vertex_normals(V, Q, fn)
    cls = classify(fn, R)
    rep = {"view": name, "depth_source": "gt" if args.gt_depth else args.depth,
           "vertices": int(len(V)), "quads": int(len(Q)), "triangles": 0,
           "cut_edges": int(m["cut_h"].sum() + m["cut_v"].sum()),
           "class_share": {CLASS_NAME[k]: round(float((cls == k).mean()), 4) for k in range(5)}}

    tris = quads_to_tris(Q)
    # 1) 자기 뷰 실루엣
    r1 = render(V, tris, K, R, t, H, W, attr=vn, ssaa=4)
    rep["iou_binary"] = round(float(iou(r1["cover"] >= 0.5, mask)), 5)
    rep["iou_soft"] = round(float(soft_iou(r1["cover"], alpha)), 5)

    # 2) 깊이·법선 정확도 (정답과 비교)
    both = mask & (gt_depth > 0) & (r1["depth"] > 0)
    e = np.abs(r1["depth"][both] - gt_depth[both])
    A = np.stack([r1["depth"][both], np.ones(both.sum())], 1)
    s, b = np.linalg.lstsq(A, gt_depth[both], rcond=None)[0]
    ea = np.abs(s * r1["depth"][both] + b - gt_depth[both])
    rng = float(gt_depth[both].max() - gt_depth[both].min())
    rep["depth_err_raw_median_over_range"] = round(float(np.median(e) / rng), 4)
    rep["depth_err_aligned_median_over_range"] = round(float(np.median(ea) / rng), 4)
    rn = r1["attr"][both]
    rn /= np.linalg.norm(rn, axis=1, keepdims=True) + 1e-12
    gn = gt_norm[both]
    ang = np.degrees(np.arccos(np.clip((rn * gn).sum(1), -1, 1)))
    rep["normal_err_deg_median"] = round(float(np.median(ang)), 2)
    rep["normal_err_deg_mean"] = round(float(ang.mean()), 2)

    # 3) 다른 뷰에서 봤을 때 정답 실루엣 밖으로 삐져나온 면 (막 / 잘못된 깊이)
    spill = {}
    for j in range(len(cams["views"])):
        if j == args.view:
            continue
        Kj, Rj, tj = load_cam(cams, j)
        nj = cams["views"][j]["name"]
        aj = cv2.imread(os.path.join(IN_DIR, f"{nj}.png"), cv2.IMREAD_UNCHANGED)[..., 3] >= 128
        rj = render(V, tris, Kj, Rj, tj, H, W)["cover"] >= 0.5
        spill[nj] = round(float((rj & ~aj).sum() / max(rj.sum(), 1)), 4)
    rep["spill_outside_gt_silhouette"] = spill

    # 시각화
    sh = shade(vn, R)
    rs = render(V, tris, K, R, t, H, W, attr=np.c_[sh], ssaa=2)
    img = np.full((H, W), 110, np.float32)
    hit = rs["cover"] > 0
    img[hit] = rs["attr"][hit, 0] * 240 * rs["cover"][hit] + 110 * (1 - rs["cover"][hit])
    front = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gt_img = cv2.imread(os.path.join(IN_DIR, f"{name}_gray.png"))
    side_imgs = []
    for j in (1, 2):
        Kj, Rj, tj = load_cam(cams, j)
        nj = cams["views"][j]["name"]
        sj = shade(vn, Rj)
        rj = render(V, tris, Kj, Rj, tj, H, W, attr=np.c_[sj], ssaa=2)
        im = np.full((H, W), 110, np.float32)
        hj = rj["cover"] > 0
        im[hj] = rj["attr"][hj, 0] * 240
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        aj = (cv2.imread(os.path.join(IN_DIR, f"{nj}.png"), cv2.IMREAD_UNCHANGED)[..., 3] >= 128).astype(np.uint8)
        cnt, _ = cv2.findContours(aj, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnt, -1, (0, 0, 255), 2)
        cv2.putText(im, f"front mesh seen from {nj[5:]} deg (red = GT silhouette)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        side_imgs.append(im)
    cv2.putText(gt_img, "input", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(front, "reconstructed quad mesh", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    panel = np.hstack([gt_img, front] + side_imgs)
    cv2.imwrite(os.path.join(args.out, "overview.png"), cv2.resize(panel, None, fx=0.45, fy=0.45,
                                                                    interpolation=cv2.INTER_AREA))

    # 얼굴 부근 와이어프레임 클로즈업 (방향 분류 색 + 셰이딩)
    ys, xs = np.nonzero(mask)
    top = ys.min()
    cy0, cx0 = top + 110, int(np.median(xs[ys < top + 200])) - 60
    ch, cw, z = 120, 120, 8
    Pc = V @ R.T + t
    px = (K[0, 0] * Pc[:, 0] / Pc[:, 2] + K[0, 2] - cx0) * z
    py = (K[1, 1] * Pc[:, 1] / Pc[:, 2] + K[1, 2] - cy0) * z
    canvas = np.zeros((ch * z, cw * z, 3), np.uint8)
    canvas[:] = (24, 16, 12)
    sel = np.nonzero((m["pix"][:, 0] >= cy0 - 1) & (m["pix"][:, 0] < cy0 + ch + 1) &
                     (m["pix"][:, 1] >= cx0 - 1) & (m["pix"][:, 1] < cx0 + cw + 1))[0]
    shq = shade(fn[sel], R)
    order = np.argsort(-Pc[Q[sel], 2].mean(1))
    for k in order:
        qi = sel[k]
        poly = np.stack([px[Q[qi]], py[Q[qi]]], 1).astype(np.int32)
        col = (CLASS_COL[cls[qi]].astype(float) * (0.35 + 0.65 * shq[k])).astype(np.uint8)
        cv2.fillConvexPoly(canvas, poly, col.tolist())
        cv2.polylines(canvas, [poly], True, (40, 200, 220), 1)
    cv2.imwrite(os.path.join(args.out, "closeup_face.png"), canvas)

    # OBJ (진짜 쿼드) 저장
    with open(os.path.join(args.out, f"{name}_quads.obj"), "w") as fp:
        fp.write("# topo3d Q1 quad mesh\n")
        np.savetxt(fp, V, fmt="v %.6f %.6f %.6f")
        for k in range(5):
            fp.write(f"g {CLASS_NAME[k]}\n")
            np.savetxt(fp, Q[cls == k] + 1, fmt="f %d %d %d %d")
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
