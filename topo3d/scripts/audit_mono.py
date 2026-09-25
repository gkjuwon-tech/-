"""단일 뷰 깊이·법선 모델을 루시 정답과 비교한다.

사용: python scripts/audit_mono.py <npz> <view_name> [--crop]
npz: depth (H,W) 와 normal (H,W,3, 카메라 좌표) 포함.
"""
import json
import sys

import cv2
import numpy as np

npz, name = sys.argv[1], sys.argv[2]
r = dict(np.load(npz))
cams = json.load(open("inputs/lucy_8v/cameras.json"))
v = [c for c in cams["views"] if c["name"] == name][0]
R = np.array(v["R"])
gt_d = np.load(f"data/lucy/turnaround_8v/{name}_depth.npy")
gt_n = np.load(f"data/lucy/turnaround_8v/{name}_normal.npy") @ R.T   # 월드 → 카메라
m = gt_d > 0
# 경계 1px 제외 (안티앨리어싱 섞임)
m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
rep = {"model": npz}
if "depth" in r:
    d = r["depth"]
    d = cv2.resize(d, gt_d.shape[::-1], interpolation=cv2.INTER_LINEAR) if d.shape != gt_d.shape else d
    ok = m & np.isfinite(d)
    A = np.stack([d[ok], np.ones(ok.sum())], 1)
    s, b = np.linalg.lstsq(A, gt_d[ok], rcond=None)[0]
    e = np.abs(s * d[ok] + b - gt_d[ok])
    rep["depth_corr"] = round(float(np.corrcoef(d[ok], gt_d[ok])[0, 1]), 3)
    rep["depth_medErr_over_range"] = round(float(np.median(e) / (gt_d[ok].max() - gt_d[ok].min())), 4)
if "normal" in r:
    n = r["normal"]
    n = cv2.resize(n, gt_d.shape[::-1], interpolation=cv2.INTER_LINEAR) if n.shape[:2] != gt_d.shape else n
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)
    ok = m & np.isfinite(n).all(-1)
    ang = np.degrees(np.arccos(np.clip((n[ok] * gt_n[ok]).sum(-1), -1, 1)))
    rep["normal_err_median_deg"] = round(float(np.median(ang)), 2)
    rep["normal_err_mean_deg"] = round(float(ang.mean()), 2)
    rep["normal_within_11.25deg"] = round(float((ang < 11.25).mean()), 3)
    rep["normal_within_30deg"] = round(float((ang < 30).mean()), 3)
print(json.dumps(rep))
