# 카메라 역추정: 궤도 영상 프레임들 → VGGT (Meta, CVPR 2025 최우수 논문) → 프레임별 카메라, 깊이, 점구름.
# Veo처럼 카메라 값을 안 주는 영상 모델의 결과를 우리 파이프라인(짭광 → stage2)에 넣기 위한 단계.
# 입력 데이터셋: frames/*.png (+ 선택: mask/*.png). 정답 카메라(transforms.json, SEVA 출력)가 있으면 오차도 잰다.
import glob, json, os, subprocess, sys, time

def sh(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=True)
    print(f"[exit={r.returncode} {time.time()-t:.0f}s]", flush=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r.returncode == 0

W = "/kaggle/working"
TAG = "__TAG__"
OUT = f"{W}/out/{TAG}"
os.makedirs(OUT, exist_ok=True)
sh("pip install -q git+https://github.com/facebookresearch/vggt.git")

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

SRC = os.path.dirname(glob.glob("/kaggle/input/**/frames/*.png", recursive=True)[0])
paths = sorted(glob.glob(f"{SRC}/*.png"))
print(len(paths), "frames", flush=True)
dev = "cuda"
model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()
imgs = load_and_preprocess_images(paths).to(dev)
t0 = time.time()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    pred = model(imgs[None])
extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], imgs.shape[-2:])
res = dict(seconds=round(time.time() - t0, 1), n=len(paths), size=list(imgs.shape[-2:]))
extr = extr[0].float().cpu().numpy()                  # (N, 3, 4) w2c, OpenCV, 첫 프레임 = 기준
intr = intr[0].float().cpu().numpy()                  # (N, 3, 3), 전처리된 크기 기준 픽셀
np.save(f"{OUT}/extrinsic_w2c.npy", extr)
np.save(f"{OUT}/intrinsic.npy", intr)
np.save(f"{OUT}/depth.npy", pred["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16))
np.save(f"{OUT}/depth_conf.npy", pred["depth_conf"][0].float().cpu().numpy().astype(np.float16))
json.dump([os.path.basename(p) for p in paths], open(f"{OUT}/frames.json", "w"))

# 정답 카메라가 있으면 (SEVA transforms.json, OpenGL c2w) 회전 오차와 위치 오차 (유사변환 정렬 후)
gt = glob.glob("/kaggle/input/**/transforms.json", recursive=True)
if gt:
    meta = json.load(open(gt[0]))
    G = np.array([f["transform_matrix"] for f in meta["frames"]])[:len(paths)]
    G[:, :3, 1:3] *= -1                                # OpenGL → OpenCV
    P = np.stack([np.linalg.inv(np.vstack([e, [0, 0, 0, 1]])) for e in extr])   # 예측 c2w
    # 첫 프레임 기준 상대 회전 오차
    def rel(M):
        return np.stack([np.linalg.inv(M[0]) @ m for m in M])
    Gr, Pr = rel(G), rel(P)
    rot_err = [np.degrees(np.arccos(np.clip((np.trace(g[:3, :3].T @ p[:3, :3]) - 1) / 2, -1, 1))) for g, p in zip(Gr, Pr)]
    # 카메라 중심 궤적: 크기 맞춘 뒤 비교
    cg, cp = Gr[:, :3, 3], Pr[:, :3, 3]
    s = np.linalg.norm(cg) / (np.linalg.norm(cp) + 1e-9)
    pos_err = np.linalg.norm(cg - s * cp, axis=1) / (np.linalg.norm(cg, axis=1).max() + 1e-9)
    res.update(rot_err_mean=float(np.mean(rot_err)), rot_err_max=float(np.max(rot_err)),
               pos_err_rel_mean=float(pos_err.mean()), rot_err=[round(float(x), 2) for x in rot_err])
json.dump(res, open(f"{OUT}/stats.json", "w"), indent=2)
print(json.dumps(res, indent=2), flush=True)
