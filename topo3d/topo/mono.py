"""단일 이미지 기하 모델 래퍼 (결과를 npz로 캐시). 법선은 OpenCV 카메라 좌표 (x 오른쪽, y 아래, z 앞)."""
import os

import cv2
import numpy as np

_M = {}


def run_moge(image_path, cache, model="Ruicheng/moge-2-vitl-normal", resolution_level=9, fov_x_deg=None):
    if os.path.exists(cache):
        return dict(np.load(cache))
    import torch
    from moge.model.v2 import MoGeModel
    if model not in _M:
        _M[model] = MoGeModel.from_pretrained(model).to("cpu").eval()
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    x = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
    with torch.no_grad():
        o = _M[model].infer(x, resolution_level=resolution_level, fov_x=fov_x_deg)
    out = {k: v.cpu().numpy() for k, v in o.items() if hasattr(v, "cpu")}
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, **out)
    return out
