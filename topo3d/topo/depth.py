"""Depth Anything 3 추론 래퍼 (결과를 npz로 캐시)."""
import os

import numpy as np

_MODEL = None


def load_model(path="models/DA3-BASE"):
    global _MODEL
    if _MODEL is None:
        from depth_anything_3.api import DepthAnything3
        _MODEL = DepthAnything3.from_pretrained(path).to("cpu").eval()
    return _MODEL


def run_da3(images, cache, process_res=1008, intrinsics=None, extrinsics=None):
    """images: 파일 경로 리스트. 반환: dict(depth[N,H,W], conf, intrinsics, extrinsics), 입력 해상도로 리사이즈하지 않음."""
    if os.path.exists(cache):
        return dict(np.load(cache))
    import torch
    m = load_model()
    with torch.no_grad():
        p = m.inference(images, intrinsics=intrinsics, extrinsics=extrinsics, process_res=process_res)
    out = {"depth": p.depth, "conf": p.conf, "intrinsics": p.intrinsics, "extrinsics": p.extrinsics}
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, **out)
    return out
