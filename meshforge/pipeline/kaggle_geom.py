"""Kaggle T4 job: per-frame cameras (VGGT-1B) and normals (StableNormal turbo) for SV3D orbit frames.
Input dataset: <set>/<orbit>/<frame>.png (+ poses.json). Output /kaggle/working/out/<set>/:
  vggt.npz   extrinsic [N,3,4] (world->cam, OpenCV), intrinsic [N,3,3], names, depth_conf summary
  normals/<orbit>__<frame>.npy  float16 camera-space normals as the estimator returns them"""
import glob, json, os, subprocess, sys, time
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("pip install -q git+https://github.com/facebookresearch/vggt rembg onnxruntime-gpu")
import numpy as np, torch
from PIL import Image
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"])
root = None
for r, _, files in os.walk("/kaggle/input"):
    if "poses.json" in files:
        root = os.path.dirname(r); break
sets = {}
for pj in glob.glob(os.path.join(root, "**", "poses.json"), recursive=True):
    orbit = os.path.basename(os.path.dirname(pj))
    tag = orbit.split("_el")[0]
    sets.setdefault(tag, []).extend(sorted(glob.glob(os.path.join(os.path.dirname(pj), "*.png"))))
out = "/kaggle/working/out"
# ---- VGGT: all frames of one object jointly (camera tokens attend across every view)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
for tag, paths in sets.items():
    # the hero frame (azimuth 0, level) first: VGGT's world frame is the first camera's
    paths = sorted(set(paths), key=lambda p: (0 if p.endswith("e+00_a000.png") else 1, p))
    t0 = time.time()
    imgs = load_and_preprocess_images(paths).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        agg, ps_idx = model.aggregator(imgs[None])
        pose_enc = model.camera_head(agg)[-1]
    E, K = pose_encoding_to_extri_intri(pose_enc, imgs.shape[-2:])
    os.makedirs(os.path.join(out, tag), exist_ok=True)
    np.savez(os.path.join(out, tag, "vggt.npz"), extrinsic=E[0].float().cpu().numpy(), intrinsic=K[0].float().cpu().numpy(),
             names=np.array([os.path.basename(os.path.dirname(p)) + "__" + os.path.basename(p) for p in paths]))
    print(tag, len(paths), "frames VGGT", round(time.time() - t0, 1), "s, peak", round(torch.cuda.max_memory_allocated() / 1e9, 1), "GB", flush=True)
    del agg; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
del model; torch.cuda.empty_cache()
# ---- masks (SV3D paints a white background; ivory horns are near-white too, so segment, don't threshold)
from rembg import remove, new_session
sess = new_session("isnet-general-use")
for tag, paths in sets.items():
    md = os.path.join(out, tag, "masks"); os.makedirs(md, exist_ok=True)
    for p in paths:
        a = np.asarray(remove(Image.open(p).convert("RGB"), session=sess, only_mask=True))
        Image.fromarray(a).save(os.path.join(md, os.path.basename(os.path.dirname(p)) + "__" + os.path.basename(p)))
    print(tag, "masks done", flush=True)
# ---- StableNormal turbo (the Stable-X estimator family Hi3DGen builds on)
predictor = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True)
for tag, paths in sets.items():
    nd = os.path.join(out, tag, "normals"); os.makedirs(nd, exist_ok=True)
    t0 = time.time()
    for p in paths:
        im = Image.open(p).convert("RGB")
        n = predictor(im)
        a = np.asarray(n).astype(np.float32) / 255.0 * 2 - 1
        np.save(os.path.join(nd, os.path.basename(os.path.dirname(p)) + "__" + os.path.basename(p)[:-4] + ".npy"), a.astype(np.float16))
    print(tag, "normals", round(time.time() - t0, 1), "s", flush=True)
