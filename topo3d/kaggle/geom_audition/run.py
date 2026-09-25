# topo3d: 루시 8뷰에 대해 MoGe-2 / StableNormal / DA3(고해상도) 기하 추정 → /kaggle/working/*.npz
import glob, json, math, os, subprocess, sys, time, traceback
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
def sh(c):
    log("$ " + c); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True); return r.returncode

IN = glob.glob("/kaggle/input/**/cameras.json", recursive=True)[0].rsplit("/", 1)[0]
OUT = "/kaggle/working"
cams = json.load(open(f"{IN}/cameras.json"))
names = [v["name"] for v in cams["views"]]
imgs = [f"{IN}/{n}_gray.png" for n in names]
log(f"input {IN}, views {names}")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

import torch, cv2
dev = "cuda"
status = {}

# 1) MoGe-2 ViT-L normal (v1에서 완료, 건너뜀)
try:
    raise RuntimeError("skip")
    sh("pip install -q git+https://github.com/microsoft/MoGe.git")
    from moge.model.v2 import MoGeModel
    m = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
    K = np.array(cams["views"][0]["K"]); fovx = math.degrees(2 * math.atan(cams["width"] / 2 / K[0, 0]))
    N, D, M = [], [], []
    for p in imgs:
        t = time.time()
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        x = torch.tensor(img / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad():
            o = m.infer(x, fov_x=fovx, resolution_level=9)
        N.append(o["normal"].cpu().numpy()); D.append(o["depth"].cpu().numpy()); M.append(o["mask"].cpu().numpy())
        log(f"moge {p.rsplit('/',1)[1]} {time.time()-t:.1f}s")
    np.savez_compressed(f"{OUT}/moge2_vitl.npz", normal=np.array(N, np.float16), depth=np.array(D, np.float32), mask=np.array(M))
    status["moge2"] = "ok"; del m; torch.cuda.empty_cache()
except Exception:
    traceback.print_exc(); status["moge2"] = "fail"

# 2) StableNormal (turbo 먼저, 실패하면 일반)
try:
    sh("pip install -q 'diffusers==0.30.3' 'transformers==4.44.2' 'accelerate==0.34.2' 'huggingface_hub<0.26'")
    from PIL import Image
    N = []
    try:
        pred = torch.hub.load("Stable-X/StableNormal", "StableNormal", trust_repo=True, device=dev)
        tag = "stablenormal"
    except Exception:
        traceback.print_exc()
        pred = torch.hub.load("Stable-X/StableNormal", "StableNormal_turbo", trust_repo=True, device=dev)
        tag = "stablenormal_turbo"
    for p in imgs:
        t = time.time()
        out = pred(Image.open(p).convert("RGB"))
        a = np.asarray(out).astype(np.float32) / 255.0 * 2 - 1      # RGB → [-1,1], 규약은 로컬에서 판정
        if a.shape[:2] != (cams["height"], cams["width"]):
            a = cv2.resize(a, (cams["width"], cams["height"]))
        N.append(a); log(f"{tag} {p.rsplit('/',1)[1]} {time.time()-t:.1f}s")
    np.savez_compressed(f"{OUT}/{tag}.npz", normal=np.array(N, np.float16))
    status[tag] = "ok"; del pred; torch.cuda.empty_cache()
except Exception:
    traceback.print_exc(); status["stablenormal"] = "fail"

# 3) DA3-BASE 멀티뷰, 정답 카메라, 크롭 + 1008
try:
    sh("git clone -q --depth 1 https://github.com/ByteDance-Seed/Depth-Anything-3 /kaggle/da3 && pip install -q --no-deps -e /kaggle/da3 && pip install -q addict einops omegaconf evo pycolmap e3nn plyfile trimesh 'moviepy==1.0.3'")
    sys.path.insert(0, "/kaggle/da3/src")
    from depth_anything_3.api import DepthAnything3
    m = DepthAnything3.from_pretrained("depth-anything/DA3-BASE").to(dev).eval()
    H, W = cams["height"], cams["width"]
    U = np.zeros((H, W), bool)
    for n in names: U |= cv2.imread(f"{IN}/{n}.png", cv2.IMREAD_UNCHANGED)[..., 3] > 0
    ys, xs = np.nonzero(U); pad = 16
    y0, y1, x0, x1 = max(ys.min()-pad, 0), min(ys.max()+pad+1, H), max(xs.min()-pad, 0), min(xs.max()+pad+1, W)
    os.makedirs("/kaggle/crop", exist_ok=True)
    cp, Ks, Es = [], [], []
    for v, p in zip(cams["views"], imgs):
        q = f"/kaggle/crop/{v['name']}.png"; cv2.imwrite(q, cv2.imread(p)[y0:y1, x0:x1]); cp.append(q)
        k = np.array(v["K"]); k[0, 2] -= x0; k[1, 2] -= y0; Ks.append(k)
        e = np.eye(4); e[:3, :3] = v["R"]; e[:3, 3] = v["t"]; Es.append(e)
    for res in (630,):
        t = time.time()
        with torch.no_grad():
            pr = m.inference(cp, intrinsics=np.array(Ks, np.float32), extrinsics=np.array(Es, np.float32), process_res=res)
        log(f"da3 res {res} {time.time()-t:.1f}s depth {pr.depth.shape}")
        np.savez_compressed(f"{OUT}/da3_base_crop{res}.npz", depth=pr.depth.astype(np.float32), conf=pr.conf.astype(np.float16),
                            crop=np.array([y0, y1, x0, x1]))
    status["da3_630"] = "ok"
except Exception:
    traceback.print_exc(); status["da3_630"] = "fail"

json.dump(status, open(f"{OUT}/status.json", "w"))
log(f"status {status}")
