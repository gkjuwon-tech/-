# topo3d: 확대 크롭 MoGe-2. 뷰마다 물체 영역을 겹치는 타일로 잘라 3배 확대 후 추론, 원래 크기로 줄여 저장.
import glob, json, math, subprocess, time
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
subprocess.run("pip install -q git+https://github.com/microsoft/MoGe.git", shell=True)
import cv2, torch
from moge.model.v2 import MoGeModel

IN = glob.glob("/kaggle/input/**/cameras.json", recursive=True)[0].rsplit("/", 1)[0]
cams = json.load(open(f"{IN}/cameras.json"))
TILE, STRIDE, ZOOM = 320, 192, 3
m = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval()
out = {"boxes": [], "normals": [], "views": []}
for vi, v in enumerate(cams["views"]):
    img = cv2.imread(f"{IN}/{v['name']}_gray.png")
    a = cv2.imread(f"{IN}/{v['name']}.png", cv2.IMREAD_UNCHANGED)[..., 3] >= 128
    H, W = a.shape
    f = v["K"][0][0]
    ys, xs = np.nonzero(a)
    y0s = list(range(max(ys.min() - 32, 0), max(ys.max() - TILE + 33, 1), STRIDE)) + [min(ys.max() + 32, H) - TILE]
    x0s = list(range(max(xs.min() - 32, 0), max(xs.max() - TILE + 33, 1), STRIDE)) + [min(xs.max() + 32, W) - TILE]
    n = 0
    for y0 in sorted(set(max(0, y) for y in y0s)):
        for x0 in sorted(set(max(0, x) for x in x0s)):
            if a[y0:y0 + TILE, x0:x0 + TILE].mean() < 0.05:
                continue
            crop = cv2.resize(img[y0:y0 + TILE, x0:x0 + TILE], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
            x = torch.tensor(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
            fovx = math.degrees(2 * math.atan(TILE / 2 / f))
            with torch.no_grad():
                o = m.infer(x, fov_x=fovx, resolution_level=9)
            nm = cv2.resize(o["normal"].cpu().numpy(), (TILE, TILE), interpolation=cv2.INTER_AREA)
            out["boxes"].append([y0, x0, TILE]); out["normals"].append(nm.astype(np.float16)); out["views"].append(vi)
            n += 1
    log(f"{v['name']} tiles {n}")
np.savez_compressed("/kaggle/working/moge2_zoom.npz", boxes=np.array(out["boxes"]), normals=np.array(out["normals"]),
                    views=np.array(out["views"]), zoom=ZOOM)
log("done")
