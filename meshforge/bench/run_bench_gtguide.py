"""Same benchmark, but the relighting guides come from the ground-truth shape (upper bound on guide quality)."""
import json, os, time
import numpy as np
from PIL import Image
from mf.photometric import lights18, shade, to_linear, solve, fuse_frequencies, normals_from_depth
from mf.composite import align_to_guide
from mf.hull import Grid, carve, render, iou
from mf.segment import mask_from_bg
from mf.imagegen import generate_many
from bench.gt import load_mesh, render_gt, angular_error

src, od = "bench/out/stanford", "bench/out/stanford_gtguide"
os.makedirs(od, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
v, vn = load_mesh("bench/data/dragon_recon/dragon_vrip.ply")
gmask, gdepth, gN = render_gt(v, vn, 0)
masks = {t: render_gt(v, vn, t)[0] for t in (0, 30, 60, 90, 120, 150)}
g = Grid(step=2); occ = carve(g, masks, {}); sil, hdepth, _ = render(g, occ, 0, 0.0)
Nh = normals_from_depth(hdepth, sil, sigma=2.5)
mask = gmask & sil
base = np.asarray(Image.open(f"{src}/base.png"))
L, desc = lights18()
prompt = open("prompts/relight.txt").read().replace("dragon figure", "statue").replace("dragon", "statue") \
    .replace("scales, spikes and wing membranes", "surface details")
jobs = []
for k in range(18):
    Image.fromarray(shade(gN, gmask, L[k])).resize((1024, 1024), Image.LANCZOS).save(f"{od}/guide_{k:02d}.png")
    if not os.path.exists(f"{od}/lit_{k:02d}.png"):
        jobs.append((prompt.format(desc=desc[k]), [f"{src}/base_1024.png", f"{od}/guide_{k:02d}.png"], f"{od}/lit_{k:02d}.png"))
log(f"generating {len(jobs)} relit images with ground-truth guides...")
generate_many(jobs, workers=4)
lit, ok = [], []
for k in range(18):
    p = f"{od}/lit_{k:02d}.png"
    if os.path.exists(p):
        im = np.asarray(Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS))
        im, _ = align_to_guide(im, base, mask)
        if iou(mask_from_bg(im, tol=10), gmask) > 0.9:
            lit.append(to_linear(im)); ok.append(k)
log(f"{len(ok)}/18 usable")
N_ps, *_ = solve(np.stack(lit), L[ok], gmask)
res = {"PS (GT guides)": angular_error(N_ps, gN, gmask),
       "FUSED hull+PS": angular_error(fuse_frequencies(N_ps, Nh, mask, 12), gN, mask)}
for k, r in res.items():
    log(f"{k:<16}:", {a: round(b, 3) for a, b in r.items()})
json.dump(res, open(f"{od}/results.json", "w"), indent=1)
Image.fromarray(np.concatenate([((X * 0.5 + 0.5) * 255 * gmask[..., None]).astype(np.uint8) for X in (gN, N_ps)], 1)).save(f"{od}/normals_gt_ps.png")
