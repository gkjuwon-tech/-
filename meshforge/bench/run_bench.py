"""Ground-truth benchmark: how accurate are normals from generator-relit photometric stereo?
usage: python -m bench.run_bench"""
import json, os, time
import numpy as np
from PIL import Image
from mf.hull import Grid, carve, render
from mf.photometric import lights18, normals_from_depth, shade, to_linear, solve, fuse_frequencies
from mf.composite import align_to_guide
from mf.hull import iou
from mf.imagegen import generate_many
from bench.gt import load_mesh, render_gt, base_image, angular_error

od = "bench/out/stanford"
os.makedirs(od, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
v, vn = load_mesh("bench/data/dragon_recon/dragon_vrip.ply")

VIEW = 0
masks = {}
for t in (0, 30, 60, 90, 120, 150):
    masks[t], d, N = render_gt(v, vn, t)
    if t == VIEW:
        gmask, gdepth, gN = masks[t], d, N
g = Grid(step=2)
occ = carve(g, masks, {})
sil, hdepth, _ = render(g, occ, VIEW, 0.0)
Nh = normals_from_depth(hdepth, sil, sigma=2.5)
mask = gmask & sil
log(f"hull silhouette IoU vs GT at view {VIEW}: {iou(sil, gmask):.3f}")

base = base_image(gN, gmask)
Image.fromarray(base).save(f"{od}/base.png")
Image.fromarray(base).resize((1024, 1024), Image.LANCZOS).save(f"{od}/base_1024.png")
Image.fromarray(((gN * 0.5 + 0.5) * 255 * gmask[..., None]).astype(np.uint8)).save(f"{od}/normals_gt.png")
L, desc = lights18()

# oracle: physically rendered 18-light images (validates the solver, 0-error ceiling)
I_or = np.stack([np.clip(gN @ l, 0, None) * 0.8 for l in L])
N_or, *_ = solve(I_or, L, mask)
log("ORACLE (physical renders) :", {k: round(x, 3) for k, x in angular_error(N_or, gN, mask).items()})
log("HULL normals              :", {k: round(x, 3) for k, x in angular_error(Nh, gN, mask).items()})

prompt = open("prompts/relight.txt").read().replace("dragon figure", "statue").replace("dragon", "statue") \
    .replace("scales, spikes and wing membranes", "surface details")
jobs = []
for k in range(18):
    Image.fromarray(shade(Nh, sil, L[k])).resize((1024, 1024), Image.LANCZOS).save(f"{od}/guide_{k:02d}.png")
    if not os.path.exists(f"{od}/lit_{k:02d}.png"):
        jobs.append((prompt.format(desc=desc[k]), [f"{od}/base_1024.png", f"{od}/guide_{k:02d}.png"], f"{od}/lit_{k:02d}.png"))
log(f"generating {len(jobs)} relit images...")
generate_many(jobs, workers=4)

lit, ok = [], []
from mf.segment import mask_from_bg
for k in range(18):
    p = f"{od}/lit_{k:02d}.png"
    if not os.path.exists(p):
        continue
    im = np.asarray(Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS))
    im, _ = align_to_guide(im, base, mask)
    a = iou(mask_from_bg(im, tol=10), gmask)
    if a > 0.9:
        lit.append(to_linear(im)); ok.append(k)
log(f"{len(ok)}/18 relit images usable")
N_ps, rho, res, Lc = solve(np.stack(lit), L[ok], mask)
N_fu = fuse_frequencies(N_ps, Nh, mask, 12)
N_fu_gt = fuse_frequencies(N_ps, gN, mask, 12)   # upper bound: perfect low frequencies
out = {}
for name, Nx in (("HULL", Nh), ("PS (GPT relit)", N_ps), ("FUSED hull+PS", N_fu), ("FUSED gtlow+PS", N_fu_gt)):
    out[name] = angular_error(Nx, gN, mask)
    log(f"{name:<16}:", {k: round(x, 3) for k, x in out[name].items()})
json.dump(out, open(f"{od}/results.json", "w"), indent=1)
vis = [((X * 0.5 + 0.5) * 255 * mask[..., None]).astype(np.uint8) for X in (gN, Nh, N_ps, N_fu)]
Image.fromarray(np.concatenate(vis, 1)).save(f"{od}/normals_gt_hull_ps_fused.png")
