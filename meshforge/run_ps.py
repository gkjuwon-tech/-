"""v5: photometric stereo on the front view -> real depth -> projection that does not lie.
usage: python run_ps.py REF SHEET VIEW30 OUTDIR"""
import json, os, sys, time
import numpy as np
from PIL import Image
from mf.sheet import load_cells
from mf.segment import mask_from_bg, canonicalize
from mf.hull import Grid, carve, reprojection_scores, calibrate_offsets, render, iou
from mf.photometric import lights18, normals_from_depth, shade, to_linear, solve, integrate
from mf.composite import align_to_guide, agree
from mf.fidelity import edge_fscore
from mf.warp import warp
from mf.imagegen import generate_many

ref, sheet, v30, od = sys.argv[1:5]
os.makedirs(od, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
save = lambda arr, name: Image.fromarray(arr).save(os.path.join(od, name))

refimg = np.asarray(Image.open(ref).convert("RGB"))
m0, r0 = canonicalize(mask_from_bg(refimg), refimg)                                   # 512
m0h, r0h = canonicalize(mask_from_bg(refimg), refimg, top=64, bottom=960, size=1024)  # 1024
save(r0h, "base_1024.png")
cells = load_cells(sheet)
V, RGB = {0: m0}, {0: r0}
for t in (45, 315, 270):
    V[t], RGB[t] = canonicalize(mask_from_bg(cells[t]), cells[t])
w30 = np.asarray(Image.open(v30).convert("RGB"))
V[30], RGB[30] = mask_from_bg(w30), w30

g4, g = Grid(step=4), Grid(step=2)
off = calibrate_offsets(g4, V, search=10, step=2, rounds=1)
occ = carve(g, V, off)
sc = reprojection_scores(g, occ, V, off)
log("hull (ref front + 30, 45, 315, 270):", {k: round(v, 3) for k, v in sorted(sc.items())})
np.savez_compressed(os.path.join(od, "hull.npz"), occ=occ, step=2, off=json.dumps({str(k): v for k, v in off.items()}))

sil, hdepth, _ = render(g, occ, 0, off[0])
mask = m0 & sil
Nh = normals_from_depth(hdepth, sil, sigma=2.5)
L, desc = lights18()
prompt = open("prompts/relight.txt").read()
jobs = []
for k in range(18):
    gd = shade(Nh, sil, L[k])
    Image.fromarray(gd).resize((1024, 1024), Image.LANCZOS).save(os.path.join(od, f"guide_{k:02d}.png"))
    outp = os.path.join(od, f"lit_{k:02d}.png")
    if not os.path.exists(outp):
        jobs.append((prompt.format(desc=desc[k]), [os.path.join(od, "base_1024.png"), os.path.join(od, f"guide_{k:02d}.png")], outp))
log(f"generating {len(jobs)} relit images (4 in parallel)...")
generate_many(jobs, workers=4)
lit, ok = [], []
for k in range(18):
    p = os.path.join(od, f"lit_{k:02d}.png")
    if not os.path.exists(p):
        continue
    im = np.asarray(Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS))
    im, sh = align_to_guide(im, r0, mask)
    lm = mask_from_bg(im, tol=10)
    agreement = iou(lm, m0)
    log(f"light {k:02d}: shift {tuple(round(s, 1) for s in sh)}, silhouette IoU vs base {agreement:.3f}")
    if agreement > 0.9:
        lit.append(to_linear(im)); ok.append(k)
log(f"{len(ok)}/18 relit images usable")
I = np.stack(lit)
N, rho, res, Lc = solve(I, L[ok], mask)
ang = np.degrees(np.arccos(np.clip((N * Nh).sum(-1), -1, 1)))
log(f"PS residual median {np.median(res[mask]):.3f}; angle PS vs hull normals median {np.median(ang[mask]):.1f} deg")
save(((N * 0.5 + 0.5) * 255 * mask[..., None]).astype(np.uint8), "normals_ps.png")
save(((Nh * 0.5 + 0.5) * 255 * sil[..., None]).astype(np.uint8), "normals_hull.png")
weights = np.clip(1.0 - res / 0.5, 0.05, 1.0)
D = integrate(N, mask, anchor=hdepth, lam=0.02, weights=weights)
np.savez_compressed(os.path.join(od, "front_depth.npz"), ps=D, hull=np.where(mask, hdepth, np.nan), mask=mask, normals=N)
log(f"depth: PS vs hull front surface, median |diff| {np.nanmedian(np.abs(D - hdepth)[mask]):.1f}px, "
    f"90th pct {np.nanpercentile(np.abs(D - hdepth)[mask], 90):.1f}px")

# the test that matters: does the projection land where the independent witnesses say it should?
for tgt in (30, 45):
    for name, dep in (("hull", hdepth), ("PS", D)):
        img, cov, _ = warp(r0, np.nan_to_num(dep), mask, 0, tgt, off[0], off[tgt])
        tm = V[tgt]
        inside = (cov & tm).sum() / cov.sum()
        ag = cov & tm & agree(img, RGB[tgt])
        ef = edge_fscore(img, RGB[tgt], cov & tm)
        log(f"0 -> {tgt} via {name:>4}: lands on object {inside:.1%}, coarse agreement {ag.sum() / cov.sum():.1%}, edge-F {ef:.3f}")
        save(np.concatenate([np.where(cov[..., None], img, 128).astype(np.uint8), RGB[tgt]], 1), f"warp_{name}_{tgt}.png")
