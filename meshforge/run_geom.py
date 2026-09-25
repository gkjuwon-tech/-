"""v4 geometry-first: resolve contradictory opposite views, add new directions with validation.
usage: python run_geom.py SHEET REF OUTDIR"""
import json, os, sys, time
import numpy as np
from PIL import Image
from mf.sheet import load_cells, AZIMUTHS
from mf.segment import mask_from_bg, canonicalize
from mf.hull import Grid, carve, reprojection_scores, calibrate_offsets, render, iou
from mf.render import clay
from mf.imagegen import generate

sheet, ref, od = sys.argv[1:4]
os.makedirs(od, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
save = lambda arr, name: Image.fromarray(arr).save(os.path.join(od, name))

cells = load_cells(sheet)
M, RGB = {}, {}
for t in AZIMUTHS:
    M[t], RGB[t] = canonicalize(mask_from_bg(cells[t]), cells[t])
g4, g = Grid(step=4), Grid(step=2)

def build(masks, fine=True):
    off = calibrate_offsets(g4, masks, search=10, step=2, rounds=1)
    gg = g if fine else g4
    occ = carve(gg, masks, off)
    return occ, off, reprojection_scores(gg, occ, masks, off)

def fmt(sc):
    return "{" + ", ".join(f"{k}:{v:.3f}" for k, v in sorted(sc.items())) + f"}} min {min(sc.values()):.3f}"

occ6, off6, sc6 = build(M)
log("H6  (all 6 sheet views, 2 contradictory pairs):", fmt(sc6), "voxels", int(occ6.sum()))

# 1. one authority per direction: front beats back; pick the side view that agrees better
_, _, a = build({0: M[0], 45: M[45], 315: M[315], 90: M[90]}, fine=False)
_, _, b = build({0: M[0], 45: M[45], 315: M[315], 270: M[270]}, fine=False)
side = 90 if a[90] >= b[270] else 270
V = {0: M[0], 45: M[45], 315: M[315], side: M[side]}
occ, off, sc = build(V)
log(f"H4  (0, 45, 315, {side}; dropped 180 and {360 - side if side == 90 else 90}):", fmt(sc), "voxels", int(occ.sum()))
np.savez_compressed(os.path.join(od, "hull_H4.npz"), occ=occ, step=2, off=json.dumps({str(k): v for k, v in off.items()}))

# 2. new directions, generated from a clay render of the current proxy, validated before use
prompt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "newview.txt")).read()
for tgt in [30, 60, 120, 150]:
    guide, sil = clay(g, occ, tgt, off=0.0)
    save(guide, f"clay_{tgt}.png")
    Image.fromarray(guide).resize((1024, 1024), Image.LANCZOS).save(os.path.join(od, f"clay_{tgt}_1024.png"))
    near = min(RGB, key=lambda t: abs(((t - tgt + 180) % 360) - 180))
    Image.fromarray(RGB[near]).save(os.path.join(od, f"near_{tgt}.png"))
    best = None
    for attempt in range(2):
        gp = os.path.join(od, f"gen_{tgt}_{attempt}.png")
        log(f"view {tgt}: generating (attempt {attempt + 1}, nearest sheet view {near})")
        generate(prompt.format(angle=tgt), [os.path.join(od, f"clay_{tgt}_1024.png"), ref,
                                            os.path.join(od, f"near_{tgt}.png")], gp)
        gen = np.asarray(Image.open(gp).convert("RGB").resize((512, 512), Image.LANCZOS))
        m, rgb = canonicalize(mask_from_bg(gen), gen)
        # horizontal offset: best fit inside the proxy silhouette
        cand = []
        for d in range(-20, 21, 2):
            mm = np.roll(m, d, axis=1)
            outside = (mm & ~sil).sum() / max(mm.sum(), 1)
            cand.append((outside, -iou(mm, sil), d))
        outside, neg_iou, d = min(cand)
        m, rgb = np.roll(m, d, axis=1), np.roll(rgb, d, axis=1)
        trial = dict(V); trial[tgt] = m
        occ_t = carve(g, trial, off)
        sc_t = reprojection_scores(g, occ_t, trial, off)
        carved = 1 - occ_t.sum() / occ.sum()
        log(f"view {tgt}: outside proxy {outside:.1%}, IoU with proxy {-neg_iou:.3f}, would carve {carved:.1%}, "
            f"M3 after {fmt(sc_t)}")
        ok = outside < 0.04 and min(sc_t.values()) > 0.9
        if best is None or min(sc_t.values()) > best[0]:
            best = (min(sc_t.values()), m, rgb, occ_t, sc_t, ok)
        if ok:
            break
    if best[5]:
        V[tgt] = best[1]; RGB[tgt] = best[2]; occ = best[3]
        save(best[2], f"view_{tgt}.png")
        log(f"view {tgt}: ACCEPTED -> voxels {int(occ.sum())}")
    else:
        log(f"view {tgt}: REJECTED (inconsistent with the other witnesses)")

sc = reprojection_scores(g, occ, V, off)
log("FINAL hull views", sorted(V), fmt(sc), "voxels", int(occ.sum()), f"({occ.sum() / occ6.sum():.1%} of H6)")
np.savez_compressed(os.path.join(od, "hull_final.npz"), occ=occ, step=2,
                    off=json.dumps({str(k): v for k, v in off.items()}), views=json.dumps(sorted(V)))
# previews: H6 vs final from the side, 3/4 and top
rows = []
for name, o in [("H6", occ6), ("final", occ)]:
    rows.append(np.concatenate([clay(g, o, a, e)[0] for a, e in [(0, 0), (60, 0), (90, 0), (0, 89)]], 1))
save(np.concatenate(rows, 0), "hull_compare.png")
