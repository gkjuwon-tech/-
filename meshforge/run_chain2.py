"""Full 6-view chain (v2: hull refined by each generated silhouette, confidence-gated restore): sheet -> calibrated hull -> projective propagation with generator fill,
restore + seam blending -> consistency report.
usage: python run_chain.py SHEET REF OUTDIR"""
import json, os, sys, time
import numpy as np
from PIL import Image
from mf.sheet import load_cells, AZIMUTHS
from mf.segment import mask_from_bg, canonicalize
from mf.hull import Grid, carve, reprojection_scores, calibrate_offsets, calibrate_angle
from mf.propagate import propagate
from mf.composite import align_to_guide, restore_and_blend, trusted_known
from mf.fidelity import preservation
from mf.imagegen import generate

sheet, ref, od = sys.argv[1:4]
os.makedirs(od, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
save = lambda arr, name: Image.fromarray(arr).save(os.path.join(od, name))

# 1. canonical views
cells = load_cells(sheet)
M, RGB = {}, {}
for t in AZIMUTHS:
    M[t], RGB[t] = canonicalize(mask_from_bg(cells[t]), cells[t])

# 2. calibration: cardinal views first, then the true angle of the two diagonal cells
g4 = Grid(step=4)
card = {t: M[t] for t in (0, 90, 180, 270)}
off = calibrate_offsets(g4, card, search=12, step=2)
v45, a45, d45 = calibrate_angle(g4, card, off, M[45], range(15, 76, 5))
v315, a315, d315 = calibrate_angle(g4, card, off, M[315], range(285, 346, 5))
log(f"diagonal cells: nominal 45 -> {a45} (IoU {v45:.3f}), nominal 315 -> {a315} (IoU {v315:.3f})")
ANG = {0: 0, 45: a45, 90: 90, 180: 180, 270: 270, 315: a315}
masks = {ANG[t]: M[t] for t in AZIMUTHS}
rgbs = {ANG[t]: RGB[t] for t in AZIMUTHS}
off = calibrate_offsets(g4, masks, search=8, step=2, rounds=1) | {}

# 3. hull (strict vs vote) on the fine grid
g = Grid(step=2)
res = {}
for name, mv in [("strict", None), ("vote5", len(masks) - 1)]:
    occ = carve(g, masks, off, min_votes=mv)
    sc = reprojection_scores(g, occ, masks, off)
    res[name] = (occ, sc)
    log(f"hull {name}: M3", {k: round(v, 3) for k, v in sc.items()}, "min", round(min(sc.values()), 3))
occ = res["strict"][0]
np.savez_compressed(os.path.join(od, "hull.npz"), occ=occ, step=2,
                    off=json.dumps({str(k): v for k, v in off.items()}), ang=json.dumps(ANG))

# 4. chain
order = [ANG[45], ANG[315], 90, 270, 180]
final = {0: rgbs[0]}
FILL = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "fill.txt")).read()
for tgt in order:
    guide, known, sil = propagate(g, occ, off, final, tgt)
    canvas = np.full_like(guide, 128)
    canvas[known] = guide[known]
    canvas[sil & ~known] = (255, 0, 255)
    save(canvas, f"guide_{tgt}.png")
    Image.fromarray(canvas).resize((1024, 1024), Image.NEAREST).save(os.path.join(od, f"guide_{tgt}_1024.png"))
    log(f"view {tgt}: known {known.sum() / max(sil.sum(), 1):.1%} of silhouette, generating...")
    gen_path = os.path.join(od, f"gen_{tgt}.png")
    generate(FILL.format(angle=tgt), [os.path.join(od, f"guide_{tgt}_1024.png"), ref], gen_path)
    gen = np.asarray(Image.open(gen_path).convert("RGB").resize((512, 512), Image.LANCZOS))
    gen_al, shift = align_to_guide(gen, guide, known)
    before = preservation(guide, gen_al, known)
    # generator silhouette becomes the constraint for this angle -> re-carve, re-propagate
    gmask = mask_from_bg(gen_al)
    masks[tgt] = gmask
    occ = carve(g, masks, off)
    guide, known, sil = propagate(g, occ, off, final, tgt)
    trust = trusted_known(gen_al, guide, known, gmask)
    comp, _ = restore_and_blend(gen_al, guide, trust)
    after = preservation(guide, comp, trust)
    log(f"view {tgt}: shift {shift}, raw keep edgeF {before['edge_f']:.3f}; trusted {trust.sum() / max(gmask.sum(), 1):.1%} "
        f"of object -> composited edgeF {after['edge_f']:.3f}")
    final[tgt] = comp
    save(comp, f"final_{tgt}.png")

# 5. consistency report: every ordered pair of neighbouring views, sheet vs chain
def pair_scores(views):
    out = {}
    keys = sorted(views)
    for i, a in enumerate(keys):
        for b in keys:
            if a == b:
                continue
            gd, kn, _ = propagate(g, occ, off, {a: views[a]}, b)
            if kn.sum() < 2000:
                continue
            out[(a, b)] = preservation(gd, views[b], kn)["edge_f"]
    return out

occ = carve(g, masks, off)
sc = reprojection_scores(g, occ, masks, off)
log("final hull M3", {k: round(v, 3) for k, v in sc.items()}, "min", round(min(sc.values()), 3))
s_sheet, s_chain = pair_scores(rgbs), pair_scores(final)
log("pairwise overlap edge-F (sheet -> chain):")
for k in s_sheet:
    log(f"  {k[0]:>3} -> {k[1]:>3}: {s_sheet[k]:.3f} -> {s_chain.get(k, float('nan')):.3f}")
log(f"MEAN sheet {np.mean(list(s_sheet.values())):.3f}  chain {np.mean(list(s_chain.values())):.3f}")
log(f"MIN  sheet {min(s_sheet.values()):.3f}  chain {min(s_chain.values()):.3f}")
# contact sheet of results
top = np.concatenate([final[k] for k in [0, ANG[45], 90]], 1)
bot = np.concatenate([final[k] for k in [180, 270, ANG[315]]], 1)
save(np.concatenate([top, bot], 0), "chain_sheet.png")
