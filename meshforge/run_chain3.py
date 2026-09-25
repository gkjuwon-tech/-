"""Chain v3: projections are only used where two independent images agree (two-witness rule).
The generator never overrides certified pixels; uncertain regions are left for it to fill.
usage: python run_chain3.py SHEET REF HULL_NPZ OUTDIR"""
import json, os, sys, time
import numpy as np
from PIL import Image
from mf.sheet import load_cells, AZIMUTHS
from mf.segment import mask_from_bg, canonicalize
from mf.hull import Grid
from mf.propagate import propagate
from mf.composite import align_to_guide, restore_and_blend, certify, fragments
from mf.fidelity import preservation
from mf.imagegen import generate

sheet, ref, hull_npz, od = sys.argv[1:5]
os.makedirs(od, exist_ok=True)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
save = lambda arr, name: Image.fromarray(arr).save(os.path.join(od, name))

d = np.load(hull_npz)
occ = d["occ"]; off = {int(k): v for k, v in json.loads(str(d["off"])).items()}
g = Grid(step=int(d["step"]))
cells = load_cells(sheet)
W, WM = {}, {}
for t in AZIMUTHS:
    WM[t], W[t] = canonicalize(mask_from_bg(cells[t]), cells[t])

FILL = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "fill.txt")).read()
final = {0: W[0]}
for tgt in [45, 315, 90, 270, 180]:
    per = []
    for s, img in final.items():
        gi, ki, sil = propagate(g, occ, off, {s: img}, tgt)
        per.append((gi, ki))
    guide, cert, known_any = certify(per, W[tgt], WM[tgt])
    canvas = np.full_like(guide, 128)
    canvas[cert] = guide[cert]
    canvas[(sil | WM[tgt]) & ~cert] = (255, 0, 255)
    save(canvas, f"guide_{tgt}.png")
    Image.fromarray(canvas).resize((1024, 1024), Image.NEAREST).save(os.path.join(od, f"guide_{tgt}_1024.png"))
    obj = (sil | WM[tgt]).sum()
    log(f"view {tgt}: projected {known_any.sum() / obj:.1%} of object, certified {cert.sum() / obj:.1%} "
        f"(rejected {(known_any & ~cert).sum() / obj:.1%}), generating...")
    gp = os.path.join(od, f"gen_{tgt}.png")
    generate(FILL.format(angle=tgt), [os.path.join(od, f"guide_{tgt}_1024.png"), ref], gp)
    gen = np.asarray(Image.open(gp).convert("RGB").resize((512, 512), Image.LANCZOS))
    gen, shift = align_to_guide(gen, guide, cert)
    raw = preservation(guide, gen, cert)["edge_f"]
    comp, _ = restore_and_blend(gen, guide, cert)
    final[tgt] = comp
    save(comp, f"final_{tgt}.png")
    log(f"view {tgt}: generator kept certified edges {raw:.3f} -> after restore "
        f"{preservation(guide, comp, cert)['edge_f']:.3f}; fragments {fragments(mask_from_bg(comp))}")

def pair_scores(views):
    out = {}
    for a in views:
        for b in views:
            if a == b or abs(((a - b + 180) % 360) - 180) > 90:
                continue
            gd, kn, _ = propagate(g, occ, off, {a: views[a]}, b)
            out[(a, b)] = preservation(gd, views[b], kn)["edge_f"]
    return out

sc_sheet, sc_chain = pair_scores(W), pair_scores(final)
for k in sc_sheet:
    log(f"  {k[0]:>3} -> {k[1]:>3}: sheet {sc_sheet[k]:.3f}  v3 {sc_chain[k]:.3f}")
log(f"neighbour edge-F MEAN sheet {np.mean(list(sc_sheet.values())):.3f} v3 {np.mean(list(sc_chain.values())):.3f} | "
    f"MIN sheet {min(sc_sheet.values()):.3f} v3 {min(sc_chain.values()):.3f}")
log("fragments per view  sheet:", {t: fragments(WM[t]) for t in AZIMUTHS},
    " v3:", {t: fragments(mask_from_bg(final[t])) for t in final})
top = np.concatenate([final[k] for k in [0, 45, 90]], 1)
bot = np.concatenate([final[k] for k in [180, 270, 315]], 1)
save(np.concatenate([top, bot], 0), "chain_sheet.png")
