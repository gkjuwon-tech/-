"""Build a calibrated visual hull from a turnaround sheet.
usage: python run_hull.py SHEET OUT_PREFIX [--angles 0,30,90,180,270,330]"""
import sys, json, numpy as np
from PIL import Image
from mf.sheet import load_cells, AZIMUTHS
from mf.segment import mask_from_bg, canonicalize
from mf.hull import Grid, carve, reprojection_scores, calibrate_offsets

sheet, out = sys.argv[1], sys.argv[2]
angles = AZIMUTHS
if "--angles" in sys.argv:
    angles = [int(a) for a in sys.argv[sys.argv.index("--angles") + 1].split(",")]
cells = load_cells(sheet)
masks, rgbs = {}, {}
for nominal, actual in zip(AZIMUTHS, angles):
    m, rgb = canonicalize(mask_from_bg(cells[nominal]), cells[nominal])
    masks[actual], rgbs[actual] = m, rgb
off = calibrate_offsets(Grid(step=4), masks, search=12, step=2)
g = Grid(step=2)
occ = carve(g, masks, off)
sc = reprojection_scores(g, occ, masks, off)
print("offsets", off)
print("M3 reprojection IoU", {t: round(v, 3) for t, v in sc.items()}, "min", round(min(sc.values()), 3))
np.savez_compressed(out + ".npz", occ=occ, step=2, off=json.dumps({str(k): v for k, v in off.items()}))
for t in masks:
    Image.fromarray(rgbs[t]).save(f"{out}_canon_{t}.png")
    Image.fromarray(masks[t].astype(np.uint8) * 255).save(f"{out}_mask_{t}.png")
