import sys
from mf.sheet import load_cells
from mf.segment import mask_from_bg, canonicalize
from mf.metrics import m1_mirror, m2_rows

cells = load_cells(sys.argv[1])
masks = {az: mask_from_bg(c) for az, c in cells.items()}
if '--raw' not in sys.argv:
    masks = {az: canonicalize(m)[0] for az, m in masks.items()}
for pair, (v, s) in m1_mirror(masks).items():
    print(f"M1 mirror {pair:>8}: IoU {v:.3f} (shift {s}px)")
ext, m2 = m2_rows(masks)
for az, e in ext.items():
    print(f"   view {az:>3}: rows {e[0]}..{e[1]} (h={e[1]-e[0]})")
print(f"M2 row consistency: {m2:.3f}")
