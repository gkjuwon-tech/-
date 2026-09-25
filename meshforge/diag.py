import sys, numpy as np
from PIL import Image
from mf.sheet import load_cells
from mf.segment import mask_from_bg, canonicalize
from mf.metrics import best_hshift_iou
cells = load_cells(sys.argv[1]); masks = {a: canonicalize(mask_from_bg(c))[0] for a, c in cells.items()}
tiles = []
for a, b in [(0, 180), (90, 270)]:
    mb = masks[b][:, ::-1]; v, s = best_hshift_iou(masks[a], mb); mb = np.roll(mb, s, axis=1)
    t = np.full(masks[a].shape + (3,), 40, np.uint8)
    t[masks[a] & mb] = (220, 220, 220); t[masks[a] & ~mb] = (255, 60, 60); t[~masks[a] & mb] = (60, 200, 255)
    tiles.append(t)
Image.fromarray(np.concatenate(tiles, 1)).save(sys.argv[2])
