import json, sys, numpy as np
from PIL import Image
from mf.segment import mask_from_bg, canonicalize
from mf.sheet import load_cells
from mf.hull import Grid, render
from mf.photometric import normals_from_depth, integrate, fuse_frequencies
from mf.composite import agree
from mf.fidelity import edge_fscore
from mf.warp import warp

od = 'out/dragon/ps'
d = np.load(f'{od}/front_depth.npz'); mask = d['mask']; Nps = d['normals']
h = np.load(f'{od}/hull.npz'); occ = h['occ']; off = {int(k): v for k, v in json.loads(str(h['off'])).items()}
g = Grid(step=2)
sil, hdepth, _ = render(g, occ, 0, off[0])
Nh = normals_from_depth(hdepth, sil, sigma=2.5)
refimg = np.asarray(Image.open('out/dragon/dragon_ref.png').convert('RGB'))
m0, r0 = canonicalize(mask_from_bg(refimg), refimg)
cells = load_cells('out/dragon/sheet_p0.png')
W = {45: canonicalize(mask_from_bg(cells[45]), cells[45])}
w30 = np.asarray(Image.open('out/dragon/geom/view_30.png').convert('RGB')); W[30] = (mask_from_bg(w30), w30)
res = {}
for sigma in (6, 12, 24):
    N = fuse_frequencies(Nps, Nh, mask, sigma)
    D = integrate(N, mask, anchor=hdepth, lam=0.05)
    res[sigma] = (N, D)
    line = [f"sigma {sigma:>2}: |D-hull| median {np.nanmedian(np.abs(D - hdepth)[mask]):.1f}px"]
    for tgt in (30, 45):
        tm, trgb = W[tgt]
        for name, dep in (("hull", hdepth), ("fused", D)):
            if name == "hull" and sigma != 6:
                continue
            img, cov, _ = warp(r0, np.nan_to_num(dep), mask, 0, tgt, off[0], off[tgt])
            line.append(f"0->{tgt} {name}: on-object {(cov & tm).sum() / cov.sum():.1%} agree {(cov & tm & agree(img, trgb)).sum() / cov.sum():.1%} edgeF {edge_fscore(img, trgb, cov & tm):.3f}")
    print(" | ".join(line), flush=True)
N, D = res[12]
Image.fromarray(((N * 0.5 + 0.5) * 255 * mask[..., None]).astype(np.uint8)).save(f'{od}/normals_fused.png')
np.savez_compressed(f'{od}/front_depth_fused.npz', depth=D, normals=N, mask=mask)
