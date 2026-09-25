"""Photometric stereo on generator-relit images.
Camera space: x right, y down, z toward the camera. Depth w grows toward the camera."""
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import lsqr


def lights18():
    """3 rings (25, 45, 65 deg off the view axis) x 6 azimuths; middle ring rotated 30 deg."""
    L, desc = [], []
    names = {0: "right", 60: "lower right", 120: "lower left", 180: "left", 240: "upper left", 300: "upper right",
             30: "right, slightly low", 90: "below", 150: "left, slightly low", 210: "left, slightly high",
             270: "above", 330: "right, slightly high"}
    for ring, th in enumerate((25, 45, 65)):
        for k in range(6):
            ph = k * 60 + (30 if ring == 1 else 0)
            t, p = np.deg2rad(th), np.deg2rad(ph)
            L.append([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            desc.append(f"from the {names[ph]}, {th} degrees away from the camera direction")
    return np.array(L), desc


def normals_from_depth(depth, mask, sigma=2.0):
    d = np.where(mask, depth, np.nan)
    fill = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    d = d[fill[0], fill[1]]
    d = ndimage.gaussian_filter(d, sigma)
    wy, wx = np.gradient(d)
    n = np.stack([-wx, -wy, np.ones_like(d)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def shade(n, mask, l, albedo=0.85, ambient=0.06):
    s = albedo * np.clip(n @ l, 0, None) + ambient
    img = np.full(mask.shape + (3,), 128, np.uint8)
    img[mask] = (np.clip(s[mask], 0, 1)[:, None] * 255).astype(np.uint8)
    return img


def to_linear(img):
    g = img.astype(np.float32).mean(-1) / 255.0 if img.ndim == 3 else img / 255.0
    return g ** 2.2


def solve(I, L, mask, drop_dark=3, drop_bright=2, iters=3, mu=0.3):
    """I: (K, H, W) linear intensities, L: (K, 3) nominal lights.
    Robust per-pixel least squares (darkest = shadows, brightest = highlights dropped),
    alternating with a regularised light refinement. Returns normals, albedo, residual, lights."""
    K = len(L)
    Ip = I[:, mask].T                                   # (P, K)
    order = np.argsort(Ip, axis=1)
    W = np.ones_like(Ip)
    rows = np.arange(len(Ip))[:, None]
    W[rows, order[:, :drop_dark]] = 0
    W[rows, order[:, K - drop_bright:]] = 0
    Lc = L.copy()
    for it in range(iters):
        A = np.einsum("pk,ki,kj->pij", W, Lc, Lc) + 1e-6 * np.eye(3)
        rhs = np.einsum("pk,pk,ki->pi", W, Ip, Lc)
        B = np.linalg.solve(A, rhs[..., None])[..., 0]  # (P, 3) = albedo * normal
        if it == iters - 1:
            break
        # refine each light: min ||W (I_k - B l_k)||^2 + mu*|| l_k - s_k L0_k ||^2
        for k in range(K):
            w = W[:, k]
            Bw = B * w[:, None]
            s = max((Bw @ L[k] * Ip[:, k]).sum() / max(((Bw @ L[k]) ** 2).sum(), 1e-9), 1e-3)
            M = Bw.T @ B + mu * len(B) * 1e-3 * np.eye(3)
            Lc[k] = np.linalg.solve(M, Bw.T @ Ip[:, k] + mu * len(B) * 1e-3 * s * L[k])
    rho = np.linalg.norm(B, axis=1)
    n = B / np.maximum(rho[:, None], 1e-6)
    n[n[:, 2] < 0.05] *= np.array([1, 1, -1])           # never face away from the camera
    res = np.sqrt((W * (Ip - B @ Lc.T) ** 2).sum(1) / np.maximum(W.sum(1), 1)) / np.maximum(Ip.mean(1), 1e-3)
    N = np.zeros(mask.shape + (3,), np.float32); N[mask] = n
    R = np.zeros(mask.shape, np.float32); R[mask] = rho
    E = np.zeros(mask.shape, np.float32); E[mask] = res
    return N, R, E, Lc


def integrate(N, mask, anchor=None, lam=0.02, min_nz=0.15, weights=None):
    """Least-squares depth from normals (w grows toward camera), weakly anchored to `anchor`."""
    H, Wd = mask.shape
    idx = -np.ones(mask.shape, int); idx[mask] = np.arange(mask.sum())
    nz = np.maximum(N[..., 2], min_nz)
    p, q = -N[..., 0] / nz, -N[..., 1] / nz              # dw/dx, dw/dy
    wgt = np.ones(mask.shape) if weights is None else weights
    rows, cols, vals, b = [], [], [], []
    r = 0
    for dy, dx, g in ((0, 1, p), (1, 0, q)):
        a = mask[:H - dy, :Wd - dx] & mask[dy:, dx:]
        i0 = idx[:H - dy, :Wd - dx][a]; i1 = idx[dy:, dx:][a]
        gv = 0.5 * (g[:H - dy, :Wd - dx][a] + g[dy:, dx:][a])
        ww = np.sqrt(np.minimum(wgt[:H - dy, :Wd - dx][a], wgt[dy:, dx:][a]))
        n = len(i0)
        rr = np.arange(r, r + n)
        rows += [rr, rr]; cols += [i1, i0]; vals += [ww, -ww]; b.append(ww * gv); r += n
    if anchor is not None:
        n = int(mask.sum())
        rr = np.arange(r, r + n)
        rows.append(rr); cols.append(np.arange(n)); vals.append(np.full(n, lam)); b.append(lam * anchor[mask]); r += n
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, mask.sum()))
    z = lsqr(A, np.concatenate(b), atol=1e-8, btol=1e-8, iter_lim=4000)[0]
    D = np.full(mask.shape, np.nan, np.float32); D[mask] = z
    return D


def masked_blur(F, mask, sigma):
    m = mask.astype(np.float32)
    den = ndimage.gaussian_filter(m, sigma)
    out = np.stack([ndimage.gaussian_filter(F[..., c] * m, sigma) for c in range(F.shape[-1])], -1)
    return out / np.maximum(den[..., None], 1e-6)


def fuse_frequencies(N_ps, N_base, mask, sigma=8.0):
    """Low frequencies from the geometric proxy, high-frequency detail from photometric stereo.
    Generator relighting is locally faithful (scales, spikes) but globally biased."""
    detail = N_ps - masked_blur(N_ps, mask, sigma)
    N = masked_blur(N_base, mask, sigma) + detail
    N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-6)
    N[~mask] = 0
    return N
