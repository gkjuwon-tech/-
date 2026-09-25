"""법선 → 깊이 적분 (스크린드 Poisson, 뼈대 깊이에 앵커)."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def integrate_normals(n_cam, mask, z_anchor, f, anchor=1e-4, jump_rel=0.01, nz_min=0.1):
    """n_cam: (H,W,3) 카메라 좌표 법선(카메라 쪽 = -z), mask: bool, z_anchor: 뼈대 깊이 (H,W).

    국소 정사영 근사: dz/du = -(z/f)·nx/nz. 뼈대에서 깊이가 크게 끊기는 이웃은 제약에서 뺀다.
    반환: 적분된 깊이 (H,W), 마스크 밖은 z_anchor 그대로.
    """
    nz = np.minimum(n_cam[..., 2], -nz_min)
    gu = -(z_anchor / f) * n_cam[..., 0] / nz
    gv = -(z_anchor / f) * n_cam[..., 1] / nz
    jump_u = np.abs(z_anchor[:, 1:] - z_anchor[:, :-1]) > jump_rel * z_anchor[:, 1:]
    jump_v = np.abs(z_anchor[1:] - z_anchor[:-1]) > jump_rel * z_anchor[1:]
    idx = -np.ones(mask.shape, int)
    idx[mask] = np.arange(mask.sum())
    N = int(mask.sum())
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for a_idx, b_idx, g, ok in (
        (idx[:, :-1], idx[:, 1:], (gu[:, :-1] + gu[:, 1:]) / 2, mask[:, 1:] & mask[:, :-1] & ~jump_u),
        (idx[:-1], idx[1:], (gv[:-1] + gv[1:]) / 2, mask[1:] & mask[:-1] & ~jump_v),
    ):
        a, b, gg = a_idx[ok], b_idx[ok], g[ok]
        k = len(a)
        rows += [np.arange(r, r + k)] * 2
        cols += [a, b]
        vals += [-np.ones(k), np.ones(k)]
        rhs.append(gg)
        r += k
    lam = np.sqrt(anchor)
    rows.append(np.arange(r, r + N))
    cols.append(np.arange(N))
    vals.append(np.full(N, lam))
    rhs.append(lam * z_anchor[mask])
    r += N
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, N))
    b = np.concatenate(rhs)
    z, _ = spla.cg((A.T @ A).tocsc(), A.T @ b, x0=z_anchor[mask], rtol=1e-8, maxiter=3000)
    out = z_anchor.copy()
    out[mask] = z
    return out
