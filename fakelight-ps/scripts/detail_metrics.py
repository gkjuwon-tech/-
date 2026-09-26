"""노멀맵의 표면 디테일(고주파) 점수.

hp(n) = n - blur(n, σ) 로 큰 모양을 빼고, 정답의 고주파와 비교한다.
  - 상관계수: 주름 위치/모양이 정답과 얼마나 일치하나 (1이 완벽)
  - 에너지 비: 디테일의 세기 (1 = 정답과 같은 세기, <1 뭉개짐, >1 과장)
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def masked_blur(n, m, s):
    w = gaussian_filter(m.astype(float), s) + 1e-6
    b = np.stack([gaussian_filter(n[..., k] * m, s) / w for k in range(3)], 2)
    return b / (np.linalg.norm(b, axis=2, keepdims=True) + 1e-12)


def detail_scores(pred, gt, m, s=4, erode=6):
    from scipy.ndimage import binary_erosion
    mi = binary_erosion(m, iterations=erode)  # 실루엣 경계의 블러 효과 제외
    hp = lambda n: n - masked_blur(n, m, s)
    h, g = hp(pred)[mi], hp(gt)[mi]
    corr = (h * g).sum() / np.sqrt((h ** 2).sum() * (g ** 2).sum())
    energy = np.sqrt((h ** 2).mean() / (g ** 2).mean())
    return float(corr), float(energy)
