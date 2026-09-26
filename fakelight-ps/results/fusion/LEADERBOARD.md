# 버니 정면 노멀 기록판 (합체)

입력: MV-Adapter + Juggernaut 정면 뷰. 채점: 정렬된 정답 노멀, 평균 각도 오차.

| 방법 | 오차 |
|---|---|
| IC-Light 회전 트릭 s1 → SDM-UniPS 1024px | 12.66° |
| Neural LightRig 시드 3개 (27장) → SDM-UniPS 1024px | 12.72° |
| IC-Light(rot s1 + v2 all) + Neural LightRig 평균 | 11.09° |
| **주파수 합체: 저주파 = 위 평균, 고주파 = IC-Light (σ=4px)** | **10.89°** |
| RoSE (ICLR 2026) | 16.70° |
| Marigold v1.1 | 16.16° |

주의: 합체 조합과 σ는 버니 정답을 보고 고른 것이라 과적합 가능성이 있다. DiLiGenT에서 고정된 레시피로 검증해야 한다.
