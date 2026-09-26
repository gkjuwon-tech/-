# 🔦 짭광도 스테레오 (Fake-Light Photometric Stereo)

> *"빛은 가짜지만, 노멀은 진짜다"*

사진 1장으로 무료 고품질 3D를 만드는 파이프라인 실험장.

```
사진 1장 → MV-Adapter + DreamShaper (멀티뷰) → 조명 시퀀스 생성 → 광도 스테레오 (SDM-UniPS)
        → 앙상블 + 자기검증 → 멀티뷰 융합 → 메쉬
```

## 현재 단계: 0단계, 버니로 MV-Adapter까지

| 파일 | 내용 |
|---|---|
| `assets/stanford-bunny.obj` | 스탠포드 버니 ([common-3d-test-models](https://github.com/alecjacobson/common-3d-test-models)) |
| `scripts/render_front.py` | numpy 정사영 래스터라이저. 정면 RGB/RGBA, 마스크, 정답 노멀맵 저장 |
| `renders/` | 768px 정면 렌더 (yaw 80°) |
| `kaggle/mvadapter_dreamshaper_template.py` | Kaggle T4용 MV-Adapter I2MV + DreamShaper XL 스크립트 |
| `kaggle/build_kernel.py` | 입력 이미지를 끼워 넣어 Kaggle 커널 폴더 생성 |

### 다시 돌리기

```bash
pip install trimesh numpy pillow scipy kaggle
python scripts/render_front.py --res 768 --yaw 80 --out renders

python kaggle/build_kernel.py --image renders/bunny_front_rgba.png --user <kaggle_user> --out build/kernel
export KAGGLE_API_TOKEN=...   # 토큰은 절대 커밋하지 말 것
kaggle kernels push -p build/kernel
kaggle kernels output <kaggle_user>/fakelight-mvadapter-dreamshaper -p results/mvadapter
```

정답 노멀(`bunny_front_normal_gt.npy`)은 용량 때문에 git에서 제외했다. 렌더 스크립트를 다시 돌리면 생긴다.
