"""템플릿에 입력 이미지를 base64로 넣어서 Kaggle 커널 폴더를 만든다.

사용법:
    python kaggle/build_kernel.py --image renders/bunny_front_rgba.png --user <kaggle_user> --out build/kernel \
        --models Lykon/dreamshaper-xl-1-0 stabilityai/stable-diffusion-xl-base-1.0
    KAGGLE_API_TOKEN=... kaggle kernels push -p build/kernel
"""
import argparse
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="renders/bunny_front_rgba.png")
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", default="fakelight-mvadapter-dreamshaper")
    ap.add_argument("--models", nargs="+", default=["Lykon/dreamshaper-xl-1-0"],
                    help="fp16 variant가 있는 diffusers 형식 SDXL 모델들 (순서대로 실행)")
    ap.add_argument("--wheel-kernel", default=None,
                    help="nvdiffrast wheel을 빌드해둔 커널 (예: user/mvadapter-env)")
    ap.add_argument("--out", default="build/kernel")
    args = ap.parse_args()

    tpl = open(os.path.join(HERE, "mvadapter_i2mv_template.py"), encoding="utf-8").read()
    b64 = base64.b64encode(open(args.image, "rb").read()).decode()
    code = tpl.replace("__INPUT_PNG_B64__", b64).replace("__BASE_MODELS__", json.dumps(args.models))

    os.makedirs(args.out, exist_ok=True)
    code_file = f"{args.slug}.py"
    with open(os.path.join(args.out, code_file), "w", encoding="utf-8") as f:
        f.write(code)
    meta = {
        "id": f"{args.user}/{args.slug}",
        "title": args.slug,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [args.wheel_kernel] if args.wheel_kernel else [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    with open(os.path.join(args.out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"built {args.out}/{code_file} ({len(code) // 1024} KB)")


if __name__ == "__main__":
    main()
