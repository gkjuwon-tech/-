#!/bin/sh
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
cd /home/user/gkjuwon-tech/3d
for n in sdragon14 sdragon6 sdragon6_hard sdragon6_hard_fixed; do
  echo "=== PUSH $n $(date +%H:%M:%S)"
  python3 tools/kaggle_stage2.py push $n --gpu 2>&1 | tail -4
done
echo "=== ALL PUSHED $(date +%H:%M:%S)"
