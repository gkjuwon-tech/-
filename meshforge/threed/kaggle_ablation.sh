#!/bin/sh
# push -> wait -> pull -> score, at most 2 GPU sessions at a time
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
cd /home/user/gkjuwon-tech/3d
st() { kaggle kernels status flyjw12/threed-stage2-$(echo $1 | tr _ -)-gpu 2>&1 | grep -oiE "complete|error|running|queued|cancel" | head -1; }
run_one() {
  n=$1
  python3 tools/kaggle_stage2.py push $n --gpu 2>&1 | grep -E "successfully|rror" | sed "s/^/[$n] /"
  sleep 60
  while true; do s=$(st $n); case "$s" in COMPLETE|complete|ERROR|error|CANCEL*) break;; esac; sleep 30; done
  echo "[$n] kernel $s"
  rm -rf data/$n/recon; python3 tools/kaggle_stage2.py pull $n --gpu > /dev/null 2>&1
  if [ -f data/$n/recon/mesh.ply ]; then
    echo "=== SCORE $n"; python3 tools/eval_surface.py --gt-mesh data/$n/gt_mesh.ply --views data/$n/views --cache data/$n/gt_samples.npz --regions none --recon data/$n/recon/mesh.ply 2>&1 | tail -1
  else echo "=== FAILED $n"; grep -m2 -E "Error" data/$n/recon/stage2.log; fi
}
run_one sdragon6_only_normals & run_one sdragon6_only_camera & wait
run_one sdragon6_only_mask
echo "=== ABLATION DONE"
