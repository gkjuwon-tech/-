#!/bin/sh
# Sequential experiment queue for the user's stage2 (CPU). Scores with the repo's own evaluators.
T=/home/user/gkjuwon-tech/3d
D=/home/user/-/meshforge/threed/data
cd $T
score() {  # name
  echo "=== SCORE $1" ; 
  python3 tools/eval_hull.py --gt-mesh $D/$1/gt_mesh.ply --views $D/$1/views --recon $D/$1/recon/mesh.ply 2>&1 | grep -iE "chamfer|F@|f-score|precision|recall|containment|volume" | head -12
  python3 tools/eval_surface.py --gt-mesh $D/$1/gt_mesh.ply --views $D/$1/views --cache $D/$1/gt_samples.npz --regions none --recon $D/$1/recon/mesh.ply 2>&1 | tail -12
}
stage() {  # name
  echo "=== STAGE2 $1 $(date +%H:%M:%S)"
  python3 stage2.py --name $1 --data $D --stop-after fuse --workers 3 > $D/../$1_stage2.out 2>&1 || { echo "FAILED $1"; tail -5 $D/$1/recon/stage2.log; return 1; }
  echo "=== DONE $1 $(date +%H:%M:%S)"; score $1
}
while pgrep -f "stage2.py --name sdragon14" > /dev/null; do sleep 20; done
echo "=== DONE sdragon14 $(date +%H:%M:%S)"; score sdragon14
stage s6_512_clean
stage s6_512_hard
stage s6_512_hard_fixed
python3 /home/user/-/meshforge/threed/consensus_refine.py $D/s6_512_hard_fixed $D/s6_512_hard_fixed/recon/mesh.ply --out $D/s6_512_hard_cons
stage s6_512_hard_cons
echo "=== QUEUE FINISHED $(date +%H:%M:%S)"
