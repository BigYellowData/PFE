#!/bin/bash
OUTDIR=/mnt/c/Users/nadir/Desktop/PFE-NetworkSlicing/work/simu5g/loss/resultat
mkdir -p "$OUTDIR" logs

echo "=== Batch 1 : NormalLoad + LowTrafficNight + FifaWorldCup ==="
./run_sim.sh NormalLoad            "$OUTDIR" > logs/NormalLoad.log 2>&1 &
./run_sim.sh LowTrafficNight       "$OUTDIR" > logs/LowTrafficNight.log 2>&1 &
./run_sim.sh FifaWorldCup_Commerce "$OUTDIR" > logs/FifaWorldCup.log 2>&1 &
wait && echo "=== Batch 1 terminé ==="

echo "=== Batch 2 : KddiOutage + GlobalSaturation + SLABoundary ==="
./run_sim.sh KddiOutage_Storm   "$OUTDIR" > logs/KddiOutage.log 2>&1 &
./run_sim.sh GlobalSaturation   "$OUTDIR" > logs/GlobalSaturation.log 2>&1 &
./run_sim.sh SLABoundary_URLLC  "$OUTDIR" > logs/SLABoundary.log 2>&1 &
wait && echo "=== Batch 2 terminé ==="

echo "=== Batch 3 : ModerateLoad_eMBB + ModerateLoad_URLLC + HetLoad_A ==="
./run_sim.sh ModerateLoad_eMBB    "$OUTDIR" > logs/ModerateLoad_eMBB.log 2>&1 &
./run_sim.sh ModerateLoad_URLLC   "$OUTDIR" > logs/ModerateLoad_URLLC.log 2>&1 &
./run_sim.sh HetLoad_Asymmetric_A "$OUTDIR" > logs/HetLoad_A.log 2>&1 &
wait && echo "=== Batch 3 terminé ==="

echo "=== Batch 4 : HetLoad_B + HetLoad_C + OverloadeMBB ==="
./run_sim.sh HetLoad_Asymmetric_B  "$OUTDIR" > logs/HetLoad_B.log 2>&1 &
./run_sim.sh HetLoad_Asymmetric_C  "$OUTDIR" > logs/HetLoad_C.log 2>&1 &
./run_sim.sh OverloadeMBB_Commerce "$OUTDIR" > logs/OverloadeMBB.log 2>&1 &
wait && echo "=== Batch 4 terminé ==="

echo "=== Toutes les simulations terminées ==="
echo "Résultats dans : $OUTDIR"
