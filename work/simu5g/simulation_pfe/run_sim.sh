#!/bin/bash
# Usage : ./run_sim.sh <ConfigName> [ResultDir]
# Exemple : ./run_sim.sh QuickTest_ModerateLoad_eMBB
# Exemple : ./run_sim.sh NormalLoad /loss/resultat

CONFIG=${1:?"Usage: $0 <ConfigName> [ResultDir]"}
RESULTDIR=${2:-results}

mkdir -p "$RESULTDIR"

opp_run -u Cmdenv -c "$CONFIG" \
  --result-dir="$RESULTDIR" \
  -n .:/home/nadir/opp_env_workspace/simu5g-1.4.2/src:/home/nadir/opp_env_workspace/inet-4.6.0/src \
  -l /home/nadir/opp_env_workspace/simu5g-1.4.2/src/simu5g \
  -l /home/nadir/opp_env_workspace/inet-4.6.0/src/INET \
  --image-path=/home/nadir/opp_env_workspace/simu5g-1.4.2/images \
  omnetpp_hetnet.ini
