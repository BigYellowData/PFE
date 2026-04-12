#!/bin/bash
# run_comparison.sh — Lance 3 simulations identiques pour comparaison equitable
#
# Mode 1 : Baseline     — aucune reallocation (DEFAULT_RBS fixes)
# Mode 2 : MILP periodique — reallocation toutes les 15s sans LSTM
# Mode 3 : LSTM + MILP  — reallocation adaptative avec prediction LSTM
#
# Meme config + meme seed pour les 3 runs → comparaison juste.
#
# Usage :
#   ./run_comparison.sh [ConfigName] [SeedNumber]
#   ./run_comparison.sh GlobalSaturation 0
#   ./run_comparison.sh RampUp_Commerce 42

CONFIG=${1:-"GlobalSaturation"}
SEED=${2:-0}
SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="$SIM_DIR/comparison_results/${CONFIG}_seed${SEED}"
METRICS_F="$SIM_DIR/metrics_live.csv"
CONFIG_F="$SIM_DIR/rb_config.json"

# Chemins OMNeT++
OPP_LIBS="-n .:/home/nadir/opp_env_workspace/simu5g-1.4.2/src:/home/nadir/opp_env_workspace/inet-4.6.0/src \
          -l /home/nadir/opp_env_workspace/simu5g-1.4.2/src/simu5g \
          -l /home/nadir/opp_env_workspace/inet-4.6.0/src/INET \
          --image-path=/home/nadir/opp_env_workspace/simu5g-1.4.2/images"

SIM_TIME=300   # duree simulation en secondes (doit correspondre a omnetpp_hetnet.ini)

echo "========================================================"
echo "  Comparaison : $CONFIG  (seed=$SEED)"
echo "  Resultats   : $RESULTS_BASE"
echo "========================================================"
mkdir -p "$RESULTS_BASE"

# ── Fonction utilitaire ───────────────────────────────────────────────────────
run_one_mode() {
    local MODE=$1
    local CTRL_CMD=$2  # commande du controleur, ou "" pour baseline
    local OUT_DIR="$RESULTS_BASE/$MODE"
    mkdir -p "$OUT_DIR"

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "  Mode : $MODE"
    echo "────────────────────────────────────────────────────────"

    # Reset fichiers live
    echo "simtime_s,gnb,slice,avg_latency_ms,throughput_mbps,pkt_count,pkt_loss_pct" > "$METRICS_F"
    echo "{}" > "$CONFIG_F"

    # Lancer le controleur en arriere-plan (si pas baseline)
    CTRL_PID=""
    if [ -n "$CTRL_CMD" ]; then
        eval "$CTRL_CMD" > "$OUT_DIR/controller.log" 2>&1 &
        CTRL_PID=$!
        echo "  [ctrl] PID=$CTRL_PID  log=$OUT_DIR/controller.log"
        sleep 2  # laisser le controleur ecrire rb_config.json initial
    else
        # Baseline : ecrire config par defaut et ne jamais la changer
        python3 -c "
import json
DEFAULT = {
    'Macro':     {'eMBB': 20, 'URLLC': 17, 'mMTC': 13},
    'Commerce':  {'eMBB': 21, 'URLLC': 11, 'mMTC':  3},
    'Industrie': {'eMBB':  6, 'URLLC': 12, 'mMTC':  7},
}
with open('$CONFIG_F', 'w') as f:
    json.dump(DEFAULT, f, indent=2)
print('[baseline] rb_config.json fixe (DEFAULT_RBS)')
"
    fi

    # Lancer la simulation OMNeT++
    echo "  [sim]  opp_run -c $CONFIG --seed-set=$SEED ..."
    opp_run -u Cmdenv -c "$CONFIG" \
        --seed-set=$SEED \
        --result-dir="$OUT_DIR/sim_results" \
        $OPP_LIBS \
        --sim-time-limit=300s omnetpp_hetnet.ini > "$OUT_DIR/sim.log" 2>&1
    SIM_EXIT=$?

    # Arreter le controleur
    if [ -n "$CTRL_PID" ]; then
        kill "$CTRL_PID" 2>/dev/null
        wait "$CTRL_PID" 2>/dev/null
        echo "  [ctrl] arrete (PID=$CTRL_PID)"
    fi

    # Copier les metriques live
    cp "$METRICS_F" "$OUT_DIR/metrics_live.csv" 2>/dev/null

    if [ $SIM_EXIT -eq 0 ]; then
        echo "  [ok]   Simulation terminee -> $OUT_DIR"
    else
        echo "  [ERR]  Simulation echouee (exit=$SIM_EXIT) — voir $OUT_DIR/sim.log"
    fi
}

# ── Run 1 : Baseline ──────────────────────────────────────────────────────────
run_one_mode "baseline" ""

# ── Run 2 : MILP periodique ───────────────────────────────────────────────────
run_one_mode "periodic_milp" \
    "python3 $SIM_DIR/periodic_controller.py --sim-dir $SIM_DIR --control-period 15"

# ── Run 3 : LSTM + MILP ───────────────────────────────────────────────────────
run_one_mode "lstm_milp" \
    "python3 $SIM_DIR/closed_loop_controller.py --sim-dir $SIM_DIR --control-period 15"

# ── Analyse comparative ───────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Analyse comparative..."
echo "========================================================"
python3 "$SIM_DIR/compare_results.py" "$RESULTS_BASE"

echo ""
echo "Done. Resultats dans : $RESULTS_BASE"
