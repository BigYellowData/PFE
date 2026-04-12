"""
periodic_controller.py — MILP periodique sans LSTM.
Realloue les RBs toutes les 15s en se basant uniquement sur le
throughput et la latence observes (pas de prediction LSTM).

Utilise exactement le meme MILP que closed_loop_controller.py
pour une comparaison equitable.

Usage:
  python3 periodic_controller.py [--sim-dir /path/to/simulation_pfe]
"""

import os, sys, time, json, argparse, warnings
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
warnings.filterwarnings('ignore')

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--sim-dir', default=os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--control-period', type=float, default=15.0)
args = parser.parse_args()

SIM_DIR   = args.sim_dir
METRICS_F = os.path.join(SIM_DIR, 'metrics_live.csv')
CONFIG_F  = os.path.join(SIM_DIR, 'rb_config.json')
PERIOD    = args.control_period

print(f"[periodic] SIM_DIR   = {SIM_DIR}")
print(f"[periodic] METRICS_F = {METRICS_F}")
print(f"[periodic] CONFIG_F  = {CONFIG_F}")
print(f"[periodic] Period    = {PERIOD}s (MILP periodique, sans LSTM)")

# ── Topologie (identique a closed_loop_controller.py) ─────────────────────────
DEFAULT_RBS = {
    'Macro':     {'eMBB': 20, 'URLLC': 17, 'mMTC': 13},
    'Commerce':  {'eMBB': 21, 'URLLC': 11, 'mMTC':  3},
    'Industrie': {'eMBB':  6, 'URLLC': 12, 'mMTC':  7},
}
R_MAX    = {'Macro': 50, 'Commerce': 35, 'Industrie': 25}
R_PER_RB = {
    'Macro':     {'eMBB': 0.560, 'URLLC': 0.125, 'mMTC': 0.050},
    'Commerce':  {'eMBB': 0.229, 'URLLC': 0.097, 'mMTC': 0.050},
    'Industrie': {'eMBB': 0.400, 'URLLC': 0.195, 'mMTC': 0.050},
}
SLA = {
    'eMBB':  {'T_min': 10.0,    'L_max_ms':  60.0},
    'URLLC': {'T_min':  0.5,    'L_max_ms':  25.0},
    'mMTC':  {'T_min':  0.0003, 'L_max_ms': 300.0},
}
MIN_RBS = {
    'Macro':     {'eMBB': 5,  'URLLC': 10, 'mMTC': 1},
    'Commerce':  {'eMBB': 5,  'URLLC':  7, 'mMTC': 1},
    'Industrie': {'eMBB': 3,  'URLLC':  8, 'mMTC': 1},
}
OMEGA_T        = 0.6
OMEGA_P        = 0.4
SLICE_PRIORITY = {'eMBB': 1.0, 'URLLC': 3.0, 'mMTC': 0.5}
SLICES         = ['eMBB', 'URLLC', 'mMTC']
GNBS           = ['Macro', 'Commerce', 'Industrie']

SIGMOID_PARAMS = {
    ('Macro',     'eMBB'):  (4.68, 0.798),
    ('Macro',     'URLLC'): (2.91, 1.093),
    ('Macro',     'mMTC'):  (10.50, 1.245),
    ('Commerce',  'eMBB'):  (5.93, 1.404),
    ('Commerce',  'URLLC'): (15.27, 0.406),
    ('Commerce',  'mMTC'):  (10.51, 1.245),
    ('Industrie', 'eMBB'):  (1.33, 2.304),
    ('Industrie', 'URLLC'): (4.67, 1.307),
    ('Industrie', 'mMTC'):  (11.88, 1.444),
}

def sigmoid(rho, k, rho_c):
    return 1.0 / (1.0 + np.exp(-k * (rho - rho_c)))

def obs_rho(df_live, gnb, stype):
    """Estime rho depuis le throughput observe (pas de LSTM)."""
    sub = df_live[df_live['slice'] == stype].tail(5)
    if sub.empty:
        return 0.1
    tp  = float(sub['throughput_mbps'].mean())
    lat = float(sub['avg_latency_ms'].mean())
    cap = DEFAULT_RBS[gnb][stype] * R_PER_RB[gnb][stype]
    # Si latence > SLA → saturation probable
    if lat > SLA[stype]['L_max_ms']:
        return 2.0
    rho = float(np.clip(tp / max(cap, 1e-6), 0.0, 2.0))
    return rho

def milp_optimize(gnb, rho_dict):
    R_total  = R_MAX[gnb]
    min_rbs  = MIN_RBS[gnb]
    var_list = [(s, k) for s in SLICES for k in range(min_rbs[s], R_total + 1)]
    n_vars   = len(var_list)
    idx      = {v: i for i, v in enumerate(var_list)}

    cost_table = {}
    for stype in SLICES:
        rho_base = rho_dict.get(stype, 0.1)
        lam      = rho_base * DEFAULT_RBS[gnb][stype] * R_PER_RB[gnb][stype]
        k_s, rc  = SIGMOID_PARAMS[(gnb, stype)]
        T_min    = SLA[stype]['T_min']
        prio     = SLICE_PRIORITY[stype]
        for k in range(min_rbs[stype], R_total + 1):
            rho_k = rho_base * DEFAULT_RBS[gnb][stype] / k if k > 0 else 2.0
            p_s   = float(np.clip(sigmoid(rho_k, k_s, rc), 0.0, 1.0))
            cap_k = k * R_PER_RB[gnb][stype]
            pen_t = max(0.0, T_min - min(lam, cap_k)) / max(T_min, 1e-9)
            cost_table[(stype, k)] = OMEGA_T * pen_t + OMEGA_P * prio * p_s
        cost_table[(stype, 0)] = 3.0 * prio

    c = np.array([cost_table[(s, k)] for s, k in var_list], dtype=float)

    n  = len(SLICES)
    A_eq  = np.zeros((n, n_vars))
    b_eq  = np.ones(n)
    for si, stype in enumerate(SLICES):
        for k in range(min_rbs[stype], R_total + 1):
            A_eq[si, idx[(stype, k)]] = 1.0

    A_rbs = np.zeros((1, n_vars))
    for stype in SLICES:
        for k in range(min_rbs[stype], R_total + 1):
            A_rbs[0, idx[(stype, k)]] = k

    A_all = np.vstack([A_eq, A_rbs])
    lb    = np.concatenate([b_eq, [0.0]])
    ub    = np.concatenate([b_eq, [float(R_total)]])

    res = milp(c, constraints=LinearConstraint(A_all, lb, ub),
               integrality=np.ones(n_vars), bounds=Bounds(0, 1))

    if res.success:
        x = np.round(res.x).astype(int)
        alloc = {}
        for stype in SLICES:
            chosen = [k for k in range(min_rbs[stype], R_total + 1)
                      if x[idx[(stype, k)]] == 1]
            alloc[stype] = chosen[0] if chosen else DEFAULT_RBS[gnb][stype]
    else:
        alloc = {s: DEFAULT_RBS[gnb][s] for s in SLICES}
    return alloc

# ── Config initiale ───────────────────────────────────────────────────────────
with open(CONFIG_F, 'w') as f:
    json.dump({gnb: dict(DEFAULT_RBS[gnb]) for gnb in GNBS}, f, indent=2)
print(f"[periodic] Config initiale ecrite (DEFAULT_RBS)")

# ── Boucle principale ─────────────────────────────────────────────────────────
print(f"\n[periodic] En attente de {METRICS_F} ...")
print(f"[periodic] Ctrl+C pour arreter.\n")

step_idx          = 0
last_processed_t  = -1.0

while True:
    if not os.path.exists(METRICS_F):
        time.sleep(1.0)
        continue
    try:
        df_live = pd.read_csv(METRICS_F)
    except Exception:
        time.sleep(1.0)
        continue

    if df_live.empty or 'simtime_s' not in df_live.columns:
        time.sleep(1.0)
        continue

    latest_time = df_live['simtime_s'].max()
    if latest_time - last_processed_t < PERIOD:
        time.sleep(1.0)
        continue

    last_processed_t = latest_time
    step_idx += 1
    print(f"\n[step {step_idx:4d}]  simtime={latest_time:.0f}s")

    allocation = {}
    for gnb in GNBS:
        rho_dict = {s: obs_rho(df_live, gnb, s) for s in SLICES}
        alloc    = milp_optimize(gnb, rho_dict)
        allocation[gnb] = {s: int(alloc[s]) for s in SLICES}
        total = sum(alloc[s] for s in SLICES)
        print(f"  {gnb}: rho={[f'{rho_dict[s]:.2f}' for s in SLICES]} "
              f"-> eMBB={alloc['eMBB']:2d} URLLC={alloc['URLLC']:2d} "
              f"mMTC={alloc['mMTC']:2d} (total={total}/{R_MAX[gnb]})")

    with open(CONFIG_F, 'w') as f:
        json.dump(allocation, f, indent=2)

    time.sleep(max(0.5, PERIOD - 2.0))
