"""
closed_loop_controller.py — Python controller for closed-loop 5G slice management
Runs in parallel with Simu5G (HetNetSlicing).

Loop every 15s:
  1. Read metrics_live.csv (written by SliceController C++ module)
  2. Build features from last 30s of data
  3. LSTM predict → MILP optimize per gNB
  4. Write rb_config.json (read by SliceController C++ module)
  5. Optional: online learning gradient step

Usage:
  python3 closed_loop_controller.py [--sim-dir /path/to/simulation_pfe]
"""

import os, sys, time, json, pickle, copy, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.optim import Adam
from scipy.optimize import milp, LinearConstraint, Bounds
warnings.filterwarnings('ignore')

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--sim-dir', default=os.path.dirname(os.path.abspath(__file__)),
                    help='Directory containing rb_config.json and metrics_live.csv')
parser.add_argument('--models-dir', default=None,
                    help='Directory with trained LSTM models (default: auto-detect)')
parser.add_argument('--control-period', type=float, default=15.0,
                    help='Control period in seconds (must match omnetpp.ini)')
parser.add_argument('--no-online', action='store_true',
                    help='Disable online learning')
parser.add_argument('--demo-init', action='store_true',
                    help='Start with suboptimal allocation (eMBB starved) to show controller correction')
args = parser.parse_args()

SIM_DIR    = args.sim_dir
METRICS_F  = os.path.join(SIM_DIR, 'metrics_live.csv')
CONFIG_F   = os.path.join(SIM_DIR, 'rb_config.json')
PERIOD     = args.control_period

# Auto-detect models directory
if args.models_dir:
    MODELS_DIR = args.models_dir
else:
    # Script: work/simu5g/simulation_pfe/closed_loop_controller.py
    # Models: work/simu5g/loss/models_lstm_v3/
    base = os.path.dirname(os.path.abspath(__file__))  # simulation_pfe/
    MODELS_DIR = os.path.join(base, '..', 'loss', 'models_lstm_v3')

print(f"[controller] SIM_DIR    = {SIM_DIR}")
print(f"[controller] MODELS_DIR = {MODELS_DIR}")
print(f"[controller] METRICS_F  = {METRICS_F}")
print(f"[controller] CONFIG_F   = {CONFIG_F}")

# ── Constants (must match training) ──────────────────────────────────────────
# metrics_live.csv is now written every 1s (controlPeriod=1s in C++).
# SliceController applies rb_config every 15s (rbPeriod=15s).
# Python runs MILP every 15 new rows (= 15s simulated) — matches training.
INPUT_SEC   = 60   # 60s input window (matches training)
OUTPUT_SEC  = 15   # 15s prediction horizon (matches training)
N_IN        = 15     # 5 features × (orig + trend + vol)
N_OUT       = 3      # throughput, latency_log1p, loss
HIDDEN_SIZE = 256
NUM_LAYERS  = 3
DROPOUT     = 0.25
ENG_WINDOW  = 5
LAT_IDX     = 1      # index of latency in prediction output

FEATURES_LIVE = ['avg_latency_ms', 'throughput_mbps', 'pkt_count']  # from metrics_live.csv

# Network topology
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
OMEGA_T = 0.6
OMEGA_P = 0.4

# Slice priority weights for MILP (higher = more protected against violations)
# URLLC has strictest SLA (25ms) → highest priority
SLICE_PRIORITY = {'eMBB': 1.0, 'URLLC': 3.0, 'mMTC': 0.5}

# Minimum RBs per slice to prevent starvation even under overload
MIN_RBS = {
    'Macro':     {'eMBB': 5,  'URLLC': 10, 'mMTC': 1},
    'Commerce':  {'eMBB': 5,  'URLLC':  7, 'mMTC': 1},
    'Industrie': {'eMBB': 3,  'URLLC':  8, 'mMTC': 1},
}
SLICES  = ['eMBB', 'URLLC', 'mMTC']
GNBS    = ['Macro', 'Commerce', 'Industrie']
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Online learning
ONLINE_LR   = 1e-5
REPLAY_SIZE = 64
BATCH_SIZE  = 8
FEDAVG_EVERY = 5

# ── LSTM architecture ─────────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=128):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v     = nn.Linear(attn_dim, 1, bias=False)
    def forward(self, enc_out, dec_hidden):
        energy = self.v(torch.tanh(
            self.W_enc(enc_out) + self.W_dec(dec_hidden).unsqueeze(1)
        )).squeeze(-1)
        attn_w  = F.softmax(energy, dim=1)
        context = (attn_w.unsqueeze(-1) * enc_out).sum(1)
        return context, attn_w

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.proj_h      = nn.Linear(hidden_size * 2, hidden_size)
        self.proj_c      = nn.Linear(hidden_size * 2, hidden_size)
        self.num_layers  = num_layers
        self.hidden_size = hidden_size
    def forward(self, x):
        enc_out, (h, c) = self.lstm(x)
        h = h.view(self.num_layers, 2, -1, self.hidden_size)
        c = c.view(self.num_layers, 2, -1, self.hidden_size)
        h = torch.tanh(self.proj_h(torch.cat([h[:,0], h[:,1]], dim=-1)))
        c = torch.tanh(self.proj_c(torch.cat([c[:,0], c[:,1]], dim=-1)))
        return enc_out, h, c

class LSTMDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        enc_dim  = hidden_size * 2
        self.attn    = BahdanauAttention(enc_dim, hidden_size, 128)
        self.lstm    = nn.LSTM(output_size + enc_dim, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0)
        self.fc_out  = nn.Linear(hidden_size + enc_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    def forward_step(self, prev_pred, enc_out, h, c):
        context, _ = self.attn(enc_out, h[-1])
        lstm_in    = torch.cat([prev_pred, context], dim=1).unsqueeze(1)
        out_lstm, (h, c) = self.lstm(lstm_in, (h, c))
        out = self.fc_out(self.dropout(torch.cat([out_lstm.squeeze(1), context], dim=1)))
        return out, h, c

class Seq2Seq(nn.Module):
    def __init__(self, n_in, n_out, hidden_size, num_layers, dropout):
        super().__init__()
        self.encoder = LSTMEncoder(n_in, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoder(n_out, hidden_size, num_layers, dropout)
        self.n_out   = n_out
    def forward(self, src, tgt_len):
        enc_out, h, c = self.encoder(src)
        prev  = torch.zeros(src.size(0), self.n_out, device=src.device)
        preds = []
        for _ in range(tgt_len):
            out, h, c = self.decoder.forward_step(prev, enc_out, h, c)
            preds.append(out.unsqueeze(1))
            prev = out
        return torch.cat(preds, dim=1)

# ── Load models ───────────────────────────────────────────────────────────────

models     = {}
scalers    = {}
optimizers = {}
replay_buffers = {}

for stype in SLICES:
    mp = os.path.join(MODELS_DIR, f'model_final_{stype}.pt')
    sp = os.path.join(MODELS_DIR, f'scalers_final_{stype}.pkl')
    if not os.path.exists(mp):
        print(f"[controller] WARNING: model not found: {mp}")
        continue
    model = Seq2Seq(N_IN, N_OUT, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(mp, map_location=DEVICE, weights_only=False))
    model.train()
    with open(sp, 'rb') as fh:
        sc_all = pickle.load(fh)
    for gnb in GNBS:
        models[(stype, gnb)]         = model
        optimizers[(stype, gnb)]     = Adam(model.parameters(), lr=ONLINE_LR)
        replay_buffers[(stype, gnb)] = deque(maxlen=REPLAY_SIZE)
        sc_entry = sc_all[(stype, gnb)]
        scalers[(stype, gnb)]        = {'src': sc_entry['x'], 'tgt': sc_entry['y']}
    print(f"[controller] Loaded {stype} -> partage sur {GNBS}")

print(f"[controller] Loaded {len(models)}/9 models from {MODELS_DIR}")

# ── Sigmoid calibration (default params if no training data yet) ───────────────

def sigmoid(rho, k, rho_c):
    return 1.0 / (1.0 + np.exp(-k * (rho - rho_c)))

# Default calibrated params (from offline training)
SIGMOID_PARAMS = {
    ('Macro',     'eMBB'):  (5.0, 1.0),
    ('Macro',     'URLLC'): (5.0, 1.0),
    ('Macro',     'mMTC'):  (5.0, 1.0),
    ('Commerce',  'eMBB'):  (5.0, 1.0),
    ('Commerce',  'URLLC'): (5.0, 1.0),
    ('Commerce',  'mMTC'):  (5.0, 1.0),
    ('Industrie', 'eMBB'):  (0.50, 2.785),  # degenerate — keep conservative
    ('Industrie', 'URLLC'): (5.0, 1.0),
    ('Industrie', 'mMTC'):  (5.0, 1.0),
}

# Load sigmoid params from offline calibration if available
CALIB_PATH = os.path.join(os.path.dirname(MODELS_DIR), 'simu5g', 'output', 'sigmoid_params.pkl')
if os.path.exists(CALIB_PATH):
    with open(CALIB_PATH, 'rb') as f:
        SIGMOID_PARAMS.update(pickle.load(f))
    print(f"[controller] Sigmoid params loaded from {CALIB_PATH}")
else:
    print(f"[controller] Using default sigmoid params")

# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(series, window=ENG_WINDOW):
    """series: (T, 5) → (T, 15) with original + trend + volatility"""
    trend = np.zeros_like(series)
    trend[1:] = np.diff(series, axis=0)
    vol = np.zeros_like(series)
    if len(series) >= window:
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(series, window_shape=window, axis=0)
        vol[window-1:] = wins.std(axis=-1)
    return np.concatenate([series, trend, vol], axis=1).astype(np.float32)

def build_input_from_live(df_live, stype, gnb, scaler):
    """
    Build LSTM input from metrics_live.csv data.
    df_live: rows with columns [simtime_s, gnb, slice, avg_latency_ms, throughput_mbps, pkt_count, pkt_loss_pct]
    Returns: tensor (1, INPUT_SEC, N_IN) or None if insufficient data
    """
    sub = df_live[(df_live['slice'] == stype) & (df_live['gnb'] == gnb)].sort_values('simtime_s').tail(INPUT_SEC)
    if len(sub) < INPUT_SEC:
        return None

    # Build 5-feature matrix matching FEATURES_PROC used in training:
    # ['Slice_Throughput_Mbps', 'Slice_Latency_log1p', 'Slice_Packet_Loss_pct',
    #  'Slice_Jitter_log1p', 'Slice_Network_Load_pct']
    tp  = sub['throughput_mbps'].values
    lat = np.log1p(sub['avg_latency_ms'].clip(0, 600_000).values)  # log1p transform
    # Use pkt_loss_pct directly if available, else fall back to pkt_count proxy
    if 'pkt_loss_pct' in sub.columns:
        loss = sub['pkt_loss_pct'].values.astype(float)
    else:
        pkt  = sub['pkt_count'].values.astype(float)
        loss = np.where(pkt > 0, 0.0, 1.0)
    load   = np.clip(tp / max(R_MAX[gnb] * R_PER_RB[gnb][stype], 1e-6), 0, 2)
    jitter = np.zeros_like(tp)                # not available from aggregate metrics

    raw  = np.stack([tp, lat, loss, jitter, load], axis=1).astype(np.float32)
    feat = engineer_features(raw)             # (T, 15)

    src_scaler = scaler['src'] if isinstance(scaler, dict) else scaler
    feat_scaled = src_scaler.transform(feat)
    x = torch.tensor(feat_scaled[np.newaxis], dtype=torch.float32).to(DEVICE)
    return x

# ── MILP optimizer ────────────────────────────────────────────────────────────

def milp_optimize(gnb, rho_dict):
    """
    Optimize RB allocation for one gNB given predicted rho per slice.
    Returns dict {slice: nb_rbs}
    """
    R_total = R_MAX[gnb]
    n = len(SLICES)

    cost_table = {}
    for stype in SLICES:
        rho_base = rho_dict.get(stype, 0.1)
        lam      = rho_base * DEFAULT_RBS[gnb][stype] * R_PER_RB[gnb][stype]
        k_s, rc  = SIGMOID_PARAMS[(gnb, stype)]
        R_def    = DEFAULT_RBS[gnb][stype]
        T_min    = SLA[stype]['T_min']
        prio     = SLICE_PRIORITY[stype]
        for k in range(1, R_total + 1):
            rho_k = rho_base * R_def / k if k > 0 else 2.0
            p_s   = float(np.clip(sigmoid(rho_k, k_s, rc), 0.0, 1.0))
            cap_k = k * R_PER_RB[gnb][stype]
            tp_ok = min(lam, cap_k)
            pen_t = max(0.0, T_min - tp_ok) / max(T_min, 1e-9)
            cost_table[(stype, k)] = OMEGA_T * pen_t + OMEGA_P * prio * p_s
        cost_table[(stype, 0)] = 3.0 * prio  # worst cost for k=0

    # Build MILP problem: one binary variable per (slice, k)
    # Only allow k >= MIN_RBS[gnb][stype] to prevent starvation
    min_rbs = MIN_RBS[gnb]
    var_list = [(stype, k) for stype in SLICES
                for k in range(min_rbs[stype], R_total + 1)]
    n_vars   = len(var_list)
    c        = np.array([cost_table[(s, k)] for s, k in var_list], dtype=float)

    idx = {v: i for i, v in enumerate(var_list)}

    # Constraint 1: exactly 1 allocation per slice
    A_eq  = np.zeros((n, n_vars))
    b_eq  = np.ones(n)
    for si, stype in enumerate(SLICES):
        for k in range(min_rbs[stype], R_total + 1):
            A_eq[si, idx[(stype, k)]] = 1.0

    # Constraint 2: total RBs <= R_total
    A_rbs = np.zeros((1, n_vars))
    for stype in SLICES:
        for k in range(min_rbs[stype], R_total + 1):
            A_rbs[0, idx[(stype, k)]] = k

    A_all = np.vstack([A_eq, A_rbs])
    lb    = np.concatenate([b_eq, [0.0]])
    ub    = np.concatenate([b_eq, [float(R_total)]])

    res = milp(c,
               constraints=LinearConstraint(A_all, lb, ub),
               integrality=np.ones(n_vars),
               bounds=Bounds(0, 1))

    alloc = {}
    if res.success:
        x = np.round(res.x).astype(int)
        for stype in SLICES:
            chosen = [k for k in range(min_rbs[stype], R_total + 1) if x[idx[(stype, k)]] == 1]
            alloc[stype] = chosen[0] if chosen else max(DEFAULT_RBS[gnb][stype], min_rbs[stype])
    else:
        alloc = {s: max(DEFAULT_RBS[gnb][s], min_rbs[s]) for s in SLICES}

    return alloc

# ── Predict rho from live metrics ─────────────────────────────────────────────

def predict_rho(df_live, gnb, stype):
    """
    Use LSTM to predict next 15s and estimate rho.
    Returns (rho, info_dict) where info_dict contains prediction details.
    Falls back to throughput-based rho if model not available or not enough data.
    """
    key = (stype, gnb)

    if key not in models or key not in scalers:
        sub = df_live[(df_live['slice'] == stype) & (df_live['gnb'] == gnb)].tail(1)
        if sub.empty:
            return 0.1, {'mode': 'no_model', 'rho': 0.1}
        tp = float(sub['throughput_mbps'].iloc[0])
        cap = DEFAULT_RBS[gnb][stype] * R_PER_RB[gnb][stype]
        rho = float(np.clip(tp / max(cap, 1e-6), 0.0, 2.0))
        return rho, {'mode': 'no_model', 'rho': rho, 'obs_tp': tp}

    x = build_input_from_live(df_live, stype, gnb, scalers[key])
    if x is None:
        sub = df_live[(df_live['slice'] == stype) & (df_live['gnb'] == gnb)].tail(1)
        if sub.empty:
            return 0.1, {'mode': 'fallback_nodata', 'rho': 0.1}
        tp = float(sub['throughput_mbps'].iloc[0])
        lat = float(sub['avg_latency_ms'].iloc[0])
        cap = DEFAULT_RBS[gnb][stype] * R_PER_RB[gnb][stype]
        rho = float(np.clip(tp / max(cap, 1e-6), 0.0, 2.0))
        rows_available = len(df_live[(df_live['slice'] == stype) & (df_live['gnb'] == gnb)])
        return rho, {
            'mode': 'fallback_warmup',
            'rho': rho,
            'rows': rows_available,
            'need': INPUT_SEC,
            'obs_tp': tp,
            'obs_lat_ms': lat,
        }

    model = models[key]
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x, OUTPUT_SEC).cpu().numpy()[0]  # (OUTPUT_SEC, 3)

    # Inverse transform using target scaler
    tgt_scaler = scalers[key]['tgt'] if isinstance(scalers[key], dict) else scalers[key]
    pred_inv = tgt_scaler.inverse_transform(pred_scaled)  # (OUTPUT_SEC, 3)

    tp_pred_vec = pred_inv[:, 0]
    lat_ms_vec  = np.expm1(pred_inv[:, 1])
    tp_pred     = float(tp_pred_vec.mean())
    lat_pred    = float(lat_ms_vec.mean())

    # ── Distribution shift guard ──────────────────────────────────────────────
    # If the LSTM was trained on a different operating regime (e.g. unbounded
    # MAC buffer → high latency), its latency predictions will be far off.
    # Detect this: if pred_lat >> obs_lat, trust the observed latency instead.
    obs_sub = df_live[(df_live['slice'] == stype) & (df_live['gnb'] == gnb)].tail(5)
    obs_lat_ms   = float(obs_sub['avg_latency_ms'].mean()) if not obs_sub.empty else None
    shift_detected = False
    if obs_lat_ms is not None and lat_pred > obs_lat_ms * 5:
        # Significant distribution shift — use observed latency for p_viol
        shift_detected = True
        p_viol = float(obs_lat_ms > SLA[stype]['L_max_ms'])
    else:
        p_viol = float((lat_ms_vec > SLA[stype]['L_max_ms']).mean())

    k_s, rc = SIGMOID_PARAMS[(gnb, stype)]
    if p_viol >= 0.99:
        rho = 2.0
    elif p_viol <= 0.01:
        cap = DEFAULT_RBS[gnb][stype] * R_PER_RB[gnb][stype]
        rho = float(np.clip(tp_pred / max(cap, 1e-6), 0.0, 1.5))
    else:
        rho = float(np.clip(
            rc - np.log(1.0 / max(p_viol, 1e-6) - 1.0) / max(k_s, 0.01),
            0.0, 2.0))

    model.train()
    return rho, {
        'mode': 'lstm',
        'rho': rho,
        'pred_tp_mbps': tp_pred,
        'pred_lat_ms': lat_pred,
        'obs_lat_ms': obs_lat_ms,
        'p_viol': p_viol,
        'sla_lat_ms': SLA[stype]['L_max_ms'],
        'shift_detected': shift_detected,
    }

# ── Online learning ───────────────────────────────────────────────────────────

def online_update(gnb, stype, x_np, y_np):
    """Single gradient step using replay buffer."""
    key = (stype, gnb)
    if key not in models:
        return

    # Normalize
    sc = scalers[key]
    src_sc = sc['src'] if isinstance(sc, dict) else sc
    tgt_sc = sc['tgt'] if isinstance(sc, dict) else sc
    x_scaled = src_sc.transform(x_np).astype(np.float32)
    y_scaled = tgt_sc.transform(y_np).astype(np.float32)

    replay_buffers[key].append((x_scaled, y_scaled))
    if len(replay_buffers[key]) < BATCH_SIZE:
        return

    batch = list(replay_buffers[key])[-BATCH_SIZE:]
    xs  = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32).to(DEVICE)
    ys  = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32).to(DEVICE)

    models[key].train()
    optimizers[key].zero_grad()
    pred = models[key](xs, OUTPUT_SEC)
    loss = nn.MSELoss()(pred, ys)
    loss.backward()
    optimizers[key].step()

def fedavg_embb():
    """Average eMBB models across gNBs (consistent with training strategy)."""
    embb_models = [models.get(('eMBB', g)) for g in GNBS if ('eMBB', g) in models]
    if len(embb_models) < 2:
        return
    avg_state = {}
    for key in embb_models[0].state_dict():
        avg_state[key] = torch.stack([m.state_dict()[key].float() for m in embb_models]).mean(0)
    for m in embb_models:
        m.load_state_dict(avg_state)

# ── Main control loop ─────────────────────────────────────────────────────────

print(f"\n[controller] Waiting for {METRICS_F} ...")
print(f"[controller] Will write allocations to {CONFIG_F}")
print(f"[controller] Press Ctrl+C to stop.\n")

# Write initial rb_config.json
# For ControllerDemo: start with suboptimal allocation to create visible initial violation
# Python will correct it after the first MILP step (~15s simulation time)
# For other scenarios: DEFAULT_RBS is used
DEMO_INIT_CONFIG = {
    # Moderate under-allocation for eMBB (25% below DEFAULT) — enough to create
    # a visible violation without catastrophically degrading URLLC/mMTC.
    # DEFAULT → DEMO_INIT: eMBB 20→15, URLLC 17→22, mMTC 13→13  (Macro, total=50)
    # DEFAULT → DEMO_INIT: eMBB 21→15, URLLC 11→14, mMTC  3→6   (Commerce, total=35)
    # DEFAULT → DEMO_INIT: eMBB  6→ 4, URLLC 12→14, mMTC  7→7   (Industrie, total=25)
    'Macro':     {'eMBB': 15, 'URLLC': 22, 'mMTC': 13},  # eMBB 25% below DEFAULT
    'Commerce':  {'eMBB': 15, 'URLLC': 14, 'mMTC':  6},  # eMBB 29% below DEFAULT
    'Industrie': {'eMBB':  4, 'URLLC': 14, 'mMTC':  7},  # eMBB 33% below DEFAULT
}
initial_config = DEMO_INIT_CONFIG if args.demo_init else {gnb: dict(DEFAULT_RBS[gnb]) for gnb in GNBS}
with open(CONFIG_F, 'w') as f:
    json.dump(initial_config, f, indent=2)
print(f"[controller] Initial config written: {initial_config}")
if args.demo_init:
    print(f"[controller] ** demo-init mode: starting with suboptimal allocation (eMBB starved) **")

step_idx = 0
last_processed_time = -1.0

while True:
    # ── Wait for new data ──────────────────────────────────────────────────────
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
    # Run MILP every 15s of simulation time (rb_config applied by C++ every 15s)
    if latest_time - last_processed_time < 15.0:
        time.sleep(1.0)
        continue

    # New 15s window available — process it
    last_processed_time = latest_time
    step_idx += 1

    n_rows = len(df_live)
    print(f"\n{'='*70}")
    print(f"[step {step_idx:4d}]  simtime={latest_time:.0f}s  rows_in_csv={n_rows}")
    print(f"{'='*70}")

    # ── Observed metrics (last row per slice) ─────────────────────────────────
    print(f"  {'':12s}  {'Throughput':>12s}  {'Latency':>10s}  {'Pkts':>6s}")
    for stype in SLICES:
        sub = df_live[df_live['slice'] == stype].tail(1)
        if not sub.empty:
            tp  = float(sub['throughput_mbps'].iloc[0])
            lat = float(sub['avg_latency_ms'].iloc[0])
            pkts = int(sub['pkt_count'].iloc[0])
            sla_ok = '✓' if lat <= SLA[stype]['L_max_ms'] else '✗ SLA!'
            print(f"  obs {stype:8s}:  {tp:10.3f} Mbps  {lat:8.2f} ms  {pkts:6d}  {sla_ok}")

    # ── LSTM predictions + MILP per gNB ──────────────────────────────────────
    allocation = {}
    for gnb in GNBS:
        print(f"\n  ── {gnb} (R_MAX={R_MAX[gnb]} RBs) ──")

        rho_dict = {}
        info_dict = {}
        for stype in SLICES:
            rho, info = predict_rho(df_live, gnb, stype)
            rho_dict[stype] = rho
            info_dict[stype] = info

            mode = info['mode']
            if mode == 'lstm':
                sla_flag = '✗ PRED VIOL' if info['p_viol'] > 0.5 else '✓'
                shift_tag = ' [SHIFT→obs]' if info.get('shift_detected') else ''
                obs_lat_str = f"  obs_lat={info['obs_lat_ms']:.1f}ms" if info.get('obs_lat_ms') is not None else ''
                print(f"    {stype:6s}: LSTM  rho={rho:.3f}  "
                      f"pred_tp={info['pred_tp_mbps']:.2f}Mbps  "
                      f"pred_lat={info['pred_lat_ms']:.1f}ms{obs_lat_str}  "
                      f"p_viol={info['p_viol']:.0%}  {sla_flag}{shift_tag}")
            elif mode == 'fallback_warmup':
                print(f"    {stype:6s}: WARMUP ({info['rows']}/{info['need']} rows)  "
                      f"rho={rho:.3f}  obs_tp={info['obs_tp']:.2f}Mbps  "
                      f"obs_lat={info['obs_lat_ms']:.1f}ms")
            else:
                print(f"    {stype:6s}: FALLBACK  rho={rho:.3f}")

        # MILP
        prev_alloc = allocation.get(gnb, DEFAULT_RBS[gnb])
        alloc = milp_optimize(gnb, rho_dict)
        allocation[gnb] = {s: int(alloc[s]) for s in SLICES}

        total = sum(alloc[s] for s in SLICES)
        print(f"    MILP → eMBB={alloc['eMBB']:2d} URLLC={alloc['URLLC']:2d} mMTC={alloc['mMTC']:2d}  "
              f"(total={total}/{R_MAX[gnb]})")

    print()

    # ── Write rb_config.json ──────────────────────────────────────────────────
    with open(CONFIG_F, 'w') as f:
        json.dump(allocation, f, indent=2)

    # ── Online learning ───────────────────────────────────────────────────────
    n_rows_per_slice = len(df_live[df_live['slice'] == SLICES[0]])
    if not args.no_online and n_rows_per_slice >= INPUT_SEC + OUTPUT_SEC:
        updated = []
        for gnb in GNBS:
            for stype in SLICES:
                sub = df_live[df_live['slice'] == stype].sort_values('simtime_s')
                if len(sub) >= INPUT_SEC + OUTPUT_SEC:
                    x_rows = sub.iloc[-(INPUT_SEC + OUTPUT_SEC):-OUTPUT_SEC]
                    y_rows = sub.iloc[-OUTPUT_SEC:]
                    tp     = x_rows['throughput_mbps'].values
                    lat    = np.log1p(x_rows['avg_latency_ms'].clip(0, 600_000))
                    loss   = np.zeros_like(tp)
                    jitter = np.zeros_like(tp)
                    load   = np.clip(tp / max(R_MAX[gnb] * R_PER_RB[gnb][stype], 1e-6), 0, 2)
                    x_raw  = np.stack([tp, lat, loss, jitter, load], axis=1).astype(np.float32)
                    x_feat = engineer_features(x_raw)
                    y_raw  = np.stack([
                        y_rows['throughput_mbps'].values,
                        np.log1p(y_rows['avg_latency_ms'].clip(0, 600_000)),
                        np.zeros(OUTPUT_SEC)
                    ], axis=1).astype(np.float32)
                    if (stype, gnb) in models:
                        buf_len = len(replay_buffers[(stype, gnb)])
                        online_update(gnb, stype, x_feat, y_raw)
                        if buf_len >= BATCH_SIZE:  # gradient step actually ran
                            updated.append(f"{stype}/{gnb}")

        if updated:
            print(f"  [OnlineLR] gradient step: {', '.join(updated)}")
        else:
            print(f"  [OnlineLR] warmup replay buffer ({buf_len}/{BATCH_SIZE} samples)")

        # FedAvg for eMBB only (consistent with training)
        if step_idx % FEDAVG_EVERY == 0:
            fedavg_embb()
            print(f"  [FedAvg]   eMBB models averaged across {len(GNBS)} gNBs")
    else:
        remaining = (INPUT_SEC + OUTPUT_SEC) - n_rows_per_slice
        print(f"  [OnlineLR] waiting for data ({n_rows_per_slice}/{INPUT_SEC + OUTPUT_SEC} rows)")

    time.sleep(max(0.5, PERIOD - 2.0))  # check slightly before next period
