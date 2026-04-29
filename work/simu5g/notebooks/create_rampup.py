"""
Création de scénarios de transition synthétiques (ramp-up / ramp-down)
par interpolation sigmoïde entre un scénario LOW et un scénario HIGH.

Approche :
  alpha(t) = sigmoid((t - t_mid) / steepness)
  kpi(t)   = (1 - alpha(t)) * kpi_low(t) + alpha(t) * kpi_high(t)

Les KPIs interpolés : Latency, Packet_Loss, Jitter, Network_Load, Throughput
SLA_OK, SLA_OK_in_15s, SLA_Violated_in_15s sont recalculés après interpolation.
"""

import pandas as pd
import numpy as np

df = pd.read_csv('all_simu5g.csv')

SLA_THRESHOLDS = {
    'eMBB':  {'latency': 50.0,  'loss': 1.0},
    'URLLC': {'latency': 20.0,  'loss': 0.001},
    'mMTC':  {'latency': 8.8,   'loss': 1.0},
}

KPI_COLS = ['Slice_Latency_ms', 'Slice_Packet_Loss_pct',
            'Slice_Jitter_ms', 'Slice_Network_Load_pct',
            'Slice_Throughput_Mbps']

def sigmoid(t, t_mid, steepness):
    return 1.0 / (1.0 + np.exp(-(t - t_mid) / steepness))


def make_rampup(df, sl, gnb, sc_low, sc_high, name, steepness=180, noise=0.02):
    """
    Crée une série de transition de sc_low vers sc_high.
    steepness : largeur de la transition (secondes)
    noise     : bruit gaussien relatif pour rendre la série réaliste
    """
    low  = df[(df['Slice_Type']==sl) & (df['gNB_id']==gnb) &
              (df['Scenario']==sc_low)].sort_values('Time_Sec').reset_index(drop=True)
    high = df[(df['Slice_Type']==sl) & (df['gNB_id']==gnb) &
              (df['Scenario']==sc_high)].sort_values('Time_Sec').reset_index(drop=True)

    # Aligner sur la longueur minimale
    T = min(len(low), len(high))
    low  = low.iloc[:T].reset_index(drop=True)
    high = high.iloc[:T].reset_index(drop=True)

    t_arr  = np.arange(T, dtype=float)
    t_mid  = T / 2.0
    alpha  = sigmoid(t_arr, t_mid, steepness)

    row = low.copy()
    row['Scenario'] = name

    for col in KPI_COLS:
        interpolated = (1 - alpha) * low[col].values + alpha * high[col].values
        # Bruit gaussien relatif
        std = np.abs(interpolated) * noise + 1e-6
        interpolated += np.random.normal(0, std)
        # Clip pour rester physiquement plausible
        if 'Loss' in col or 'Load' in col:
            interpolated = np.clip(interpolated, 0, 100)
        elif col == 'Slice_Latency_ms':
            interpolated = np.clip(interpolated, 0, 600_000)
        else:
            interpolated = np.clip(interpolated, 0, None)
        row[col] = interpolated

    # Recalculer SLA_OK
    thr = SLA_THRESHOLDS[sl]
    row['SLA_OK'] = ((row['Slice_Latency_ms'] <= thr['latency']) &
                     (row['Slice_Packet_Loss_pct'] <= thr['loss'])).astype(int)

    # Recalculer SLA_OK_in_15s et SLA_Violated_in_15s (lookahead 15 pas)
    sla_arr = row['SLA_OK'].values
    ok_15   = np.zeros(T, dtype=int)
    viol_15 = np.zeros(T, dtype=int)
    for i in range(T):
        future = sla_arr[i:i+15]
        ok_15[i]   = int(np.all(future == 1))
        viol_15[i] = int(np.any(future == 0))
    row['SLA_OK_in_15s']      = ok_15
    row['SLA_Violated_in_15s'] = viol_15

    return row


def make_transition(df, sl, gnb, sc_low, sc_high, name,
                    t_start_frac=0.3, t_end_frac=0.7, noise=0.02):
    """
    Crée une série avec :
      [0, t_start]       : régime LOW pur
      [t_start, t_end]   : transition linéaire lissée
      [t_end, T]         : régime HIGH pur
    Garantit ~30% LOW + ~40% transition + ~30% HIGH → viol_rate ~30-70%.
    """
    low  = df[(df['Slice_Type']==sl) & (df['gNB_id']==gnb) &
              (df['Scenario']==sc_low)].sort_values('Time_Sec').reset_index(drop=True)
    high = df[(df['Slice_Type']==sl) & (df['gNB_id']==gnb) &
              (df['Scenario']==sc_high)].sort_values('Time_Sec').reset_index(drop=True)

    T = min(len(low), len(high))
    low  = low.iloc[:T].reset_index(drop=True)
    high = high.iloc[:T].reset_index(drop=True)

    t_start = int(T * t_start_frac)
    t_end   = int(T * t_end_frac)

    # Alpha : 0 avant t_start, montée linéaire, 1 après t_end
    alpha = np.zeros(T)
    alpha[t_start:t_end] = np.linspace(0, 1, t_end - t_start)
    alpha[t_end:] = 1.0

    row = low.copy()
    row['Scenario'] = name

    for col in KPI_COLS:
        interpolated = (1 - alpha) * low[col].values + alpha * high[col].values
        std = np.abs(interpolated) * noise + 1e-6
        interpolated += np.random.normal(0, std)
        if 'Loss' in col or 'Load' in col:
            interpolated = np.clip(interpolated, 0, 100)
        elif col == 'Slice_Latency_ms':
            interpolated = np.clip(interpolated, 0, 600_000)
        else:
            interpolated = np.clip(interpolated, 0, None)
        row[col] = interpolated

    thr = SLA_THRESHOLDS[sl]
    row['SLA_OK'] = ((row['Slice_Latency_ms'] <= thr['latency']) &
                     (row['Slice_Packet_Loss_pct'] <= thr['loss'])).astype(int)

    sla_arr = row['SLA_OK'].values
    ok_15   = np.zeros(T, dtype=int)
    viol_15 = np.zeros(T, dtype=int)
    for i in range(T):
        future = sla_arr[i:i+15]
        ok_15[i]   = int(np.all(future == 1))
        viol_15[i] = int(np.any(future == 0))
    row['SLA_OK_in_15s']       = ok_15
    row['SLA_Violated_in_15s'] = viol_15

    return row


np.random.seed(42)
ramp_scenarios = []

# ── eMBB ──────────────────────────────────────────────────────────────────
for gnb in ['Macro', 'Commerce', 'Industrie']:
    # ~70% violation : 30% LOW + 40% transition + 30% HIGH
    ramp_scenarios.append(make_transition(
        df, 'eMBB', gnb, 'NormalLoad', 'GlobalSaturation',
        'Trans_RampUp_eMBB', t_start_frac=0.3, t_end_frac=0.7))
    # ~70% violation : descente (HIGH → LOW, même proportion)
    ramp_scenarios.append(make_transition(
        df, 'eMBB', gnb, 'GlobalSaturation', 'NormalLoad',
        'Trans_RampDown_eMBB', t_start_frac=0.3, t_end_frac=0.7))
    # ~60% violation : pic rapide (40% LOW + 20% trans + 40% HIGH)
    ramp_scenarios.append(make_transition(
        df, 'eMBB', gnb, 'NormalLoad', 'GlobalSaturation',
        'Trans_Peak_eMBB', t_start_frac=0.4, t_end_frac=0.6))
    # ~40% violation : 60% LOW + 30% trans + 10% HIGH
    ramp_scenarios.append(make_transition(
        df, 'eMBB', gnb, 'NormalLoad', 'GlobalSaturation',
        'Trans_Gradual_eMBB', t_start_frac=0.6, t_end_frac=0.9))
    # ~25% violation : 75% LOW + 20% trans + 5% HIGH
    ramp_scenarios.append(make_transition(
        df, 'eMBB', gnb, 'NormalLoad', 'GlobalSaturation',
        'Trans_LowViol_eMBB', t_start_frac=0.75, t_end_frac=0.95))

# ── URLLC ─────────────────────────────────────────────────────────────────
for gnb in ['Macro', 'Commerce', 'Industrie']:
    ramp_scenarios.append(make_transition(
        df, 'URLLC', gnb, 'LowTrafficNight', 'GlobalSaturation',
        'Trans_RampUp_URLLC', t_start_frac=0.3, t_end_frac=0.7))
    ramp_scenarios.append(make_transition(
        df, 'URLLC', gnb, 'GlobalSaturation', 'LowTrafficNight',
        'Trans_RampDown_URLLC', t_start_frac=0.3, t_end_frac=0.7))
    ramp_scenarios.append(make_transition(
        df, 'URLLC', gnb, 'LowTrafficNight', 'GlobalSaturation',
        'Trans_Peak_URLLC', t_start_frac=0.4, t_end_frac=0.6))
    # ~40% violation
    ramp_scenarios.append(make_transition(
        df, 'URLLC', gnb, 'LowTrafficNight', 'GlobalSaturation',
        'Trans_Gradual_URLLC', t_start_frac=0.6, t_end_frac=0.9))
    # ~25% violation
    ramp_scenarios.append(make_transition(
        df, 'URLLC', gnb, 'LowTrafficNight', 'GlobalSaturation',
        'Trans_LowViol_URLLC', t_start_frac=0.75, t_end_frac=0.95))

# ── mMTC ──────────────────────────────────────────────────────────────────
for gnb in ['Macro', 'Commerce', 'Industrie']:
    ramp_scenarios.append(make_transition(
        df, 'mMTC', gnb, 'NormalLoad', 'ModerateLoad_eMBB',
        'Trans_RampUp_mMTC', t_start_frac=0.3, t_end_frac=0.7))
    ramp_scenarios.append(make_transition(
        df, 'mMTC', gnb, 'ModerateLoad_eMBB', 'NormalLoad',
        'Trans_RampDown_mMTC', t_start_frac=0.3, t_end_frac=0.7))
    # ~40% violation
    ramp_scenarios.append(make_transition(
        df, 'mMTC', gnb, 'NormalLoad', 'ModerateLoad_eMBB',
        'Trans_Gradual_mMTC', t_start_frac=0.6, t_end_frac=0.9))
    # ~25% violation
    ramp_scenarios.append(make_transition(
        df, 'mMTC', gnb, 'NormalLoad', 'ModerateLoad_eMBB',
        'Trans_LowViol_mMTC', t_start_frac=0.75, t_end_frac=0.95))

all_ramp = pd.concat(ramp_scenarios, ignore_index=True)

print('=== Scénarios synthétiques créés ===')
print(f'Lignes totales : {len(all_ramp):,}')
print()

# Stats viol_rate par scenario × slice
stats = all_ramp.groupby(['Scenario','Slice_Type'])['SLA_OK'].apply(
    lambda x: (x==0).mean()*100).round(1).reset_index()
stats.columns = ['Scenario','Slice','viol_rate%']
print(stats.to_string(index=False))

# Combiner avec le dataset original
df_combined = pd.concat([df, all_ramp], ignore_index=True)
print(f'\nDataset original : {len(df):,} lignes')
print(f'Scénarios ramp   : {len(all_ramp):,} lignes')
print(f'Dataset combiné  : {len(df_combined):,} lignes')
print(f'Scenarios total  : {df_combined["Scenario"].nunique()}')

df_combined.to_csv('all_simu5g_trans.csv', index=False)
print('\nSauvegarde : all_simu5g_trans.csv')
