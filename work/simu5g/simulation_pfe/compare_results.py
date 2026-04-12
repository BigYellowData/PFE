"""
compare_results.py — Analyse comparative des 3 modes de simulation.

Usage:
    python3 compare_results.py <results_base_dir>
    python3 compare_results.py comparison_results/GlobalSaturation_seed0
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

MODES = {
    'baseline':     'Baseline (aucune realloc)',
    'periodic_milp': 'MILP periodique (sans LSTM)',
    'lstm_milp':    'LSTM + MILP adaptatif',
}

SLA = {
    'eMBB':  {'latency_max_ms': 60.0,  'loss_max_pct': 1.0},
    'URLLC': {'latency_max_ms': 25.0,  'loss_max_pct': 0.001},
    'mMTC':  {'latency_max_ms': 300.0, 'loss_max_pct': 1.0},
}

SLICES = ['eMBB', 'URLLC', 'mMTC']


def load_metrics(base_dir, mode):
    path = os.path.join(base_dir, mode, 'metrics_live.csv')
    if not os.path.exists(path):
        print(f"  [warn] {path} non trouve")
        return None
    df = pd.read_csv(path)
    df['mode'] = mode
    return df


def compute_kpis(df):
    results = {}
    for sl in SLICES:
        sub = df[df['slice'] == sl]
        if sub.empty:
            continue
        lat = sub['avg_latency_ms'].values
        thr = sub['throughput_mbps'].values
        loss = sub['pkt_loss_pct'].values if 'pkt_loss_pct' in sub.columns else np.zeros(len(sub))
        sla_lat  = SLA[sl]['latency_max_ms']
        sla_loss = SLA[sl]['loss_max_pct']
        viol_rate = float(((lat > sla_lat) | (loss > sla_loss)).mean() * 100)
        results[sl] = {
            'mean_lat_ms':  float(lat.mean()),
            'p95_lat_ms':   float(np.percentile(lat, 95)),
            'mean_thr_mbps': float(thr.mean()),
            'viol_rate_pct': viol_rate,
            'n_windows':    len(sub),
        }
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_results.py <results_base_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]
    print(f"\n=== Comparaison : {base_dir} ===\n")

    all_dfs = {}
    all_kpis = {}
    for mode in MODES:
        df = load_metrics(base_dir, mode)
        if df is not None:
            all_dfs[mode] = df
            all_kpis[mode] = compute_kpis(df)

    if not all_kpis:
        print("Aucune donnee trouvee.")
        sys.exit(1)

    # ── Tableau texte ─────────────────────────────────────────────────────────
    print(f"{'Mode':<30} {'Slice':<6} {'Viol%':>7} {'Lat moy':>9} {'Lat p95':>9} {'Thr moy':>9}")
    print("-" * 75)
    for mode, label in MODES.items():
        if mode not in all_kpis:
            continue
        for sl in SLICES:
            if sl not in all_kpis[mode]:
                continue
            k = all_kpis[mode][sl]
            print(f"  {label:<28} {sl:<6} {k['viol_rate_pct']:>6.1f}% "
                  f"{k['mean_lat_ms']:>8.1f}ms {k['p95_lat_ms']:>8.1f}ms "
                  f"{k['mean_thr_mbps']:>8.2f}Mbps")
        print()

    # ── Gain par rapport au baseline ──────────────────────────────────────────
    if 'baseline' in all_kpis:
        print(f"\n{'Mode':<30} {'Slice':<6} {'Gain viol%':>12} {'Gain lat':>10}")
        print("-" * 65)
        base = all_kpis['baseline']
        for mode in ['periodic_milp', 'lstm_milp']:
            if mode not in all_kpis:
                continue
            label = MODES[mode]
            for sl in SLICES:
                if sl not in all_kpis[mode] or sl not in base:
                    continue
                gain_viol = base[sl]['viol_rate_pct'] - all_kpis[mode][sl]['viol_rate_pct']
                gain_lat  = base[sl]['mean_lat_ms'] - all_kpis[mode][sl]['mean_lat_ms']
                print(f"  {label:<28} {sl:<6} {gain_viol:>+10.1f}pp {gain_lat:>+8.1f}ms")
            print()

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(len(SLICES), 2, figsize=(14, 4 * len(SLICES)))
    colors = {'baseline': '#e74c3c', 'periodic_milp': '#f39c12', 'lstm_milp': '#2ecc71'}

    for row, sl in enumerate(SLICES):
        ax_lat = axes[row, 0]
        ax_viol = axes[row, 1]

        for mode, label in MODES.items():
            if mode not in all_dfs:
                continue
            sub = all_dfs[mode][all_dfs[mode]['slice'] == sl].sort_values('simtime_s')
            if sub.empty:
                continue
            c = colors[mode]
            ax_lat.plot(sub['simtime_s'], sub['avg_latency_ms'],
                        label=label, color=c, alpha=0.8, linewidth=1.2)

        sla_lat = SLA[sl]['latency_max_ms']
        ax_lat.axhline(sla_lat, color='k', linestyle='--', linewidth=1, label=f'SLA ({sla_lat}ms)')
        ax_lat.set_title(f'{sl} — Latence')
        ax_lat.set_xlabel('Temps simulation (s)')
        ax_lat.set_ylabel('Latence (ms)')
        ax_lat.legend(fontsize=7)
        ax_lat.grid(True, alpha=0.3)

        # Taux de violation cumule
        for mode, label in MODES.items():
            if mode not in all_dfs:
                continue
            sub = all_dfs[mode][all_dfs[mode]['slice'] == sl].sort_values('simtime_s')
            if sub.empty:
                continue
            viol = (sub['avg_latency_ms'] > sla_lat).astype(float)
            cum_viol = viol.expanding().mean() * 100
            ax_viol.plot(sub['simtime_s'].values, cum_viol.values,
                         label=label, color=colors[mode], linewidth=1.5)

        ax_viol.set_title(f'{sl} — Taux violation cumule (%)')
        ax_viol.set_xlabel('Temps simulation (s)')
        ax_viol.set_ylabel('Viol. rate (%)')
        ax_viol.legend(fontsize=7)
        ax_viol.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = os.path.join(base_dir, 'comparison_plot.png')
    plt.savefig(out_fig, dpi=120)
    print(f"Figure sauvegardee : {out_fig}")

    # ── CSV resume ────────────────────────────────────────────────────────────
    rows = []
    for mode in MODES:
        if mode not in all_kpis:
            continue
        for sl in SLICES:
            if sl not in all_kpis[mode]:
                continue
            row = {'mode': mode, 'slice': sl}
            row.update(all_kpis[mode][sl])
            rows.append(row)
    df_summary = pd.DataFrame(rows)
    out_csv = os.path.join(base_dir, 'comparison_summary.csv')
    df_summary.to_csv(out_csv, index=False)
    print(f"Resume CSV       : {out_csv}")


if __name__ == '__main__':
    main()
