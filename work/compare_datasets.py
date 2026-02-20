"""
Analyse comparative entre les donnees de simulation simu5G
et les donnees analytiques (modele M/M/1)
"""

import pandas as pd
import numpy as np

# Charger les deux datasets
print("=" * 60)
print("ANALYSE COMPARATIVE DES DATASETS")
print("=" * 60)

# Dataset simulation simu5G
df_simu = pd.read_csv("simu5g/output/all_simu5g.csv")
print(f"\n[SIMU] Dataset Simulation (simu5G): {len(df_simu)} lignes")

# Dataset analytique M/M/1
df_analytical = pd.read_csv("output/slice_sla_dataset/part-00000-96521db4-af02-4df5-82b2-b5cac9821557-c000.csv")
print(f"[ANAL] Dataset Analytique (M/M/1): {len(df_analytical)} lignes")

# Colonnes communes pour comparaison
metrics = ['Slice_Throughput_Mbps', 'Slice_Jitter_ms', 'Slice_Packet_Loss_pct',
           'Slice_Network_Load_pct', 'Slice_Latency_ms']

print("\n" + "=" * 60)
print("1. TYPES DE SLICES")
print("=" * 60)

print("\n>> Simulation simu5G:")
print(df_simu['Slice Type'].value_counts())

print("\n>> Analytique M/M/1:")
print(df_analytical['Slice Type'].value_counts())

print("\n" + "=" * 60)
print("2. STATISTIQUES PAR METRIQUE (GLOBAL)")
print("=" * 60)

for metric in metrics:
    print(f"\n>> {metric}:")
    print(f"   Simulation  -> min: {df_simu[metric].min():.4f}, max: {df_simu[metric].max():.4f}, mean: {df_simu[metric].mean():.4f}, std: {df_simu[metric].std():.4f}")
    print(f"   Analytique  -> min: {df_analytical[metric].min():.4f}, max: {df_analytical[metric].max():.4f}, mean: {df_analytical[metric].mean():.4f}, std: {df_analytical[metric].std():.4f}")

print("\n" + "=" * 60)
print("3. STATISTIQUES PAR SLICE TYPE")
print("=" * 60)

# Slices communes
common_slices = set(df_simu['Slice Type'].unique()) & set(df_analytical['Slice Type'].unique())
print(f"\nSlices communes: {common_slices}")

for slice_type in sorted(common_slices):
    print(f"\n{'='*40}")
    print(f">> SLICE: {slice_type}")
    print(f"{'='*40}")

    simu_slice = df_simu[df_simu['Slice Type'] == slice_type]
    anal_slice = df_analytical[df_analytical['Slice Type'] == slice_type]

    print(f"\n   Nombre d'entrees: Simu={len(simu_slice)}, Analytique={len(anal_slice)}")

    for metric in metrics:
        simu_mean = simu_slice[metric].mean()
        anal_mean = anal_slice[metric].mean()
        diff_pct = ((simu_mean - anal_mean) / anal_mean * 100) if anal_mean != 0 else float('inf')

        print(f"\n   {metric}:")
        print(f"      Simulation:  mean={simu_mean:.4f}, std={simu_slice[metric].std():.4f}")
        print(f"      Analytique:  mean={anal_mean:.4f}, std={anal_slice[metric].std():.4f}")
        print(f"      Difference:  {diff_pct:+.2f}%")

print("\n" + "=" * 60)
print("4. VIOLATIONS SLA")
print("=" * 60)

print("\n>> Taux de violation SLA (SLA_Violated_in_15s=1):")

simu_violation_rate = df_simu['SLA_Violated_in_15s'].mean() * 100
anal_violation_rate = df_analytical['SLA_Violated_in_15s'].mean() * 100

print(f"   Simulation:  {simu_violation_rate:.2f}%")
print(f"   Analytique:  {anal_violation_rate:.2f}%")

print("\n>> Par type de slice:")
for slice_type in sorted(common_slices):
    simu_slice = df_simu[df_simu['Slice Type'] == slice_type]
    anal_slice = df_analytical[df_analytical['Slice Type'] == slice_type]

    simu_viol = simu_slice['SLA_Violated_in_15s'].mean() * 100
    anal_viol = anal_slice['SLA_Violated_in_15s'].mean() * 100

    print(f"   {slice_type}: Simu={simu_viol:.2f}%, Analytique={anal_viol:.2f}%")

print("\n" + "=" * 60)
print("5. DISTRIBUTION DE LA LATENCE (metrique cle)")
print("=" * 60)

print("\n>> Percentiles de latence (ms):")
percentiles = [25, 50, 75, 90, 95, 99]

print("\n   Simulation simu5G:")
for p in percentiles:
    val = np.percentile(df_simu['Slice_Latency_ms'], p)
    print(f"      P{p}: {val:.2f} ms")

print("\n   Analytique M/M/1:")
for p in percentiles:
    val = np.percentile(df_analytical['Slice_Latency_ms'], p)
    print(f"      P{p}: {val:.2f} ms")

print("\n" + "=" * 60)
print("6. RESUME DES DIFFERENCES CLES")
print("=" * 60)

print("""
Observations principales:
""")

# Calculer les differences moyennes
latency_diff = df_simu['Slice_Latency_ms'].mean() - df_analytical['Slice_Latency_ms'].mean()
throughput_diff = df_simu['Slice_Throughput_Mbps'].mean() - df_analytical['Slice_Throughput_Mbps'].mean()
jitter_diff = df_simu['Slice_Jitter_ms'].mean() - df_analytical['Slice_Jitter_ms'].mean()

print(f"   * Latence moyenne: Simu={df_simu['Slice_Latency_ms'].mean():.2f}ms vs Analytique={df_analytical['Slice_Latency_ms'].mean():.2f}ms (diff: {latency_diff:+.2f}ms)")
print(f"   * Throughput moyen: Simu={df_simu['Slice_Throughput_Mbps'].mean():.4f}Mbps vs Analytique={df_analytical['Slice_Throughput_Mbps'].mean():.4f}Mbps (diff: {throughput_diff:+.4f}Mbps)")
print(f"   * Jitter moyen: Simu={df_simu['Slice_Jitter_ms'].mean():.4f}ms vs Analytique={df_analytical['Slice_Jitter_ms'].mean():.4f}ms (diff: {jitter_diff:+.4f}ms)")
print(f"   * Violation SLA: Simu={simu_violation_rate:.2f}% vs Analytique={anal_violation_rate:.2f}%")

print("\n" + "=" * 60)
print("ANALYSE TERMINEE")
print("=" * 60)
