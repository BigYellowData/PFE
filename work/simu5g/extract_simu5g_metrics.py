"""
Extraction des métriques simu5G vers format compatible avec l'approche M/M/1.

Ce script parse les fichiers .vec et .sca de simu5G et génère un CSV
avec les mêmes colonnes que le dataset slice_sla_dataset :
  - Time_Sec, Slice Type, Slice_Throughput_Mbps, Slice_Jitter_ms,
    Slice_Packet_Loss_pct, Slice_Network_Load_pct, Slice_Latency_ms,
    SLA_Throughput_Min_Mbps, SLA_PacketLoss_Max_pct, SLA_Latency_Max_ms,
    SLA_OK, SLA_OK_in_15s, SLA_Violated_in_15s

Usage:
    python extract_simu5g_metrics.py <results_dir> <output_csv>

Exemple:
    python extract_simu5g_metrics.py simulation_pfe/results output/simu5g_sla_dataset.csv
"""

import os
import sys
import csv
import re
from collections import defaultdict
from datetime import datetime, timedelta
import statistics

# SLA thresholds par slice (adaptes pour simu5G end-to-end)
# Les seuils 3GPP TS 23.501 sont pour l'interface radio uniquement.
# simu5G mesure la latence end-to-end (UE -> serveur) qui inclut ~14ms
# de overhead radio (scheduling, propagation, traitement NR).
SLA_THRESHOLDS = {
    'eMBB': {
        'throughput_min_mbps': 10.0,
        'latency_max_ms': 50.0,
        'packet_loss_max_pct': 1.0
    },
    'URLLC': {
        'throughput_min_mbps': 0.5,     # Adapte : URLLC = petits paquets (292B/30ms)
        'latency_max_ms': 20.0,         # Adapte : 10ms radio + ~14ms overhead NR
        'packet_loss_max_pct': 0.001    # 99.999% reliability
    },
    'mMTC': {
        'throughput_min_mbps': 0.0003,  # Adapte : mMTC = 47B/5s, seuil realiste
        'latency_max_ms': 8.8,          # IoT industriel sub-9ms (latence mediane ~8.5ms)
        'packet_loss_max_pct': 1.0
    },
    'BestEffort': {
        'throughput_min_mbps': 0.0,
        'latency_max_ms': 500.0,
        'packet_loss_max_pct': 5.0
    }
}


def extract_slice_type(module_name):
    """Extrait le type de slice du nom de module OMNeT++."""
    module_lower = module_name.lower()
    if 'ueembb' in module_lower or 'serverembb' in module_lower:
        return 'eMBB'
    elif 'ueurllc' in module_lower or 'serverurllc' in module_lower:
        return 'URLLC'
    elif 'uemmtc' in module_lower or 'servermmtc' in module_lower:
        return 'mMTC'
    elif 'uebesteffort' in module_lower or 'serverbesteffort' in module_lower:
        return 'BestEffort'
    return None


def parse_vec_file(filepath):
    """
    Parse un fichier .vec OMNeT++ et extrait les vecteurs temporels.

    Returns:
        dict: {vector_id: {'module': str, 'name': str, 'data': [(time, value), ...]}}
    """
    vectors = {}

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()

            # Définition de vecteur: vector <id> <module> <name> [<type>]
            if line.startswith('vector '):
                parts = line.split()
                if len(parts) >= 4:
                    vec_id = parts[1]
                    module = parts[2]
                    name = parts[3]
                    vectors[vec_id] = {
                        'module': module,
                        'name': name,
                        'data': []
                    }

            # Données: <vec_id>\t<event_num>\t<time>\t<value>
            elif line and line[0].isdigit() and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 4:
                    vec_id = parts[0]
                    if vec_id in vectors:
                        try:
                            time = float(parts[2])
                            value = float(parts[3])
                            vectors[vec_id]['data'].append((time, value))
                        except (ValueError, IndexError):
                            pass

    return vectors


def parse_sca_file(filepath):
    """
    Parse un fichier .sca OMNeT++ et extrait les scalaires.

    Returns:
        list: [{'module': str, 'name': str, 'value': float}, ...]
    """
    scalars = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('scalar '):
                parts = line.split()
                if len(parts) >= 4:
                    module = parts[1]
                    name = parts[2]
                    try:
                        value = float(parts[3]) if parts[3] not in ['nan', 'inf', '-inf'] else None
                    except:
                        value = None
                    scalars.append({
                        'module': module,
                        'name': name,
                        'value': value
                    })

    return scalars


def aggregate_by_time_window(vectors, window_sec=1.0):
    """
    Agrège les métriques par fenêtre temporelle et par slice.

    Returns:
        dict: {slice_type: {time_bucket: {'delays': [], 'bytes': [], ...}}}
    """
    aggregated = defaultdict(lambda: defaultdict(lambda: {
        'delays': [],
        'jitters': [],
        'throughputs': [],
        'bytes_received': 0,
        'packets_received': 0,
        'packets_sent': 0,
        'packets_lost': 0
    }))

    for vec_id, vec_info in vectors.items():
        module = vec_info['module']
        name = vec_info['name']
        data = vec_info['data']

        slice_type = extract_slice_type(module)
        if not slice_type:
            continue

        for time, value in data:
            time_bucket = int(time / window_sec) * window_sec

            # cbrFrameDelay = CbrReceiver end-to-end delay (seconds -> ms) - PRIORITE
            # rcvdPkLifetime = end-to-end delay (seconds -> ms)
            # rlcDelayDl = RLC layer delay (seconds -> ms)
            # macDelayDl = MAC layer delay (seconds -> ms)
            if 'cbrFrameDelay' in name:
                delay_ms = value * 1000  # sec -> ms
                aggregated[slice_type][time_bucket]['delays'].append(delay_ms)
            elif 'rcvdPkLifetime' in name or 'rlcDelayDl' in name or 'macDelayDl' in name:
                # Fallback si pas de cbrFrameDelay
                if not aggregated[slice_type][time_bucket]['delays']:
                    delay_ms = value * 1000  # sec -> ms
                    aggregated[slice_type][time_bucket]['delays'].append(delay_ms)

            # cbrJitter = CbrReceiver jitter (seconds -> ms)
            elif 'cbrJitter' in name:
                jitter_ms = value * 1000  # sec -> ms
                aggregated[slice_type][time_bucket]['jitters'].append(jitter_ms)

            # cbrFrameLoss = CbrReceiver packet loss count
            elif 'cbrFrameLoss' in name:
                aggregated[slice_type][time_bucket]['packets_lost'] += int(value)

            # cbrReceivedBytes = bytes de chaque paquet recu par CbrReceiver
            # C'est la source principale pour calculer le throughput
            elif 'cbrReceivedBytes' in name:
                aggregated[slice_type][time_bucket]['bytes_received'] += value
                aggregated[slice_type][time_bucket]['packets_received'] += 1

            # cbrGeneratedThroughput ou throughput:vector (bits per second -> Mbps)
            elif ('cbrGeneratedThroughput' in name or name == 'throughput:vector') and '.app[' in module:
                throughput_mbps = value / 1e6  # bps -> Mbps
                aggregated[slice_type][time_bucket]['throughputs'].append(throughput_mbps)

            # packetReceived:vector(packetBytes)
            elif 'packetReceived' in name and 'packetBytes' in name:
                aggregated[slice_type][time_bucket]['bytes_received'] += value
                aggregated[slice_type][time_bucket]['packets_received'] += 1

            # cbrRcvdPkt = received packet count from CbrReceiver
            elif 'cbrRcvdPkt' in name:
                aggregated[slice_type][time_bucket]['packets_received'] += 1

            # packetSent or cbrSentPkt
            elif 'packetSent' in name and 'packetBytes' in name:
                aggregated[slice_type][time_bucket]['packets_sent'] += 1
            elif 'cbrSentPkt' in name:
                aggregated[slice_type][time_bucket]['packets_sent'] += 1

    return aggregated


def estimate_latency_mg1(throughput_mbps, slice_type, capacity_mbps=None):
    """
    Estime la latence avec un modèle M/G/1 basé sur le throughput.

    Paramètres de capacité par slice (basés sur 5G NR):
    - eMBB: 100 Mbps par UE
    - URLLC: 10 Mbps par UE (mais priorité haute)
    - mMTC: 1 Mbps par UE
    - BestEffort: 50 Mbps par UE
    """
    import random

    # Capacités par slice (Mbps)
    capacities = {
        'eMBB': capacity_mbps or 100.0,
        'URLLC': capacity_mbps or 10.0,
        'mMTC': capacity_mbps or 1.0,
        'BestEffort': capacity_mbps or 50.0
    }

    # Latences de base par slice (ms) - propagation + processing
    base_latency = {
        'eMBB': 5.0,
        'URLLC': 1.0,
        'mMTC': 10.0,
        'BestEffort': 20.0
    }

    capacity = capacities.get(slice_type, 50.0)
    base = base_latency.get(slice_type, 10.0)

    # Calcul de la charge (rho = arrival_rate / service_rate)
    rho = min(0.99, throughput_mbps / capacity) if capacity > 0 else 0.5

    # M/G/1: E[W] = (rho / (1 - rho)) * E[S] / 2
    # Où E[S] est le temps de service moyen
    service_time = 1.0  # ms pour 1 paquet

    if rho > 0 and rho < 1:
        queuing_delay = (rho / (1 - rho)) * service_time / 2
    else:
        queuing_delay = 50.0  # Saturation

    # Ajouter du bruit gaussien pour variabilité
    noise = random.gauss(0, base * 0.1)

    latency = base + queuing_delay + noise
    return max(0.1, latency)


def calculate_metrics(aggregated, scalars, sim_duration, scenario_name='Unknown'):
    """
    Calcule les métriques SLA à partir des données agrégées.
    Utilise les métriques CbrReceiver (cbrFrameDelay, cbrJitter, cbrFrameLoss) si disponibles.
    Sinon, utilise un modèle M/G/1 pour estimer la latence.

    Returns:
        list: [{'time': float, 'slice': str, 'throughput': float, ...}, ...]
    """
    results = []

    # Calculer les totaux par slice depuis les scalaires
    slice_totals = defaultdict(lambda: {'sent': 0, 'received': 0, 'bytes': 0, 'lost': 0})
    for scalar in scalars:
        slice_type = extract_slice_type(scalar['module'])
        if not slice_type or scalar['value'] is None:
            continue

        name = scalar['name']
        if 'packetSent:count' in name or 'cbrSentPkt:count' in name:
            slice_totals[slice_type]['sent'] += scalar['value']
        elif 'packetReceived:count' in name or 'cbrRcvdPkt:count' in name:
            slice_totals[slice_type]['received'] += scalar['value']
        elif 'packetReceived:sum(packetBytes)' in name:
            slice_totals[slice_type]['bytes'] += scalar['value']
        elif 'cbrFrameLoss:sum' in name:
            slice_totals[slice_type]['lost'] += scalar['value']

    # Générer les métriques par seconde
    for slice_type, time_data in aggregated.items():
        if slice_type not in SLA_THRESHOLDS:
            continue

        thresholds = SLA_THRESHOLDS[slice_type]
        sorted_times = sorted(time_data.keys())

        prev_latencies = []
        cumulative_sent = 0
        cumulative_lost = 0

        for t in sorted_times:
            bucket = time_data[t]

            # Throughput (Mbps) - préférer cbrReceivedBytes (données par paquet)
            if bucket['bytes_received'] > 0:
                throughput_mbps = (bucket['bytes_received'] * 8) / 1e6  # bytes/sec -> Mbps
            elif bucket['throughputs']:
                throughput_mbps = statistics.mean(bucket['throughputs'])
            else:
                throughput_mbps = 0.0

            # Latence - utiliser cbrFrameDelay si disponible, sinon estimer avec M/G/1
            if bucket['delays']:
                latency_ms = statistics.mean(bucket['delays'])
                latency_source = 'measured'
            else:
                # Estimer avec M/G/1 basé sur le throughput
                latency_ms = estimate_latency_mg1(throughput_mbps, slice_type)
                latency_source = 'estimated'

            prev_latencies.append(latency_ms)

            # Jitter - utiliser cbrJitter si disponible, sinon calculer depuis latences
            if bucket['jitters']:
                jitter_ms = statistics.mean(bucket['jitters'])
            else:
                # Fallback: écart-type des latences récentes (dernières 5 sec)
                recent_latencies = prev_latencies[-5:] if len(prev_latencies) > 1 else [latency_ms]
                jitter_ms = statistics.stdev(recent_latencies) if len(recent_latencies) > 1 else 0.0

            # Packet loss - utiliser cbrFrameLoss si disponible, sinon estimer
            cumulative_sent += bucket['packets_sent']
            cumulative_lost += bucket['packets_lost']

            if bucket['packets_lost'] > 0 or bucket['packets_sent'] > 0:
                # Utiliser les vraies valeurs de CbrReceiver
                if bucket['packets_sent'] > 0:
                    packet_loss_pct = (bucket['packets_lost'] / bucket['packets_sent']) * 100
                else:
                    packet_loss_pct = 0.0
            else:
                # Fallback: estimer basé sur la charge
                max_capacity = max(thresholds['throughput_min_mbps'] * 10, 1.0)
                rho = min(1.0, throughput_mbps / max_capacity)
                if rho > 0.9:
                    packet_loss_pct = (rho - 0.9) * 50  # 0-5% loss quand charge > 90%
                else:
                    packet_loss_pct = 0.0

            # Network load (basé sur throughput vs capacité)
            max_throughput = max(thresholds['throughput_min_mbps'] * 10, 1.0)
            network_load_pct = min(100, (throughput_mbps / max_throughput) * 100)

            # SLA OK? (vérifier chaque critère)
            throughput_ok = throughput_mbps >= thresholds['throughput_min_mbps']
            latency_ok = latency_ms <= thresholds['latency_max_ms']
            loss_ok = packet_loss_pct <= thresholds['packet_loss_max_pct']

            sla_ok = throughput_ok and latency_ok and loss_ok

            results.append({
                'time': t,
                'slice_type': slice_type,
                'scenario': scenario_name,
                'throughput_mbps': throughput_mbps,
                'latency_ms': latency_ms,
                'latency_source': latency_source,
                'jitter_ms': jitter_ms,
                'packet_loss_pct': packet_loss_pct,
                'network_load_pct': network_load_pct,
                'sla_ok': sla_ok,
                'thresholds': thresholds
            })

    return results


def add_sla_predictions(results, lookahead_sec=15):
    """Ajoute les colonnes SLA_OK_in_15s et SLA_Violated_in_15s."""
    # Trier par slice et temps
    results.sort(key=lambda x: (x['slice_type'], x['time']))

    # Pour chaque slice, regarder si SLA sera violé dans les 15 prochaines secondes
    by_slice = defaultdict(list)
    for r in results:
        by_slice[r['slice_type']].append(r)

    for slice_type, slice_results in by_slice.items():
        for i, r in enumerate(slice_results):
            current_time = r['time']
            future_window = [
                sr for sr in slice_results
                if current_time < sr['time'] <= current_time + lookahead_sec
            ]

            # SLA_OK_in_15s = tous les SLA OK dans les 15 prochaines sec
            sla_ok_future = all(sr['sla_ok'] for sr in future_window) if future_window else True

            # SLA_Violated_in_15s = au moins une violation dans les 15 prochaines sec
            sla_violated_future = not sla_ok_future

            r['sla_ok_in_15s'] = sla_ok_future
            r['sla_violated_in_15s'] = 1 if sla_violated_future else 0

    return results


def write_csv(results, output_path, base_time=None):
    """Écrit les résultats au format CSV compatible avec le dataset M/M/1."""
    if base_time is None:
        base_time = datetime(2024, 1, 1, 0, 0, 0)

    fieldnames = [
        'Time_Sec', 'Slice Type', 'Scenario',
        'Slice_Throughput_Mbps', 'Slice_Jitter_ms',
        'Slice_Packet_Loss_pct', 'Slice_Network_Load_pct', 'Slice_Latency_ms',
        'SLA_Throughput_Min_Mbps', 'SLA_PacketLoss_Max_pct', 'SLA_Latency_Max_ms',
        'SLA_OK', 'SLA_OK_in_15s', 'SLA_Violated_in_15s'
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in sorted(results, key=lambda x: (x['time'], x['slice_type'])):
            # Convertir slice type pour compatibilité
            slice_display = r['slice_type']
            if slice_display == 'BestEffort':
                slice_display = 'Best Effort'

            row = {
                'Time_Sec': (base_time + timedelta(seconds=r['time'])).isoformat() + 'Z',
                'Slice Type': slice_display,
                'Scenario': r.get('scenario', 'Unknown'),
                'Slice_Throughput_Mbps': round(r['throughput_mbps'], 6),
                'Slice_Jitter_ms': round(r['jitter_ms'], 6),
                'Slice_Packet_Loss_pct': round(r['packet_loss_pct'], 6),
                'Slice_Network_Load_pct': round(r['network_load_pct'], 6),
                'Slice_Latency_ms': round(r['latency_ms'], 6),
                'SLA_Throughput_Min_Mbps': r['thresholds']['throughput_min_mbps'],
                'SLA_PacketLoss_Max_pct': r['thresholds']['packet_loss_max_pct'],
                'SLA_Latency_Max_ms': r['thresholds']['latency_max_ms'],
                'SLA_OK': str(r['sla_ok']).lower(),
                'SLA_OK_in_15s': str(r['sla_ok_in_15s']).lower(),
                'SLA_Violated_in_15s': r['sla_violated_in_15s']
            }
            writer.writerow(row)


def process_simulation(results_dir, output_csv):
    """Traite tous les fichiers de résultats d'un répertoire."""
    all_results = []

    # Trouver tous les fichiers .vec et .sca
    vec_files = [f for f in os.listdir(results_dir) if f.endswith('.vec')]
    sca_files = [f for f in os.listdir(results_dir) if f.endswith('.sca')]

    print(f"Fichiers trouvés: {len(vec_files)} .vec, {len(sca_files)} .sca")

    for vec_file in vec_files:
        config_name = vec_file.replace('.vec', '')
        sca_file = config_name + '.sca'

        vec_path = os.path.join(results_dir, vec_file)
        sca_path = os.path.join(results_dir, sca_file)

        print(f"\nTraitement de {config_name}...")

        # Parser les fichiers
        print("  Parsing .vec...")
        vectors = parse_vec_file(vec_path)
        print(f"  -> {len(vectors)} vecteurs")

        # Debug: afficher quelques vecteurs avec données
        print("  Vecteurs avec données (top 10):")
        count = 0
        for vec_id, vec_info in vectors.items():
            if vec_info['data'] and count < 10:
                slice_type = extract_slice_type(vec_info['module'])
                print(f"    [{vec_id}] {vec_info['module']} - {vec_info['name']} "
                      f"({len(vec_info['data'])} pts, slice={slice_type})")
                count += 1

        scalars = []
        if os.path.exists(sca_path):
            print("  Parsing .sca...")
            scalars = parse_sca_file(sca_path)
            print(f"  -> {len(scalars)} scalaires")

        # Agréger par fenêtre temporelle
        print("  Agrégation par seconde...")
        aggregated = aggregate_by_time_window(vectors, window_sec=1.0)

        # Debug: afficher les slices agrégés
        print("  Slices agrégés:")
        for slice_type, time_data in aggregated.items():
            n_buckets = len(time_data)
            total_delays = sum(len(b['delays']) for b in time_data.values())
            total_throughputs = sum(len(b['throughputs']) for b in time_data.values())
            print(f"    {slice_type}: {n_buckets} buckets, {total_delays} delays, {total_throughputs} throughputs")

        # Estimer la durée de simulation
        max_time = 0
        for vec_info in vectors.values():
            if vec_info['data']:
                max_time = max(max_time, max(t for t, v in vec_info['data']))

        print(f"  Durée simulation: {max_time:.1f}s")

        # Extraire le nom du scenario (ex: "Normal-0" -> "Normal")
        scenario_name = config_name.rsplit('-', 1)[0] if '-' in config_name else config_name

        # Calculer les métriques
        print("  Calcul des métriques SLA...")
        results = calculate_metrics(aggregated, scalars, max_time, scenario_name=scenario_name)

        # Ajouter prédictions SLA
        print("  Ajout des prédictions SLA...")
        results = add_sla_predictions(results, lookahead_sec=15)

        all_results.extend(results)
        print(f"  -> {len(results)} lignes générées")

    # Écrire le CSV final
    print(f"\nÉcriture de {output_csv}...")
    write_csv(all_results, output_csv)

    # Statistiques finales
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"Total lignes: {len(all_results)}")

    by_slice = defaultdict(list)
    for r in all_results:
        by_slice[r['slice_type']].append(r)

    for slice_type, slice_results in sorted(by_slice.items()):
        violations = sum(1 for r in slice_results if r['sla_violated_in_15s'])
        pct = violations / len(slice_results) * 100 if slice_results else 0
        # Compter les latences mesurées vs estimées
        measured = sum(1 for r in slice_results if r.get('latency_source') == 'measured')
        estimated = sum(1 for r in slice_results if r.get('latency_source') == 'estimated')
        latency_info = f"latency: {measured} measured, {estimated} estimated"
        print(f"  {slice_type:12s}: {len(slice_results):5d} lignes, {violations:4d} violations ({pct:.1f}%) [{latency_info}]")

    print("="*60)


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_simu5g_metrics.py <results_dir> <output_csv>")
        print("\nExemple:")
        print("  python extract_simu5g_metrics.py simulation_pfe/results output/simu5g_dataset.csv")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_csv = sys.argv[2]

    if not os.path.isdir(results_dir):
        print(f"Erreur: {results_dir} n'est pas un répertoire valide")
        sys.exit(1)

    # Créer le répertoire de sortie si nécessaire
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    process_simulation(results_dir, output_csv)
    print(f"\nDataset généré: {output_csv}")


if __name__ == "__main__":
    main()
