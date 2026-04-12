"""
extract_simu5g_metrics.py
Extraction des métriques simu5G vers CSV pour entraînement LSTM FDL.

Topologie : HetNetSlicing.ned — 3 gNodeBs : Macro (UMa) / Commerce (UMi) / Industrie (UMi)

Colonnes CSV produites :
  Time_Sec, Slice_Type, gNB_id, Scenario,
  Slice_Throughput_Mbps, Slice_Latency_ms, Slice_Jitter_ms,
  Slice_Packet_Loss_pct, Slice_Network_Load_pct,
  SLA_Throughput_Min_Mbps, SLA_Latency_Max_ms, SLA_PacketLoss_Max_pct,
  SLA_OK, SLA_OK_in_15s, SLA_Violated_in_15s

Attribution gNB_id :
  - UEs : suffixe _A / _B / _C dans le nom de module OMNeT++
           → Macro / Commerce / Industrie
  - Serveurs : index de l'app CbrReceiver (app[0]/app[1]/app[2])
           → Macro / Commerce / Industrie

Usage :
    python extract_simu5g_metrics.py <results_dir> <output_csv>
    python extract_simu5g_metrics.py simulation_pfe/results output/all_simu5g.csv
"""

import os
import sys
import csv
import re
from collections import defaultdict
from datetime import datetime, timedelta
import statistics

# ---------------------------------------------------------------------------
# Correspondance app-index serveur → gNB_id
# app[0] = Macro (port série A), app[1] = Commerce (B), app[2] = Industrie (C)
# Défini dans omnetpp_hetnet.ini (3 CbrReceiver par serveur)
# ---------------------------------------------------------------------------
SERVER_APP_TO_GNB = {0: 'Macro', 1: 'Commerce', 2: 'Industrie'}

# ---------------------------------------------------------------------------
# SLA thresholds par slice (simu5G end-to-end, overhead NR ~14 ms inclus)
# Réf : 3GPP TS 23.501 + ajustement end-to-end mesuré
# ---------------------------------------------------------------------------
SLA_THRESHOLDS = {
    'eMBB': {
        'throughput_min_mbps': 10.0,
        'latency_max_ms':      50.0,
        'packet_loss_max_pct':  1.0,
    },
    'URLLC': {
        'throughput_min_mbps':  0.5,
        'latency_max_ms':      20.0,   # 10 ms radio + ~14 ms overhead NR
        'packet_loss_max_pct':  0.001,  # 99.999 % fiabilité
    },
    'mMTC': {
        'throughput_min_mbps':  0.0003,
        'latency_max_ms':       8.8,   # IoT industriel sub-9 ms
        'packet_loss_max_pct':  1.0,
    },
}


# ---------------------------------------------------------------------------
# Extraction du type de slice depuis le nom de module OMNeT++
# Compatible avec les deux topologies (ueEmbb, ueEmbb_A, serverEmbb …)
# ---------------------------------------------------------------------------
def extract_slice_type(module_name: str) -> str | None:
    m = module_name.lower()
    if 'ueembb'   in m or 'serverembb'  in m:
        return 'eMBB'
    if 'ueurllc'  in m or 'serverurllc' in m:
        return 'URLLC'
    if 'uemmtc'   in m or 'servermmtc'  in m:
        return 'mMTC'
    return None


# ---------------------------------------------------------------------------
# Extraction du gNB_id depuis le nom de module OMNeT++
#
# Règles (par ordre de priorité) :
#   1. UE HetNet : _A[ → Macro, _B[ → Commerce, _C[ → Industrie
#   2. Serveur HetNet : app[0] → Macro, app[1] → Commerce, app[2] → Industrie
#   3. gNodeB direct : gnbmacro / gnbcommerce / gnbindustrie dans le chemin
#   4. Fallback legacy (1 gNB) : Macro
# ---------------------------------------------------------------------------
def extract_gnb_id(module_name: str) -> str:
    m = module_name.lower()

    # ── Règle 1 : UE HetNet (ueEmbb_A[0], ueUrllc_B[2], …) ────────────────
    if '_a[' in m:
        return 'Macro'
    if '_b[' in m:
        return 'Commerce'
    if '_c[' in m:
        return 'Industrie'

    # ── Règle 2 : Serveur HetNet — app index → gNB ─────────────────────────
    if 'server' in m:
        hit = re.search(r'\.app\[(\d+)\]', module_name)
        if hit:
            return SERVER_APP_TO_GNB.get(int(hit.group(1)), 'Macro')

    # ── Règle 3 : gNodeB dans le chemin de module ───────────────────────────
    if 'gnbmacro'     in m:
        return 'Macro'
    if 'gnbcommerce'  in m:
        return 'Commerce'
    if 'gnbindustrie' in m:
        return 'Industrie'

    # ── Règle 4 : module non reconnu — skip ────────────────────────────────
    return None


# ---------------------------------------------------------------------------
# Parse d'un fichier .vec OMNeT++ — streaming 2 passes, filtre CBR serveurs
#
# Format OMNeT++ 6.x ETV (Event-Time-Value) :
#   Déclaration : vector <id> <module> <signal>:vector ETV
#   Donnée      : <vecId>\t<eventNo>\t<time>\t<value>
#
# Approche 2 passes pour éviter de charger 500M+ lignes en RAM :
#   Passe 1 : identifier les vecId des CbrReceiver côté serveur
#   Passe 2 : lire UNIQUEMENT les données de ces vecId
#
# Retourne : {vec_id: {'module': str, 'name': str, 'data': [(t, v), …]}}
# ---------------------------------------------------------------------------
def parse_vec_file(filepath: str) -> dict:
    # ── Passe 1 : cartographier les vecteurs CBR serveurs ────────────────
    target_ids: set[str] = set()
    vec_meta:   dict     = {}

    with open(filepath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('vector '):
                continue
            parts = line.split(None, 4)
            if len(parts) < 4:
                continue
            vec_id = parts[1]
            module = parts[2]
            name   = parts[3].split(':')[0]          # retire ':vector' suffix

            # Filtrer : CbrReceiver côté serveur (métriques rx)
            #           + CbrSender côté UE (cbrSentPkt uniquement, pour calcul perte)
            is_server = 'server' in module.lower()
            is_ue_sent = (not is_server) and ('cbrSentPkt' in name)
            if not is_server and not is_ue_sent:
                continue

            slice_type = extract_slice_type(module)
            gnb_id     = extract_gnb_id(module)
            if slice_type is None or gnb_id is None:
                continue

            target_ids.add(vec_id)
            vec_meta[vec_id] = {'module': module, 'name': name, 'data': []}

    if not target_ids:
        return {}

    # ── Passe 2 : charger uniquement les données des vecId cibles ────────
    # Format ETV : vecId \t eventNo \t time \t value
    with open(filepath, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line or not line[0].isdigit():
                continue                              # skip déclarations et attrs
            parts = line.split('\t', 4)
            if len(parts) < 4:
                continue
            vec_id = parts[0]
            if vec_id not in target_ids:
                continue
            try:
                t = float(parts[2])                  # ETV : index 2 = time
                v = float(parts[3])                  # ETV : index 3 = value
                vec_meta[vec_id]['data'].append((t, v))
            except (ValueError, IndexError):
                pass

    return vec_meta


# ---------------------------------------------------------------------------
# Parse d'un fichier .sca OMNeT++
# Retourne : [{'module': str, 'name': str, 'value': float|None}, …]
# ---------------------------------------------------------------------------
def parse_sca_file(filepath: str) -> list:
    scalars = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('scalar '):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        value = float(parts[3]) if parts[3] not in ('nan', 'inf', '-inf') else None
                    except ValueError:
                        value = None
                    scalars.append({'module': parts[1], 'name': parts[2], 'value': value})
    return scalars


# ---------------------------------------------------------------------------
# Agrégation par fenêtre temporelle ET par (slice_type, gNB_id)
#
# Clé d'agrégation : (slice_type, gnb_id)
# Permet une extraction propre par cellule, quel que soit le nombre de gNBs.
# ---------------------------------------------------------------------------
def aggregate_by_time_window(vectors: dict, window_sec: float = 1.0) -> dict:
    """
    Retourne :
      {(slice_type, gnb_id): {time_bucket: {delays, jitters, bytes, packets…}}}
    """
    bucket_proto = lambda: {
        'delays':           [],
        'jitters':          [],
        'throughputs':      [],
        'bytes_received':   0,
        'packets_received': 0,
        'packets_sent':     0,
        'packets_lost':     0,
    }
    aggregated = defaultdict(lambda: defaultdict(bucket_proto))

    for vec_info in vectors.values():
        module     = vec_info['module']
        name       = vec_info['name']
        data       = vec_info['data']
        slice_type = extract_slice_type(module)
        gnb_id     = extract_gnb_id(module)

        if slice_type is None or gnb_id is None or not data:
            continue

        key = (slice_type, gnb_id)

        for t, v in data:
            tb = int(t / window_sec) * window_sec
            b  = aggregated[key][tb]

            if 'cbrFrameDelay' in name:
                b['delays'].append(v * 1000)            # s → ms
            elif 'cbrJitter' in name:
                b['jitters'].append(v * 1000)
            elif 'cbrFrameLoss' in name:
                # cbrFrameLoss est un ratio (double), pas un compteur entier
                # On l'ignore ici — la perte est calculée depuis sent vs received
                pass
            elif 'cbrReceivedBytes' in name:
                b['bytes_received'] += v
                # do NOT count packets here — use cbrRcvdPkt for received count
            elif 'cbrReceivedThroughput' in name:
                b['throughputs'].append(v / 1e6)        # bps → Mbps
            elif 'cbrGeneratedThroughput' in name:
                b['throughputs'].append(v / 1e6)        # fallback sender-side
            elif 'cbrRcvdPkt' in name:
                b['packets_received'] += 1
            elif 'cbrSentPkt' in name:
                b['packets_sent'] += 1

    return aggregated


# ---------------------------------------------------------------------------
# Calcul des métriques SLA à partir des buckets agrégés
# Retourne une liste de records, un par (time, slice_type, gnb_id)
# ---------------------------------------------------------------------------
def calculate_metrics(aggregated: dict, scalars: list,
                      scenario_name: str = 'Unknown') -> list:
    results = []

    # Totaux par (slice_type, gnb_id) depuis les scalaires
    slice_totals = defaultdict(lambda: {'sent': 0, 'received': 0, 'bytes': 0, 'lost': 0})
    for sc in scalars:
        if sc['value'] is None:
            continue
        st  = extract_slice_type(sc['module'])
        gid = extract_gnb_id(sc['module'])
        if st is None:
            continue
        n = sc['name']
        k = (st, gid)
        if   'packetSent:count'               in n or 'cbrSentPkt:count'      in n:
            slice_totals[k]['sent']     += sc['value']
        elif 'packetReceived:count'            in n or 'cbrRcvdPkt:count'     in n:
            slice_totals[k]['received'] += sc['value']
        elif 'packetReceived:sum(packetBytes)' in n:
            slice_totals[k]['bytes']    += sc['value']
        elif 'cbrFrameLoss:sum'               in n:
            slice_totals[k]['lost']     += sc['value']
        # cbrFrameLoss est enregistré en 'mean' (ratio 0-1) par CbrReceiver.ned
        # On le convertit en nombre de paquets perdus via le total reçu
        elif 'cbrFrameLoss:mean'              in n and sc['value'] is not None:
            # loss_ratio = lost / (sent) → on stocke le ratio, converti après
            slice_totals[k]['loss_ratio'] = sc['value']

    prev_latencies: dict[tuple, list] = defaultdict(list)

    for (slice_type, gnb_id), time_data in aggregated.items():
        if slice_type not in SLA_THRESHOLDS:
            continue
        thresholds = SLA_THRESHOLDS[slice_type]

        for t in sorted(time_data):
            b = time_data[t]

            # ── Throughput ────────────────────────────────────────────────────
            if b['bytes_received'] > 0:
                throughput_mbps = (b['bytes_received'] * 8) / 1e6
            elif b['throughputs']:
                throughput_mbps = statistics.mean(b['throughputs'])
            else:
                throughput_mbps = 0.0

            # ── Latence ───────────────────────────────────────────────────────
            key = (slice_type, gnb_id)
            if b['delays']:
                latency_ms     = statistics.mean(b['delays'])
                latency_source = 'measured'
            else:
                latency_ms     = _estimate_latency_mg1(throughput_mbps, slice_type)
                latency_source = 'estimated'
            prev_latencies[key].append(latency_ms)

            # ── Jitter ────────────────────────────────────────────────────────
            if b['jitters']:
                jitter_ms = statistics.mean(b['jitters'])
            else:
                recent = prev_latencies[key][-5:]
                jitter_ms = statistics.stdev(recent) if len(recent) > 1 else 0.0

            # ── Perte paquets ─────────────────────────────────────────────────
            # Mesurée depuis les compteurs cbrSentPkt (UE) vs cbrRcvdPkt (serveur)
            # après correction de la double-comptabilisation dans cbrReceivedBytes.
            # Le buffer RLC UM est maintenant borné (queueSize=15000B/8000B en ini)
            # donc les drops RLC réels sont capturés dans sent vs received.
            sent = b['packets_sent']
            rcvd = b['packets_received']
            if sent > 0:
                pkt_loss_pct = max(0.0, 100.0 * (1.0 - rcvd / sent))
            else:
                pkt_loss_pct = 0.0

            # ── Charge réseau ─────────────────────────────────────────────────
            max_thr       = max(thresholds['throughput_min_mbps'] * 10, 1.0)
            net_load_pct  = min(100.0, (throughput_mbps / max_thr) * 100)

            # ── SLA OK ? ──────────────────────────────────────────────────────
            # Critères : latence + perte de paquets (3GPP TS 28.554)
            # Le throughput est une feature d'entrée LSTM mais pas un critère SLA :
            # en CBR il est redondant avec la perte (throughput = offre × (1-perte))
            # et crée des faux positifs en faible charge (nuit calme).
            sla_ok = (
                latency_ms   <= thresholds['latency_max_ms']      and
                pkt_loss_pct <= thresholds['packet_loss_max_pct']
            )

            results.append({
                'time':            t,
                'slice_type':      slice_type,
                'gnb_id':          gnb_id,
                'scenario':        scenario_name,
                'throughput_mbps': throughput_mbps,
                'latency_ms':      latency_ms,
                'latency_source':  latency_source,
                'jitter_ms':       jitter_ms,
                'pkt_loss_pct':    pkt_loss_pct,
                'net_load_pct':    net_load_pct,
                'sla_ok':          sla_ok,
                'thresholds':      thresholds,
            })

    return results


# ---------------------------------------------------------------------------
# Modèle M/G/1 — fallback si cbrFrameDelay absent du .vec
# (identique à la version précédente)
# ---------------------------------------------------------------------------
def _estimate_latency_mg1(throughput_mbps: float, slice_type: str) -> float:
    import random
    capacities  = {'eMBB': 100.0, 'URLLC': 10.0, 'mMTC': 1.0}
    base_lat    = {'eMBB':   5.0, 'URLLC':  1.0, 'mMTC': 10.0}
    capacity    = capacities.get(slice_type, 50.0)
    base        = base_lat.get(slice_type, 10.0)
    rho         = min(0.99, throughput_mbps / capacity) if capacity > 0 else 0.5
    queuing     = (rho / (1 - rho)) * 0.5 if 0 < rho < 1 else 50.0
    return max(0.1, base + queuing + random.gauss(0, base * 0.1))


# ---------------------------------------------------------------------------
# Ajout des labels SLA_OK_in_15s / SLA_Violated_in_15s
# Trie et parcourt par (slice_type, gnb_id) — essentiel pour HetNet
# ---------------------------------------------------------------------------
def add_sla_predictions(results: list, lookahead_sec: int = 15) -> list:
    # Grouper par (slice_type, gnb_id)
    by_key: dict[tuple, list] = defaultdict(list)
    for r in results:
        by_key[(r['slice_type'], r['gnb_id'])].append(r)

    for key, group in by_key.items():
        group.sort(key=lambda x: x['time'])
        for i, r in enumerate(group):
            t_curr = r['time']
            future = [
                s for s in group
                if t_curr < s['time'] <= t_curr + lookahead_sec
            ]
            sla_ok_future      = all(s['sla_ok'] for s in future) if future else True
            r['sla_ok_in_15s'] = sla_ok_future
            r['sla_violated']  = 0 if sla_ok_future else 1

    return results


# ---------------------------------------------------------------------------
# Écriture CSV
# Le champ gNB_id est en 3e colonne, après Slice_Type
# ---------------------------------------------------------------------------
def write_csv(results: list, output_path: str,
              base_time: datetime | None = None) -> None:
    if base_time is None:
        base_time = datetime(2024, 1, 1, 0, 0, 0)

    fieldnames = [
        'Time_Sec', 'Slice_Type', 'gNB_id', 'Scenario',
        'Slice_Throughput_Mbps', 'Slice_Latency_ms', 'Slice_Jitter_ms',
        'Slice_Packet_Loss_pct', 'Slice_Network_Load_pct',
        'SLA_Throughput_Min_Mbps', 'SLA_Latency_Max_ms', 'SLA_PacketLoss_Max_pct',
        'SLA_OK', 'SLA_OK_in_15s', 'SLA_Violated_in_15s',
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x['time'], x['slice_type'], x['gnb_id'])):
            thr = r['thresholds']
            writer.writerow({
                'Time_Sec':               (base_time + timedelta(seconds=r['time'])).isoformat() + 'Z',
                'Slice_Type':             r['slice_type'],
                'gNB_id':                 r['gnb_id'],
                'Scenario':               r['scenario'],
                'Slice_Throughput_Mbps':  round(r['throughput_mbps'], 6),
                'Slice_Latency_ms':       round(r['latency_ms'],      6),
                'Slice_Jitter_ms':        round(r['jitter_ms'],       6),
                'Slice_Packet_Loss_pct':  round(r['pkt_loss_pct'],    6),
                'Slice_Network_Load_pct': round(r['net_load_pct'],    6),
                'SLA_Throughput_Min_Mbps':thr['throughput_min_mbps'],
                'SLA_Latency_Max_ms':     thr['latency_max_ms'],
                'SLA_PacketLoss_Max_pct': thr['packet_loss_max_pct'],
                'SLA_OK':                 str(r['sla_ok']).lower(),
                'SLA_OK_in_15s':          str(r['sla_ok_in_15s']).lower(),
                'SLA_Violated_in_15s':    r['sla_violated'],
            })


# ---------------------------------------------------------------------------
# Traitement d'un répertoire de résultats
# ---------------------------------------------------------------------------
def process_simulation(results_dir: str, output_csv: str,
                       scenario_filter: str | None = None) -> None:
    all_results = []

    vec_files = [f for f in os.listdir(results_dir) if f.endswith('.vec')]
    sca_files = [f for f in os.listdir(results_dir) if f.endswith('.sca')]

    if scenario_filter:
        vec_files = [f for f in vec_files if f.startswith(scenario_filter)]
        print(f"Filtre scénario : '{scenario_filter}'")

    print(f"Fichiers trouvés : {len(vec_files)} .vec, {len(sca_files)} .sca")

    for vec_file in vec_files:
        config_name = vec_file.replace('.vec', '')
        sca_file    = config_name + '.sca'
        vec_path    = os.path.join(results_dir, vec_file)
        sca_path    = os.path.join(results_dir, sca_file)

        print(f"\nTraitement : {config_name}")
        vectors = parse_vec_file(vec_path)
        print(f"  {len(vectors)} vecteurs parsés")

        scalars = []
        if os.path.exists(sca_path):
            scalars = parse_sca_file(sca_path)
            print(f"  {len(scalars)} scalaires parsés")

        aggregated = aggregate_by_time_window(vectors, window_sec=1.0)
        print(f"  {len(aggregated)} clés (slice, gNB) agrégées :", list(aggregated.keys()))

        # Durée maximale de simulation
        max_time = max(
            (max(t for t, _ in v['data']) for v in vectors.values() if v['data']),
            default=0.0,
        )
        print(f"  Durée simulation : {max_time:.1f} s")

        scenario_name = config_name.rsplit('-', 1)[0] if '-' in config_name else config_name
        results = calculate_metrics(aggregated, scalars, scenario_name)
        results = add_sla_predictions(results, lookahead_sec=15)
        all_results.extend(results)
        print(f"  {len(results)} lignes générées")

    print(f"\nEcriture -> {output_csv}")
    write_csv(all_results, output_csv)

    # ── Résumé par (slice, gNB) ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RÉSUMÉ PAR SLICE × gNB")
    print("=" * 70)
    print(f"{'Slice':7} {'gNB':12} {'N':>6} {'Viol%':>8} {'LatMoy':>8} {'Lat src':>10}")
    print("-" * 70)

    by_key: dict[tuple, list] = defaultdict(list)
    for r in all_results:
        by_key[(r['slice_type'], r['gnb_id'])].append(r)

    for (sl, gid), group in sorted(by_key.items()):
        n        = len(group)
        viol_pct = sum(r['sla_violated'] for r in group) / n * 100
        lat_mean = statistics.mean(r['latency_ms'] for r in group)
        n_meas   = sum(1 for r in group if r.get('latency_source') == 'measured')
        src_info = f"{n_meas}/{n} mesurées"
        print(f"{sl:7} {gid:12} {n:>6} {viol_pct:>7.1f}% {lat_mean:>7.1f}ms  {src_info}")

    print("=" * 70)
    print(f"Total : {len(all_results)} lignes")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) < 3:
        print("Usage : python extract_simu5g_metrics.py <results_dir> <output_csv> [--scenario NOM]")
        sys.exit(1)

    results_dir = sys.argv[1]
    output_csv  = sys.argv[2]

    scenario_filter = None
    if '--scenario' in sys.argv:
        idx = sys.argv.index('--scenario')
        if idx + 1 < len(sys.argv):
            scenario_filter = sys.argv[idx + 1]

    if not os.path.isdir(results_dir):
        print(f"Erreur : {results_dir} n'est pas un répertoire valide")
        sys.exit(1)

    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    process_simulation(results_dir, output_csv, scenario_filter)
    print(f"\nDataset généré : {output_csv}")


if __name__ == '__main__':
    main()
