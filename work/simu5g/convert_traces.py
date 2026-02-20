"""
Conversion des traces de trafic réel (output Spark) vers le format simu5G.

simu5G attend des fichiers de trace au format :
    <offset_seconds> <packet_size_bytes>

Ce script produit un fichier par (Slice Type, flux source-destination),
regroupé dans des sous-dossiers par slice :
    traces/eMBB/flow_0.txt
    traces/URLLC/flow_0.txt
    traces/mMTC/flow_0.txt
    traces/BestEffort/flow_0.txt

Usage :
    python convert_traces.py <input_csv> <output_dir>

Exemple :
    python convert_traces.py ../output/processed_traffic_csv/part-00000-*.csv ./traces
"""
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime


def parse_timestamp(ts_str):
    """Parse ISO timestamp (2022-09-27T13:16:17.000Z) to epoch seconds."""
    ts_str = ts_str.rstrip("Z")
    dt = datetime.fromisoformat(ts_str)
    return dt.timestamp()


def throughput_to_bytes(throughput_mbps):
    """Convert throughput (Mbps) to bytes per second."""
    return throughput_mbps * 1_000_000 / 8


def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_traces.py <input_csv> <output_dir>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_dir = sys.argv[2]

    # Lecture du CSV et regroupement par (Slice Type, Source, Destination)
    flows = defaultdict(list)

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slice_type = row["Slice Type"].replace(" ", "")
            src = row["Source"]
            dst = row["Destination"]
            protocol = row["Protocol_Norm"]
            src_port = row["Src_Port"]
            dst_port = row["Dst_Port"]

            key = (slice_type, src, dst, protocol, src_port, dst_port)
            flows[key].append({
                "time": row["Time_Sec"],
                "throughput_mbps": float(row["Throughput_Mbps"]),
            })

    print(f"Nombre de flux distincts : {len(flows)}")

    # Pour chaque flux, écrire un fichier trace
    slice_flow_counts = defaultdict(int)

    for (slice_type, src, dst, proto, sp, dp), records in flows.items():
        # Trier par timestamp
        records.sort(key=lambda r: r["time"])

        # Calcul du t0 (premier paquet du flux)
        t0 = parse_timestamp(records[0]["time"])

        # Créer le dossier du slice
        slice_dir = os.path.join(output_dir, slice_type)
        os.makedirs(slice_dir, exist_ok=True)

        flow_id = slice_flow_counts[slice_type]
        slice_flow_counts[slice_type] += 1

        trace_file = os.path.join(slice_dir, f"flow_{flow_id}.txt")

        with open(trace_file, "w", encoding="utf-8") as out:
            # Header commenté pour documentation
            out.write(f"# Slice: {slice_type} | {src}:{sp} -> {dst}:{dp} ({proto})\n")
            out.write("# offset_sec  packet_size_bytes\n")

            for rec in records:
                t = parse_timestamp(rec["time"])
                offset = t - t0
                pkt_bytes = int(throughput_to_bytes(rec["throughput_mbps"]))
                # Minimum 64 bytes (trame Ethernet minimum)
                pkt_bytes = max(pkt_bytes, 64)
                out.write(f"{offset:.3f} {pkt_bytes}\n")

        print(f"  [{slice_type}] flow_{flow_id}: {src}:{sp}->{dst}:{dp} "
              f"({len(records)} samples) -> {trace_file}")

    print(f"\nRésumé :")
    for st, count in sorted(slice_flow_counts.items()):
        print(f"  {st}: {count} flux")
    print(f"Traces écrites dans : {output_dir}")


if __name__ == "__main__":
    main()
