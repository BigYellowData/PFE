import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("5G_Traffic_Analysis_GeForceNOW") \
        .getOrCreate()

    # Set log level to warn to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    input_dir = "/home/data/Game_Streaming/GeForce_Now"

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(input_dir)
    )

    # ==========================================
    # 1. Pre-processing & Extraction
    # ==========================================

    # Cast Time to Timestamp and create Time_Sec
    df = df.withColumn("Time_Timestamp", F.to_timestamp(F.col("Time"))) \
           .withColumn("Time_Sec", F.date_trunc("second", F.col("Time_Timestamp")))

    # Extract Ports from 'Info' column using Regex
    df = df.withColumn("Src_Port", F.regexp_extract(F.col("Info"), r"^(\d+)\s+>\s+(\d+)", 1).cast("int")) \
           .withColumn("Dst_Port", F.regexp_extract(F.col("Info"), r"^(\d+)\s+>\s+(\d+)", 2).cast("int"))

    # Normalize Protocol (TLS/SSL → TCP, QUIC → UDP)
    df = df.withColumn(
        "Protocol_Norm",
        F.when(F.col("Protocol").rlike("(?i)TLS|SSL"), "TCP")
         .when(F.col("Protocol").rlike("(?i)QUIC"), "UDP")
         .otherwise(F.col("Protocol"))
    )

    # Detect Packet Loss (Retransmission or Dup ACK)
    df = df.withColumn(
        "Is_Loss",
        F.when(F.col("Info").rlike("Retransmission|Dup ACK"), 1).otherwise(0)
    )

    # ==========================================
    # 2. Jitter Calculation (Window Function)
    # ==========================================

    window_spec = Window.partitionBy(
        "Source", "Destination", "Protocol_Norm", "Src_Port", "Dst_Port"
    ).orderBy("Time_Timestamp")

    df = df.withColumn("Prev_Time", F.lag("Time_Timestamp").over(window_spec))

    df = df.withColumn(
        "Time_Diff",
        F.col("Time_Timestamp").cast("double") - F.col("Prev_Time").cast("double")
    )

    # ==========================================
    # 3. Aggregation par flux (5-tuple) et seconde
    # ==========================================

    group_cols = ["Time_Sec", "Source", "Destination", "Protocol_Norm", "Src_Port", "Dst_Port"]

    df_agg = df.groupBy(group_cols).agg(
        F.sum("Length").alias("Sum_Length"),
        F.count("*").alias("Count_Total_Packets"),
        F.max("Length").alias("Peak_Length"),
        F.sum("Is_Loss").alias("Count_Retransmissions"),
        F.stddev("Time_Diff").alias("Jitter_Raw")  # stddev inter-arrival
    )

    df_agg = df_agg.fillna(0, subset=["Jitter_Raw"])

    # ==========================================
    # 4. Classification & Metrics
    # ==========================================

    # Débit par seconde (par flux)
    df_final = df_agg.withColumn("Throughput_Mbps", (F.col("Sum_Length") * 8) / 1_000_000.0)

    # Device type : Mobile
    df_final = df_final.withColumn("Device Type", F.lit("Mobile"))

    # TRAFFIC TYPE (basé sur le papier GFN)
    is_tcp = F.col("Protocol_Norm") == "TCP"
    is_udp = F.col("Protocol_Norm") == "UDP"

    # Flots de management / admin
    is_port_322 = (F.col("Src_Port") == 322) | (F.col("Dst_Port") == 322)
    is_port_443 = (F.col("Src_Port") == 443) | (F.col("Dst_Port") == 443)

    # Ports GFN gameplay côté client (d’après le papier / NVIDIA)
    is_port_49003 = (F.col("Src_Port") == 49003) | (F.col("Dst_Port") == 49003)  # audio down
    is_port_49004 = (F.col("Src_Port") == 49004) | (F.col("Dst_Port") == 49004)  # audio up
    is_port_49005 = (F.col("Src_Port") == 49005) | (F.col("Dst_Port") == 49005)  # vidéo down
    is_port_49006 = (F.col("Src_Port") == 49006) | (F.col("Dst_Port") == 49006)  # user input

    # Classification fonctionnelle des flux
    df_final = df_final.withColumn(
        "Traffic Type",
        F.when(is_tcp & is_port_322, "Game Management")
         .when(is_tcp & is_port_443, "Platform Admin")
         .when(is_udp & is_port_49005, "Downstream Video")
         .when(is_udp & (is_port_49003 | is_port_49004), "Audio")
         .when(is_udp & is_port_49006, "User Input")
         .otherwise("Unknown")
    )

    # SLICE TYPE (mappé à partir du Traffic Type)
    df_final = df_final.withColumn(
        "Slice Type",
        F.when(F.col("Traffic Type") == "Downstream Video", "eMBB")
         .when(F.col("Traffic Type").isin("User Input", "Game Management", "Audio"), "URLLC")
         .when(F.col("Traffic Type") == "Platform Admin", "mMTC")
         .otherwise("Best Effort")
    )

    # ---------------------------------------------------------
    # 5. CAPACITÉ PAR SLICE & SIMULATION PHYSIQUE (M/M/1)
    # ---------------------------------------------------------
    # Capacités allouées par slice (Mbps) – valeurs réalistes 3GPP
    # Ces valeurs représentent le budget de ressources radio par slice.
    # eMBB  : gros débit (vidéo HD cloud gaming)
    # URLLC : faible volume mais priorité latence
    # mMTC  : signalisation / admin, faible volume

    df_final.cache()

    # Capacité par slice (Mbps)
    df_final = df_final.withColumn(
        "Slice_Capacity_Mbps",
        F.when(F.col("Slice Type") == "eMBB",        F.lit(50.0))
         .when(F.col("Slice Type") == "URLLC",       F.lit(10.0))
         .when(F.col("Slice Type") == "mMTC",        F.lit(5.0))
         .otherwise(F.lit(10.0))  # Best Effort
    )

    # Network Load = throughput / capacité du slice (pas global)
    df_final = df_final.withColumn(
        "Network Load (%)",
        F.least(
            (F.col("Throughput_Mbps") / F.col("Slice_Capacity_Mbps")) * 100,
            F.lit(100.0)
        )
    )
    # Charge normalisée [0, 1] pour les modèles physiques
    df_final = df_final.withColumn(
        "_rho",
        F.least(F.col("Network Load (%)") / 100.0, F.lit(0.99))
    )

    # --- Packet Loss réel (retransmissions observées) ---
    df_final = df_final.withColumn(
        "Packet Loss (%)",
        (F.col("Count_Retransmissions") / F.col("Count_Total_Packets")) * 100
    )
    df_final = df_final.withColumn(
        "Packet Loss (%)",
        F.when(F.col("Packet Loss (%)") > 100, 100.0).otherwise(F.col("Packet Loss (%)"))
    )

    # --- Jitter réel (stddev inter-arrivée, en ms) ---
    df_final = df_final.withColumn("Jitter (ms)", F.col("Jitter_Raw") * 1000)

    # Exigences théoriques (3GPP TS 23.501) – features du modèle
    df_final = df_final.withColumn(
        "Data Rate Requirement (Mbps)",
        F.when(F.col("Slice Type") == "eMBB",  F.lit(10.0))
         .when(F.col("Slice Type") == "URLLC", F.lit(0.1))
         .when(F.col("Slice Type") == "mMTC",  F.lit(0.01))
         .otherwise(F.lit(0.0))
    )

    df_final = df_final.withColumn(
        "Latency Requirement (ms)",
        F.when(F.col("Slice Type") == "URLLC", F.lit(10))
         .when(F.col("Slice Type") == "eMBB",  F.lit(50))
         .when(F.col("Slice Type") == "mMTC",  F.lit(100))
         .otherwise(F.lit(100))
    )

    # --- Latence simulée : modèle M/M/1 ---
    # latence = base_latency / (1 - rho)  
    # Quand la charge (rho) augmente, la latence explose (file d'attente)
    # base_latency par slice reflète la priorité de scheduling de l'opérateur
    df_final = df_final.withColumn(
        "_base_latency_ms",
        F.when(F.col("Slice Type") == "URLLC", F.lit(1.0))   # priorité max
         .when(F.col("Slice Type") == "eMBB",  F.lit(5.0))   # priorité moyenne
         .when(F.col("Slice Type") == "mMTC",  F.lit(10.0))  # basse priorité
         .otherwise(F.lit(8.0))
    )
    df_final = df_final.withColumn(
        "Slice Latency (ms)",
        F.col("_base_latency_ms") / (F.lit(1.0) - F.col("_rho"))
    )


    df_final = df_final.withColumn("Optimal Network Slice", F.col("Slice Type")) \
                       .withColumn("Slice Bandwidth (Mbps)", F.col("Throughput_Mbps"))

    # Nettoyage des colonnes internes
    df_final = df_final.drop("_rho", "_base_latency_ms", "Slice_Capacity_Mbps")

    # ==========================================
    # 6. Agrégation par slice + SLA + cible t+15 s
    # ==========================================

    df_slice = df_final.groupBy("Time_Sec", "Slice Type").agg(
        F.sum("Throughput_Mbps").alias("Slice_Throughput_Mbps"),
        F.avg("Jitter (ms)").alias("Slice_Jitter_ms"),
        F.avg("Packet Loss (%)").alias("Slice_Packet_Loss_pct"),
        F.max("Network Load (%)").alias("Slice_Network_Load_pct"),
        F.avg("Slice Latency (ms)").alias("Slice_Latency_ms")
    )

    # Seuil throughput = exigences 3GPP fixes (TS 23.501)
    # eMBB  : 10 Mbps min (vidéo HD cloud gaming)
    # URLLC : 0.1 Mbps min (contrôle temps-réel)
    # mMTC  : 0.01 Mbps min (signalisation)
    df_slice = df_slice.withColumn(
        "SLA_Throughput_Min_Mbps",
        F.when(F.col("Slice Type") == "eMBB",  F.lit(10.0))
         .when(F.col("Slice Type") == "URLLC", F.lit(0.1))
         .when(F.col("Slice Type") == "mMTC",  F.lit(0.01))
         .otherwise(F.lit(1.0))
    )

    # Seuils de pertes 3GPP (TS 23.501)
    df_slice = df_slice.withColumn(
        "SLA_PacketLoss_Max_pct",
        F.when(F.col("Slice Type") == "URLLC", F.lit(0.001))
         .when(F.col("Slice Type") == "eMBB",  F.lit(1.0))
         .when(F.col("Slice Type") == "mMTC",  F.lit(5.0))
         .otherwise(F.lit(5.0))
    )

    # Seuils de latence 3GPP (maintenant utilisés dans le SLA)
    df_slice = df_slice.withColumn(
        "SLA_Latency_Max_ms",
        F.when(F.col("Slice Type") == "URLLC", F.lit(10.0))
         .when(F.col("Slice Type") == "eMBB",  F.lit(50.0))
         .when(F.col("Slice Type") == "mMTC",  F.lit(100.0))
         .otherwise(F.lit(100.0))
    )

    # SLA basé sur débit + pertes + latence (simulation M/M/1 cohérente)
    df_slice = df_slice.withColumn(
        "SLA_OK",
        (
            (F.col("Slice_Throughput_Mbps") >= F.col("SLA_Throughput_Min_Mbps")) &
            (F.col("Slice_Packet_Loss_pct") <= F.col("SLA_PacketLoss_Max_pct")) &
            (F.col("Slice_Latency_ms")      <= F.col("SLA_Latency_Max_ms"))
        ).cast("boolean")
    )

    # Fenêtre par slice pour regarder t+15s
    window_slice = Window.partitionBy("Slice Type").orderBy("Time_Sec")

    df_slice = df_slice.withColumn(
        "SLA_OK_in_15s",
        F.lead("SLA_OK", 15).over(window_slice)
    )

    df_slice = df_slice.withColumn(
        "SLA_Violated_in_15s",
        (F.col("SLA_OK_in_15s") == F.lit(False)).cast("int")
    )

    # On enlève les lignes où le label futur est inconnu (fin de trace)
    df_slice = df_slice.filter(F.col("SLA_OK_in_15s").isNotNull())

    print("Exemple de métriques par slice + SLA + cible t+15s :")
    df_slice.show(20, truncate=False)

    print("Répartition des labels (SLA_Violated_in_15s) par Slice Type :")
    df_slice.groupBy("Slice Type", "SLA_Violated_in_15s").count().show()

    # ==========================================
    # 7. Export des CSV
    # ==========================================

    final_columns = [
        "Time_Sec", "Source", "Destination", "Protocol_Norm", "Src_Port", "Dst_Port",
        "Device Type",
        "Traffic Type",
        "Slice Type",
        "Throughput_Mbps",
        "Network Load (%)",
        "Packet Loss (%)",
        "Jitter (ms)",
        "Data Rate Requirement (Mbps)",
        "Latency Requirement (ms)",
        "Slice Latency (ms)",
        "Optimal Network Slice",
        "Slice Bandwidth (Mbps)"
    ]

    result = df_final.select(final_columns)

    print("Transformation Complete. Showing top 20 rows (flux) :")
    result.show(20, truncate=False)

    # Dataset par flux
    output_path = "/home/work/output/processed_traffic_csv"
    (result
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .option("delimiter", ",")
        .csv(output_path)
    )

    # Dataset par slice + SLA – prêt pour le modèle
    slice_output_path = "/home/work/output/slice_sla_dataset"
    (df_slice
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .option("delimiter", ",")
        .csv(slice_output_path)
    )

    spark.stop()

if __name__ == "__main__":
    main()
