"""
Conversion des captures Wireshark BRUTES vers le format simu5G avec Spark.

Ce script lit directement les fichiers GeForce_Now_*.csv (captures Wireshark)
et genere des traces compatibles simu5G, SANS passer par le modele M/M/1.

Usage (dans Docker) :
    docker exec mon_spark_lab spark-submit /home/work/simu5g/convert_raw_traces_spark.py

Les traces sont ecrites dans /home/work/simu5g/traces_raw/
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StringType
import os

def main():
    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("ConvertRawTraces_simu5G") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    # Chemin vers les donnees brutes
    input_dir = "/home/data/Game_Streaming/GeForce_Now"
    output_base = "/home/work/simu5g/traces_raw_spark"

    print("=" * 60)
    print("CONVERSION TRACES BRUTES -> simu5G (Spark)")
    print("=" * 60)

    # Lire tous les CSV
    print(f"\nLecture des fichiers depuis {input_dir}...")
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(input_dir)

    total_rows = df.count()
    print(f"Total paquets charges: {total_rows}")

    # ==========================================
    # 1. Extraction des ports depuis Info
    # ==========================================
    print("\n1. Extraction des ports...")

    df = df.withColumn("Src_Port",
        F.regexp_extract(F.col("Info"), r"^(\d+)\s*>\s*(\d+)", 1).cast("int")) \
           .withColumn("Dst_Port",
        F.regexp_extract(F.col("Info"), r"^(\d+)\s*>\s*(\d+)", 2).cast("int"))

    # Filtrer les lignes sans ports valides
    df = df.filter(F.col("Src_Port").isNotNull() & (F.col("Src_Port") > 0))

    # ==========================================
    # 2. Classification par Slice
    # ==========================================
    print("2. Classification par Slice...")

    # Normaliser le protocole
    df = df.withColumn(
        "Protocol_Norm",
        F.when(F.col("Protocol").rlike("(?i)TLS|SSL"), "TCP")
         .when(F.col("Protocol").rlike("(?i)QUIC"), "UDP")
         .otherwise(F.col("Protocol"))
    )

    # Classification des slices (identique a slices.py)
    is_tcp = F.col("Protocol_Norm") == "TCP"
    is_udp = F.col("Protocol_Norm") == "UDP"

    is_port_322 = (F.col("Src_Port") == 322) | (F.col("Dst_Port") == 322)
    is_port_443 = (F.col("Src_Port") == 443) | (F.col("Dst_Port") == 443)
    is_port_49003 = (F.col("Src_Port") == 49003) | (F.col("Dst_Port") == 49003)
    is_port_49004 = (F.col("Src_Port") == 49004) | (F.col("Dst_Port") == 49004)
    is_port_49005 = (F.col("Src_Port") == 49005) | (F.col("Dst_Port") == 49005)
    is_port_49006 = (F.col("Src_Port") == 49006) | (F.col("Dst_Port") == 49006)

    df = df.withColumn(
        "Slice_Type",
        F.when(is_udp & is_port_49005, "eMBB")
         .when(is_udp & (is_port_49003 | is_port_49004), "URLLC")
         .when(is_udp & is_port_49006, "URLLC")
         .when(is_tcp & is_port_322, "URLLC")
         .when(is_tcp & is_port_443, "mMTC")
         .otherwise("BestEffort")
    )

    # ==========================================
    # 3. Parser le timestamp
    # ==========================================
    print("3. Parsing des timestamps...")

    df = df.withColumn("Time_Timestamp", F.to_timestamp(F.col("Time")))

    # ==========================================
    # 4. Creer une cle de flux (5-tuple)
    # ==========================================
    df = df.withColumn(
        "Flow_Key",
        F.concat_ws("_",
            F.col("Slice_Type"),
            F.col("Source"),
            F.col("Destination"),
            F.col("Protocol_Norm"),
            F.col("Src_Port"),
            F.col("Dst_Port")
        )
    )

    # ==========================================
    # 5. Calculer l'offset par rapport au premier paquet du flux
    # ==========================================
    print("4. Calcul des offsets temporels...")

    window_flow = Window.partitionBy("Flow_Key").orderBy("Time_Timestamp")

    # Premier timestamp du flux
    df = df.withColumn("First_Time", F.first("Time_Timestamp").over(window_flow))

    # Offset en secondes
    df = df.withColumn(
        "Offset_Sec",
        F.col("Time_Timestamp").cast("double") - F.col("First_Time").cast("double")
    )

    # Numero de paquet dans le flux
    df = df.withColumn("Pkt_Num", F.row_number().over(window_flow) - 1)

    # ==========================================
    # 6. Statistiques par slice
    # ==========================================
    print("\n5. Statistiques par Slice:")

    stats = df.groupBy("Slice_Type").agg(
        F.count("*").alias("Packets"),
        F.countDistinct("Flow_Key").alias("Flows"),
        F.avg("Length").alias("Avg_Pkt_Size"),
        F.max("Offset_Sec").alias("Max_Duration")
    ).orderBy(F.desc("Packets"))

    stats.show()

    # ==========================================
    # 7. Calculer throughput et intervalle moyen par slice
    # ==========================================
    print("6. Parametres pour simu5G:")

    # Agreger par flux pour avoir duree et bytes totaux
    flow_stats = df.groupBy("Slice_Type", "Flow_Key").agg(
        F.count("*").alias("Pkt_Count"),
        F.sum("Length").alias("Total_Bytes"),
        F.max("Offset_Sec").alias("Duration_Sec")
    ).filter(F.col("Duration_Sec") > 0)

    # Calculer throughput par flux
    flow_stats = flow_stats.withColumn(
        "Throughput_Mbps",
        (F.col("Total_Bytes") * 8) / (F.col("Duration_Sec") * 1e6)
    )
    flow_stats = flow_stats.withColumn(
        "Avg_Interval_Ms",
        (F.col("Duration_Sec") / F.col("Pkt_Count")) * 1000
    )

    # Moyenne par slice (pondere par nombre de paquets)
    slice_params = flow_stats.groupBy("Slice_Type").agg(
        F.sum("Pkt_Count").alias("Total_Packets"),
        F.sum("Total_Bytes").alias("Total_Bytes"),
        F.max("Duration_Sec").alias("Max_Duration"),
        F.avg("Throughput_Mbps").alias("Avg_Throughput_Mbps"),
        F.avg("Avg_Interval_Ms").alias("Avg_Interval_Ms")
    )

    # Calculer taille paquet moyenne
    slice_params = slice_params.withColumn(
        "Avg_Pkt_Size",
        F.col("Total_Bytes") / F.col("Total_Packets")
    )

    print("\nParametres recommandes pour omnetpp.ini:")
    slice_params.select(
        "Slice_Type",
        F.round("Avg_Throughput_Mbps", 2).alias("Throughput_Mbps"),
        F.round("Avg_Pkt_Size", 0).alias("Pkt_Size_B"),
        F.round("Avg_Interval_Ms", 2).alias("Interval_ms")
    ).show()

    # ==========================================
    # 8. Exporter les traces au format SVC
    # ==========================================
    print("7. Export des traces...")

    # Preparer les donnees pour export
    # Format SVC: memoryAdd length lid tid qid frameType isDiscardable isTruncatable frameNumber timestamp isControl

    export_df = df.select(
        "Slice_Type",
        "Flow_Key",
        "Pkt_Num",
        F.when(F.col("Length") < 64, 64).otherwise(F.col("Length")).alias("Length"),
        "Offset_Sec",
        "Source",
        "Destination",
        "Src_Port",
        "Dst_Port",
        "Protocol_Norm"
    )

    # Creer la ligne SVC
    export_df = export_df.withColumn(
        "SVC_Line",
        F.concat_ws(" ",
            F.format_string("0x%08x", F.col("Pkt_Num")),  # memoryAdd
            F.col("Length").cast("string"),               # length
            F.lit("0"),                                   # lid
            F.lit("0"),                                   # tid
            F.lit("0"),                                   # qid
            F.when(F.col("Pkt_Num") % 25 == 0, "I").otherwise("P"),  # frameType
            F.lit("0"),                                   # isDiscardable
            F.lit("0"),                                   # isTruncatable
            F.col("Pkt_Num").cast("string"),              # frameNumber
            F.format_string("%.6f", F.col("Offset_Sec")), # timestamp
            F.lit("0")                                    # isControl
        )
    )

    # Compter les paquets par flux pour filtrer les flux trop courts
    flow_counts = df.groupBy("Flow_Key").count().filter(F.col("count") >= 100)
    export_df = export_df.join(flow_counts, "Flow_Key")

    # Exporter par slice
    for slice_type in ["eMBB", "URLLC", "mMTC", "BestEffort"]:
        slice_df = export_df.filter(F.col("Slice_Type") == slice_type)
        slice_count = slice_df.count()

        if slice_count > 0:
            output_path = f"{output_base}/{slice_type}"
            print(f"  {slice_type}: {slice_count} paquets -> {output_path}")

            # Exporter en CSV (une ligne par paquet)
            slice_df.select("Flow_Key", "Pkt_Num", "SVC_Line") \
                .orderBy("Flow_Key", "Pkt_Num") \
                .coalesce(1) \
                .write \
                .mode("overwrite") \
                .option("header", "false") \
                .csv(output_path)

    # ==========================================
    # 9. Exporter aussi le format simple
    # ==========================================
    print("\n8. Export format simple...")

    simple_df = df.select(
        "Slice_Type",
        "Flow_Key",
        "Pkt_Num",
        F.format_string("%.6f", F.col("Offset_Sec")).alias("Offset"),
        F.when(F.col("Length") < 64, 64).otherwise(F.col("Length")).alias("Length")
    )

    simple_df = simple_df.join(flow_counts, "Flow_Key")

    for slice_type in ["eMBB", "URLLC", "mMTC", "BestEffort"]:
        slice_df = simple_df.filter(F.col("Slice_Type") == slice_type)
        slice_count = slice_df.count()

        if slice_count > 0:
            output_path = f"{output_base}/simple/{slice_type}"
            print(f"  {slice_type}: {slice_count} paquets -> {output_path}")

            slice_df.select("Flow_Key", "Offset", "Length") \
                .orderBy("Flow_Key", "Offset") \
                .coalesce(1) \
                .write \
                .mode("overwrite") \
                .option("header", "true") \
                .csv(output_path)

    print("\n" + "=" * 60)
    print("CONVERSION TERMINEE")
    print(f"Traces ecrites dans: {output_base}")
    print("=" * 60)

    spark.stop()


if __name__ == "__main__":
    main()
