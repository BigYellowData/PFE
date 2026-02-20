from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main():
    spark = SparkSession.builder.appName("Slice_Count").getOrCreate()

    # Chemin vers ton CSV généré (le dossier, pas juste un fichier)
    input_path = "/home/work/output/slice_sla_dataset/part-00000-b317d3d8-1f7d-4062-840f-0f7e7c162c0e-c000.csv"  

    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv(input_path)

    # Attention : il y a un espace dans "Slice Type", il faut utiliser les backticks
    slice_counts = df.groupBy("`Slice Type`") \
                     .count() \
                     .orderBy(F.desc("count"))

    print("Nombre d'occurrences par Slice Type :")
    slice_counts.show(truncate=False)

    spark.stop()

if __name__ == "__main__":
    main()
