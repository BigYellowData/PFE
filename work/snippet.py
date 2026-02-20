import csv
import os

def create_csv_snippet(input_csv, output_csv, n_lines=20):
    with open(input_csv, newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for i, row in enumerate(reader):
            if i >= n_lines:
                break
            writer.writerow(row)

# Exemple d'utilisation
create_csv_snippet(
    input_csv=r"C:\Users\Utilisateur\Desktop\PFE_BigData\PFE_Bigdata\PFE-NetworkSlicing\PFE-NetworkSlicing\data\GeForce_Now_1.csv",
    output_csv=r"C:\Users\Utilisateur\Desktop\PFE_BigData\PFE_Bigdata\PFE-NetworkSlicing\PFE-NetworkSlicing\data\snippet.csv"
)
