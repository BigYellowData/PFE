# PFE - Network Slicing 5G : Prediction de violations SLA

Projet de fin d'etudes : prediction a court terme du trafic 5G pour declencher des
agregations dans un reseau federe (Federated Learning) et eviter les violations SLA
des differents slices reseau.

## Objectif

A partir de captures reseau reelles (NVIDIA GeForce Now - cloud gaming), construire
un dataset labellise pour entrainer un modele de prediction de violations SLA a t+15s
par type de slice 5G (eMBB, URLLC, mMTC).

## Architecture du projet

```
PFE-NetworkSlicing/
|
|-- docker-compose.yml          # Environnement Spark (Jupyter + PySpark)
|
|-- data/
|   |-- GeForce_Now_1.csv       # Capture reseau brute (~442 MB)
|   |-- snippet.csv             # Extrait de 20 lignes pour test
|   +-- 5G_Traffic_Datasets/
|       +-- Game_Streaming/
|           +-- GeForce_Now/
|               |-- GeForce_Now_2.csv ... GeForce_Now_9.csv  (~5.7 GB total)
|
|-- work/
|   |-- slices.py               # Pipeline ETL principal (PySpark)
|   |-- state_data.py           # Stats de distribution des slices
|   |-- snippet.py              # Generation de l'extrait CSV
|   |
|   |-- output/
|   |   |-- processed_traffic_csv/    # Dataset par flux (5-tuple)
|   |   +-- slice_sla_dataset/        # Dataset par slice + SLA (pour ML)
|   |
|   +-- simu5g/
|       |-- convert_traces.py   # Conversion traces vers format simu5G
|       |-- NetworkSlicing.ned  # Topologie reseau OMNeT++
|       |-- omnetpp.ini         # Configuration simulation
|       |-- sliceConfig.xml     # Allocation Resource Blocks par slice
|       +-- GUIDE.md            # Guide d'utilisation simu5G
```

## Donnees source

Les donnees brutes proviennent de captures Wireshark de sessions NVIDIA GeForce Now
(cloud gaming). Chaque CSV contient des paquets reseau avec :

| Colonne     | Description                          | Exemple                           |
|-------------|--------------------------------------|-----------------------------------|
| No.         | Numero du paquet                     | 436                               |
| Time        | Timestamp de capture                 | 2022-09-27 13:08:31.564846        |
| Source      | Adresse IP source                    | 10.215.173.1                      |
| Destination | Adresse IP destination               | 112.217.128.200                   |
| Protocol    | Protocole reseau                     | TCP, TLSv1.2, UDP, QUIC          |
| Length      | Taille du paquet (octets)            | 557                               |
| Info        | Detail du paquet (ports, flags, etc) | 58632 > 443 [SYN] Seq=0 Win=...  |

Volume total : ~6.2 GB repartis sur 9 fichiers CSV.

---

## Phase 1 : Pipeline ETL (slices.py)

### Lancement

```bash
# 1. Demarrer le conteneur Docker
docker-compose up -d

# 2. Executer le pipeline
docker exec mon_spark_lab spark-submit /home/work/slices.py
```

### Etapes du pipeline

Le script `work/slices.py` execute 7 etapes :

#### Etape 1 - Pre-processing

- Parse les timestamps, cree `Time_Sec` (arrondi a la seconde)
- Extrait les ports source/destination depuis la colonne `Info` (regex)
- Normalise les protocoles : TLS/SSL -> TCP, QUIC -> UDP
- Detecte les pertes de paquets (retransmissions, Dup ACK)

#### Etape 2 - Calcul du Jitter

- Fenetre glissante par flux (5-tuple : Src, Dst, Proto, Src_Port, Dst_Port)
- Calcul du delta inter-arrivee entre paquets consecutifs
- Jitter = ecart-type des deltas (stddev)

#### Etape 3 - Agregation par flux et par seconde

Regroupement par (Time_Sec, Source, Destination, Protocol, Src_Port, Dst_Port) :
- `Sum_Length` : somme des tailles de paquets
- `Count_Total_Packets` : nombre de paquets
- `Count_Retransmissions` : nombre de retransmissions
- `Jitter_Raw` : stddev inter-arrivee

#### Etape 4 - Classification du trafic

Classification fonctionnelle basee sur les ports NVIDIA GeForce Now :

| Port  | Type de trafic     | Slice 5G    |
|-------|--------------------|-------------|
| 322   | Game Management    | URLLC       |
| 443   | Platform Admin     | mMTC        |
| 49003 | Audio (downlink)   | URLLC       |
| 49004 | Audio (uplink)     | URLLC       |
| 49005 | Video (downlink)   | eMBB        |
| 49006 | User Input         | URLLC       |
| Autre | Unknown            | Best Effort |

Reference : documentation NVIDIA GeForce Now et 3GPP TS 23.501.

#### Etape 5 - Simulation physique des slices (M/M/1)

Puisqu'on n'a pas acces aux donnees de l'operateur (allocation de ressources,
latence reseau coeur), on simule le comportement des slices avec un modele
de file d'attente M/M/1.

**Capacite allouee par slice :**

| Slice       | Capacite (Mbps) | Base latence (ms) | Priorite |
|-------------|----------------:|-------------------:|---------:|
| eMBB        | 50              | 5                  | 2        |
| URLLC       | 10              | 1                  | 1 (max)  |
| mMTC        | 5               | 10                 | 3        |
| Best Effort | 10              | 8                  | 4        |

**Modele de latence :**

```
Network_Load = throughput / capacite_slice          (borne a 100%)
rho = Network_Load / 100                            (charge normalisee [0, 0.99])
Latence = base_latency / (1 - rho) + bruit_gaussien
```

Quand la charge (rho) augmente, la latence diverge asymptotiquement. C'est
un comportement physiquement fonde (theorie des files d'attente de Kendall).

**Metriques reelles (non simulees) :**
- Throughput : calcule directement depuis les tailles de paquets
- Packet Loss : retransmissions observees dans les captures
- Jitter : stddev inter-arrivee mesuree

#### Etape 6 - SLA et label de prediction

**Seuils SLA (3GPP TS 23.501) :**

| Slice       | Throughput min (Mbps) | Latence max (ms) | Packet Loss max (%) |
|-------------|----------------------:|------------------:|--------------------:|
| eMBB        | 10.0                  | 50                | 1.0                 |
| URLLC       | 0.1                   | 10                | 0.001               |
| mMTC        | 0.01                  | 100               | 5.0                 |
| Best Effort | 1.0                   | 100               | 5.0                 |

**SLA_OK** = (throughput >= min) ET (packet_loss <= max) ET (latence <= max)

**Label de prediction** : `SLA_Violated_in_15s`
- Fenetre glissante par slice, regarde `SLA_OK` a t+15 secondes
- `1` = violation SLA dans 15 secondes (il faut declencher l'agregation)
- `0` = pas de violation, situation stable

#### Etape 7 - Export

Deux datasets CSV produits :

**1. `processed_traffic_csv`** - Dataset par flux (granularite fine)

| Colonne                    | Description                          |
|----------------------------|--------------------------------------|
| Time_Sec                   | Timestamp (seconde)                  |
| Source, Destination        | Adresses IP                          |
| Protocol_Norm              | TCP ou UDP                           |
| Src_Port, Dst_Port         | Ports reseau                         |
| Device Type                | Type de terminal                     |
| Traffic Type               | Classification fonctionnelle         |
| Slice Type                 | eMBB, URLLC, mMTC, Best Effort      |
| Throughput_Mbps            | Debit (reel)                         |
| Network Load (%)           | Charge du slice                      |
| Packet Loss (%)            | Pertes (reel)                        |
| Jitter (ms)                | Variation latence (reel)             |
| Data Rate Requirement      | Exigence 3GPP                        |
| Latency Requirement        | Exigence 3GPP                        |
| Slice Latency (ms)         | Latence simulee M/M/1                |
| Optimal Network Slice      | Slice recommande                     |
| Slice Bandwidth (Mbps)     | Bande passante du slice              |

**2. `slice_sla_dataset`** - Dataset par slice (pour le ML)

| Colonne                 | Description                           |
|-------------------------|---------------------------------------|
| Time_Sec                | Timestamp (seconde)                   |
| Slice Type              | eMBB, URLLC, mMTC, Best Effort       |
| Slice_Throughput_Mbps   | Debit agrege du slice                 |
| Slice_Jitter_ms         | Jitter moyen du slice                 |
| Slice_Packet_Loss_pct   | Pertes moyennes du slice              |
| Slice_Network_Load_pct  | Charge max du slice                   |
| Slice_Latency_ms        | Latence moyenne simulee               |
| SLA_Throughput_Min_Mbps | Seuil throughput 3GPP                 |
| SLA_PacketLoss_Max_pct  | Seuil pertes 3GPP                     |
| SLA_Latency_Max_ms      | Seuil latence 3GPP                    |
| SLA_OK                  | SLA respecte a cet instant            |
| SLA_OK_in_15s           | SLA respecte dans 15 secondes         |
| **SLA_Violated_in_15s** | **LABEL : 1 = violation imminente**   |

---

## Phase 2 : Modelisation ML

### Dataset a utiliser

Le fichier `slice_sla_dataset` est directement exploitable pour un modele de
classification binaire supervisee.

### Features (X)

```
Slice_Throughput_Mbps, Slice_Jitter_ms, Slice_Packet_Loss_pct,
Slice_Network_Load_pct, Slice_Latency_ms
```

Plus les features categoriques : `Slice Type` (one-hot encoding).

On peut aussi deriver des features temporelles :
- Moyenne glissante (5s, 10s) du throughput
- Tendance (derivee) du throughput sur les dernieres N secondes
- Ecart-type glissant de la latence

### Label (y)

```
SLA_Violated_in_15s   (0 ou 1)
```

### Modeles recommandes

| Modele        | Avantage                              | Usage                      |
|---------------|---------------------------------------|----------------------------|
| LSTM          | Capture les patterns temporels        | Serie temporelle par slice |
| XGBoost       | Performant sur donnees tabulaires     | Baseline rapide            |
| Random Forest | Interpretable, bon pour les features  | Feature importance         |
| GRU           | Plus leger que LSTM, similar perf     | Si ressources limitees     |

### Workflow ML recommande

```
1. Charger slice_sla_dataset.csv
2. Feature engineering (moyennes glissantes, tendances)
3. Split temporel (pas aleatoire !) : train sur sessions 1-6, test sur 7-9
4. Entrainer le modele par slice OU un modele unique multi-slice
5. Evaluer : Precision, Recall, F1-score, AUC-ROC
6. Federated Learning : chaque "client" = un slice ou une session
```

### Split temporel

Ne pas utiliser de split aleatoire (fuite de donnees temporelles).
Utiliser un split chronologique :

```python
train = df[df["Time_Sec"] < "2022-09-29"]
test  = df[df["Time_Sec"] >= "2022-09-29"]
```

### Metriques d'evaluation

- **Recall** : critique (on veut detecter TOUTES les violations)
- **Precision** : importante (eviter les fausses alertes)
- **F1-score** : compromis
- **AUC-ROC** : performance globale du classifieur

---

## Phase 3 (optionnelle) : Simulation simu5G

Pour remplacer la simulation analytique (M/M/1) par un simulateur reseau complet.
Voir `work/simu5g/GUIDE.md` pour les details.

```bash
# Convertir les traces
cd work/simu5g
python convert_traces.py ../output/processed_traffic_csv/part-00000-*.csv ./traces
```

Les resultats simu5G (latence, pertes, throughput par slice) peuvent remplacer
les colonnes simulees du dataset pour un modele ML plus realiste.

---

## Environnement technique

| Composant       | Version / Outil                     |
|-----------------|-------------------------------------|
| Traitement      | Apache Spark (PySpark)              |
| Conteneur       | Docker (jupyter/pyspark-notebook)   |
| Langage         | Python 3.14                         |
| Donnees         | Captures Wireshark GeForce Now      |
| Simulation      | M/M/1 analytique / simu5G (OMNeT++)|
| Volume donnees  | ~6.2 GB (9 fichiers CSV)            |

### Commandes rapides

```bash
# Demarrer l'environnement
docker-compose up -d

# Lancer le pipeline ETL
docker exec mon_spark_lab spark-submit /home/work/slices.py

# Acceder a JupyterLab
# http://localhost:8888 (token dans docker logs mon_spark_lab)

# Spark UI
# http://localhost:4040

# Arreter l'environnement
docker-compose down
```

---

## References

- 3GPP TS 23.501 - System architecture for the 5G System (network slicing, QoS)
- NVIDIA GeForce Now - Network requirements and port documentation
- Kendall's notation (M/M/1) - Queuing theory for latency modeling
- simu5G - https://github.com/Unipisa/Simu5G
