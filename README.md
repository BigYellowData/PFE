# Prediction de violations SLA dans les reseaux 5G avec Network Slicing

**Projet de Fin d'Etudes (PFE)** - Prediction de violations SLA a 15 secondes sur des slices 5G (eMBB, URLLC, mMTC) par apprentissage profond (LSTM).

---

## Resultats

| Approche | F1 global | AUC-ROC | CV F1 | eMBB F1 | URLLC F1 | mMTC F1 |
|----------|-----------|---------|-------|---------|----------|---------|
| **A - M/M/1** | 0.8606 | 0.9710 | 0.8459 ± 0.04 | 0.4236 | 0.0000 | ~1.00 |
| **B - simu5G** | **0.6479** | **0.9871** | 0.6394 ± 0.24 | **0.9913** | **0.9913** | 0.0215 |

> L'Approche A a un F1 global plus eleve mais trompeusement gonfle par des slices deterministes (Best Effort=100%, mMTC=98% de violations dans les donnees reelles).
> L'Approche B avec LOSO (Leave-One-Scenario-Out) mesure la vraie capacite de generalisation sur des scenarios inedits.

---

## Architecture du projet

```
PFE-NetworkSlicing/
|
|-- data/
|   +-- 5G_Traffic_Datasets/Game_Streaming/GeForce_Now/
|       +-- GeForce_Now_1.csv ... GeForce_Now_9.csv   # Captures Wireshark (~7 GB, 62.8M paquets)
|
|-- work/
|   |-- slices.py                        # Pipeline ETL PySpark : classification + agregation
|   |-- ml_lstm_approach_a.ipynb         # LSTM avec latence M/M/1 (Approche A)
|   |-- ml_lstm_approach_b_simu5g.ipynb  # LSTM avec donnees simu5G + LOSO (Approche B)
|   |-- state_data.py                    # Stats de distribution des slices
|   |-- snippet.py                       # Generation extrait CSV pour test
|   |
|   |-- output/                          # Datasets generes + modeles entraines
|   |   |-- slice_sla_dataset/           # Dataset Approche A (111 040 lignes, 7 jours)
|   |   |-- lstm_approach_a.pt           # Modele entraine Approche A
|   |   +-- scaler_approach_a.pkl        # Scaler MinMax Approche A
|   |
|   +-- simu5g/
|       |-- convert_raw_traces_spark.py  # Extraction stats trafic depuis Wireshark
|       |-- extract_simu5g_metrics.py    # Conversion resultats simulation -> CSV ML
|       |-- simulation_pfe/
|       |   |-- NetworkSlicing.ned       # Topologie reseau (OMNeT++ NED)
|       |   |-- omnetpp.ini             # 13 scenarios de simulation
|       |   |-- sliceConfig.xml         # Allocation Resource Blocks par slice
|       |   +-- results/                # Fichiers .vec et .sca (resultats simulation)
|       +-- output/
|           +-- all_simu5g.csv          # Dataset Approche B (95 805 lignes, 13 scenarios)
|
|-- docker-compose.yml                  # Environnement Spark (PySpark + JupyterLab)
|-- requirements.txt                    # Dependances Python
|-- rapport_complet.txt                 # Rapport technique detaille
+-- README.md                           # Ce fichier
```

---

## Donnees source

**Dataset** : captures Wireshark de sessions NVIDIA GeForce Now (cloud gaming), publie par l'Universite de Gand.

- 9 fichiers CSV, ~7 GB, **62.8 millions de paquets**, 2.3 heures de session
- Cas d'usage ideal : une session GeForce Now genere simultanement les 3 types de trafic 5G

| Port | Type de trafic | Slice 5G | Taille paquets | Intervalle |
|------|---------------|----------|---------------|------------|
| UDP 49005 | Video streaming | **eMBB** | 1290 B | 3.8 ms |
| UDP 49003/49004/49006 | Audio + input | **URLLC** | 292 B | 30 ms |
| TCP 443 | Signalisation TLS | **mMTC** | 47 B | 5 s |
| Autres | Web, divers | Best Effort | 512 B | 100 ms |

---

## Approche A - LSTM avec modele M/M/1

### Principe

La latence n'etant pas mesurable directement depuis les captures Wireshark,
elle est **estimee** par un modele de file d'attente M/M/1 :

```
rho = throughput / capacite_slice
W   = 1 / (mu - lambda)  =  latence estimee
```

La capacite allouee par slice : eMBB=100 Mbps, URLLC=10 Mbps, mMTC=1 Mbps.

### Pipeline

```
GeForce_Now_*.csv (9 fichiers, 7 GB)
    |
    v  [Apache Spark - slices.py]
    Classification par port (49005->eMBB, 49003->URLLC, 443->mMTC)
    Agregation par seconde et par slice
    Estimation latence M/M/1
    Etiquetage SLA_Violated_in_15s
    |
    v
Dataset : 111 040 lignes, 7 jours (2022-09-27 au 2022-10-04)
    |
    v  [ml_lstm_approach_a.ipynb]
    Feature engineering (MA 5s, MA 10s, derivees)
    Split temporel 80/20
    LSTM (64->32, dropout 0.3, BCEWithLogitsLoss)
    Prediction 15s
```

### Lancement Approche A

```bash
# 1. Demarrer l'environnement Spark
docker-compose up -d

# 2. Generer le dataset
docker exec mon_spark_lab spark-submit /home/work/slices.py

# 3. Ouvrir le notebook
# http://localhost:8888 -> work/ml_lstm_approach_a.ipynb
```

### Seuils SLA Approche A

| Slice | Throughput min | Latence max | Perte paquets max |
|-------|---------------|-------------|-------------------|
| eMBB | 10.0 Mbps | 50 ms | 1.0% |
| URLLC | 0.1 Mbps | 10 ms | 0.001% |
| mMTC | 0.01 Mbps | 100 ms | 5.0% |

### Resultats Approche A

| Slice | F1 | Violations |
|-------|----|-----------|
| Best Effort | 1.0000 | 100% (toujours viole) |
| mMTC | ~1.00 | 98.4% (quasi toujours) |
| eMBB | 0.4236 | 34.0% |
| URLLC | 0.0000 | 0.03% (quasi jamais) |

**F1 global = 0.8606, AUC = 0.9710** — mais gonfle par les slices deterministes.

---

## Approche B - LSTM avec simulation simu5G

### Principe

Remplacement de la latence estimee (M/M/1) par une latence **reellement mesuree**
dans un simulateur 5G complet (OMNeT++ + simu5G).

**simu5G** = simulateur open-source (Universite de Pise, IEEE Access 2020) qui
implemente un reseau 5G complet : canal radio NR, scheduling MAC, protocoles
PDCP/RLC/MAC/PHY, tunneling GTP-U, architecture 3GPP TS 23.501.

La latence est mesuree end-to-end : `delay = t_reception_serveur - t_creation_ue`

### Architecture du reseau simule

```
UE (eMBB, URLLC, mMTC, BE)
    | radio 5G NR (3.5 GHz, 100 MHz, 50 RBs)
    v
gNodeB (scheduling MAXCI, 40 dBm)
    | N3 (GTP-U)
    v
iUpf (edge UPF)
    | N9 (GTP-U)
    v
upf (anchor UPF)
    | N6 (IP)
    v
Router -> Serveurs (serverEmbb, serverUrllc, serverMmtc)
```

Allocation des Resource Blocks par slice :
- eMBB : 40% des RBs, priorite 3
- URLLC : 35% des RBs, priorite 1 (la plus haute)
- mMTC : 15% des RBs, priorite 4

### Les 13 scenarios de simulation

| Scenario | UEs eMBB | UEs URLLC | UEs mMTC | Objectif |
|----------|----------|-----------|----------|---------|
| Normal | 4 | 4 | 2 | Charge de base |
| OverloadEmbb | 12 | 4 | 2 | Saturation eMBB |
| OverloadUrllc | 4 | 12 | 2 | Saturation URLLC |
| OverloadAll | 10 | 10 | 6 | Saturation globale |
| RealTraffic | 4 | 8 | 2 | Trafic realiste |
| RealTrafficOverload | 8 | 20 | 4 | Trafic realiste sature |
| MediumLoad | 6 | 10 | 3 | Charge intermediaire |
| TransitionEmbb | 10 | 4 | 2 | eMBB pres du seuil |
| TransitionUrllc | 4 | 16 | 2 | URLLC pres du seuil |
| StressEmbb | 20 | 4 | 2 | Saturation max eMBB |
| HeavyEmbb | 15 | 6 | 3 | Surcharge moderee eMBB |
| CongestionHigh | 12 | 15 | 5 | Congestion multi-slice |
| ExtremeLoad | 18 | 18 | 8 | Charge extreme |

**Test set (LOSO)** : MediumLoad, ExtremeLoad, RealTraffic
**Train set** : les 10 autres scenarios

### Seuils SLA Approche B

| Slice | Throughput min | Latence max | Perte paquets max |
|-------|---------------|-------------|-------------------|
| eMBB | 10.0 Mbps | 50 ms | 1.0% |
| URLLC | 0.5 Mbps | 20 ms | 0.001% |
| mMTC | 0.0003 Mbps | 8.8 ms | 1.0% |

> Note : la latence simu5G est end-to-end (UE app -> serveur), non l'interface radio seule.
> Le seuil URLLC de 20 ms et mMTC de 8.8 ms integrent le overhead reseau (~14 ms incompressible).

### Lancement Approche B

#### 1. Lancer les simulations simu5G

```bash
cd work/simu5g/simulation_pfe

# Un scenario a la fois
../../../Simu5G/src/simu5g_run -u Cmdenv -c Normal \
    -n .:../../../Simu5G/src:../../../inet4.5/src \
    omnetpp.ini

# Tous les scenarios (exemple bash)
for config in Normal OverloadEmbb OverloadUrllc OverloadAll RealTraffic \
              RealTrafficOverload MediumLoad TransitionEmbb TransitionUrllc \
              StressEmbb HeavyEmbb CongestionHigh ExtremeLoad; do
    ../../../Simu5G/src/simu5g_run -u Cmdenv -c $config \
        -n .:../../../Simu5G/src:../../../inet4.5/src omnetpp.ini
done
```

#### 2. Extraire les metriques

```bash
python work/simu5g/extract_simu5g_metrics.py \
    --results work/simu5g/simulation_pfe/results \
    --output  work/simu5g/output/all_simu5g.csv
```

#### 3. Entrainer le LSTM

```bash
# Ouvrir le notebook
jupyter notebook work/ml_lstm_approach_b_simu5g.ipynb
# Ou via Docker : http://localhost:8888
```

### Choix techniques importants

#### Leave-One-Scenario-Out (LOSO)

Sans LOSO, le modele atteint F1=0.9999 en cross-validation.
Cause : les metriques sont quasi-constantes intra-scenario, n'importe quelle
feature separe parfaitement les labels entre scenarios.

Avec LOSO, le modele doit generaliser a des scenarios inedits (charge differente,
nombre d'UEs jamais vu) -> F1 chute a 0.64 = apprentissage reel.

#### Temperature Scaling

Le LSTM produit des logits potentiellement mal calibres.
`sigmoid(logit)` peut saturer, toutes les probabilites se retrouvant proches de 1.0.

Solution : diviser les logits par une temperature T avant sigmoid :
```python
# T optimise par minimisation de la NLL sur le validation set
result = minimize_scalar(nll_temperature, bounds=(0.1, 50.0), method='bounded')
T = result.x  # T = 1.06 dans notre cas
y_proba = torch.sigmoid(logits / T)
```

Avec T=1.06, les probabilites sont dans [0.008, 0.999] et le seuil standard 0.5 fonctionne correctement.

### Resultats Approche B

**Test set (MediumLoad, ExtremeLoad, RealTraffic — scenarios jamais vus)** :

| Slice | F1 | Precision | Recall | AUC |
|-------|----|-----------|--------|-----|
| **eMBB** | **0.9913** | 0.9994 | 0.9833 | 0.9903 |
| **URLLC** | **0.9913** | 0.9994 | 0.9833 | 0.9900 |
| mMTC | 0.0215 | 0.0108 | 1.0000 | 0.3803 |
| **Global** | **0.6479** | — | — | **0.9871** |

> Seuil de decision standard 0.5 apres Temperature Scaling.

**Cross-Validation GroupKFold (5 folds par scenario)** :

| Fold | Scenarios test | F1 |
|------|---------------|-----|
| 1 | OverloadEmbb, RealTrafficOverload | 0.748 |
| 2 | Normal, RealTraffic | 0.498 |
| 3 | ExtremeLoad, OverloadUrllc, TransitionUrllc | 0.718 |
| 4 | CongestionHigh, StressEmbb, TransitionEmbb | **0.920** |
| 5 | HeavyEmbb, MediumLoad, OverloadAll | 0.313 |
| **Moy** | | **0.639 ± 0.236** |

F1 moyen par slice en CV : URLLC=0.80±0.27, eMBB=0.38±0.51, mMTC=0.26±0.36

---

## Architecture LSTM (commune aux deux approches)

```
Entree : (batch, 30 pas de temps, 23 features)

LSTM 1 : 64 neurones  + Dropout 0.30
LSTM 2 : 32 neurones  + Dropout 0.30
Dense  : 32 -> 16     + ReLU
Sortie : 16 -> 1      (logit)

Temperature Scaling : sigmoid(logit / T)
Decision           : violation si proba >= 0.5
```

**Features (23)** :
- 5 metriques de base : Throughput, Latency, Jitter, PacketLoss, NetworkLoad
- 15 features derivees : moyenne mobile 5s, 10s, et derivee pour chaque metrique
- 3 one-hot : slice_eMBB, slice_URLLC, slice_mMTC

**Entrainement** :
- Perte : `BCEWithLogitsLoss` avec `pos_weight` (desequilibre de classes)
- Optimiseur : Adam (lr=0.001)
- Scheduler : `ReduceLROnPlateau` (factor=0.5, patience=3)
- Early stopping : patience=10
- Batch size : 256, Epochs max : 50

---

## Installation

### Prerequis

- Python 3.10+
- Docker + Docker Compose (pour Spark/Approche A)
- OMNeT++ 6.0 + INET 4.5.4 + simu5G (pour Approche B)

### Dependances Python

```bash
pip install -r requirements.txt
```

Principaux packages :
```
torch>=2.0
pandas numpy scikit-learn matplotlib
scipy  # pour minimize_scalar (Temperature Scaling)
pyspark  # si execution Spark en local
```

### Docker (Spark)

```bash
docker-compose up -d
# JupyterLab : http://localhost:8888
# Spark UI   : http://localhost:4040
docker-compose down
```

---

## References

1. G. Nardini, G. Stea, A. Virdis, **"Simu5G - An OMNeT++ Library for End-to-End Performance Evaluation of 5G Networks"**, IEEE Access, vol. 8, pp. 181176-181191, 2020. DOI: 10.1109/ACCESS.2020.3028550

2. A. Varga, R. Hornig, **"An Overview of the OMNeT++ Simulation Environment"**, SIMUTools, 2008.

3. **3GPP TS 23.501** - System architecture for the 5G System (5GS), V17.5.0, 2022.

4. **3GPP TS 38.300** - NR and NG-RAN Overall Description, Release 17.

5. L. Kleinrock, **"Queueing Systems, Volume 1: Theory"**, Wiley, 1975.

6. C. Guo et al., **"On Calibration of Modern Neural Networks"**, ICML 2017. (Temperature Scaling)
