# Network Slicing 5G — Boucle Fermée LSTM + MILP

**Projet de Fin d'Études (PFE)**  
Optimisation dynamique des ressources radio en réseau 5G hétérogène par prédiction LSTM et allocation MILP en boucle fermée.

---

## Résultats

| Approche | Réduction coût F(R) | Fenêtres améliorées |
|---|---|---|
| Allocation fixe (baseline) | 0% — référence | — |
| MILP seul (réactif) | ~20–25% | — |
| **LSTM + MILP (boucle fermée)** | **41.3%** | **98.3%** |
| LSTM + MILP + online learning | 51.8% | — |

Évalué sur **255 812 fenêtres** issues de **26 scénarios** (12 réels + 14 synthétiques), 3 gNBs hétérogènes.

---

## Architecture du projet

```
PFE-NetworkSlicing/
│
├── work/simu5g/
│   │
│   ├── loss/                              # Pipeline principal
│   │   ├── 00_eda_dataset.ipynb           # Analyse exploratoire du dataset
│   │   ├── 01_lstm_retrain.ipynb          # Entraînement LSTM Seq2Seq + FedAvg LoGO
│   │   ├── 02_rb_optimization_F.ipynb     # Calibration sigmoid + MILP offline
│   │   ├── 03_closed_loop.ipynb           # Boucle fermée LSTM+MILP (résultats 41.3%)
│   │   ├── 04_closed_loop_online.ipynb    # Boucle fermée + apprentissage continu (51.8%)
│   │   ├── all_simu5g.csv                 # Dataset brut (315 892 lignes, 20 scénarios)
│   │   ├── all_simu5g_trans.csv           # Dataset enrichi (+ scénarios de transition)
│   │   ├── create_rampup.py               # Génération scénarios de transition synthétiques
│   │   ├── models_lstm_v3/                # Modèles LSTM finaux (6 fichiers)
│   │   │   ├── model_final_{eMBB,URLLC,mMTC}.pt
│   │   │   └── scalers_final_{eMBB,URLLC,mMTC}.pkl
│   │   ├── output/                        # CSV résultats + figures
│   │   └── resultat/                      # Fichiers .vec/.sca/.vci Simu5G
│   │
│   ├── simulation_pfe/                    # Boucle fermée temps réel
│   │   ├── omnetpp_hetnet.ini             # Config OMNeT++ (20 scénarios)
│   │   ├── HetNetSlicing.ned              # Topologie réseau hétérogène
│   │   ├── network_config.xml             # Config réseau IP/5G
│   │   ├── sliceConfig*.xml               # Allocation RBs par gNB
│   │   ├── closed_loop_controller.py      # Contrôleur LSTM + MILP temps réel
│   │   ├── periodic_controller.py         # Contrôleur MILP seul (comparaison)
│   │   ├── compare_results.py             # Analyse comparative des 3 modes
│   │   ├── run_comparison.sh              # Lance les 3 modes (baseline/MILP/LSTM+MILP)
│   │   ├── run_all_sims.sh                # Lance tous les scénarios
│   │   └── comparison_results/            # Résultats des runs de comparaison live
│   │
│   ├── extract_simu5g_metrics.py          # Extraction .vec/.sca → CSV
│   └── GUIDE.md                           # Guide d'installation Simu5G
│
├── requirements.txt
├── docker-compose.yml                     # Environnement JupyterLab
└── README.md
```

---

## Contexte

Un réseau 5G partage ses ressources radio (Resource Blocks) entre 3 types de slices :

| Slice | Usage | Latence max | Débit min |
|---|---|---|---|
| **eMBB** | Streaming, vidéo HD | 60 ms | 10 Mbps |
| **URLLC** | Robotique, chirurgie à distance | 25 ms | 0.5 Mbps |
| **mMTC** | Capteurs IoT | 300 ms | — |

Le réseau simule **3 gNBs hétérogènes** (Macro UMa, Commerce UMi, Industrie UMi) avec des budgets RB différents (50 / 35 / 25 RBs).

Avec une allocation fixe par défaut : **100% de violations SLA pour eMBB, 68% pour URLLC**.

---

## Approche

### 1. Collecte des données — Simu5G

Les données proviennent du simulateur **Simu5G** (OMNeT++ + INET), qui mesure latence, débit, jitter et perte de paquets end-to-end pour chaque slice. **12 scénarios réels** ont été simulés (NormalLoad, GlobalSaturation, FifaWorldCup_Commerce, HetLoad_Asymmetric_A/B/C, KddiOutage_Storm, LowTrafficNight, ModerateLoad_eMBB/URLLC, OverloadeMBB_Commerce, SLABoundary_URLLC) → `all_simu5g.csv` (183 812 lignes).

Un script de génération (`create_rampup.py`) enrichit ce dataset avec **14 scénarios de transition synthétiques** par interpolation sigmoïde entre scénarios LOW et HIGH → `all_simu5g_trans.csv` (255 812 lignes, 26 scénarios). C'est ce fichier qui est utilisé pour l'entraînement et l'évaluation.

### 2. Modèle LSTM — Prédiction de charge

Architecture **Seq2Seq avec attention de Bahdanau** :
- Entrée : 60 secondes × 15 features (throughput, latence, jitter, load, loss + dérivées)
- Sortie : prédiction sur les 15 prochaines secondes (throughput, latence log1p, loss)
- Encodeur : LSTM bidirectionnel, 3 couches, 256 hidden
- Décodeur : LSTM + Bahdanau Attention

Entraîné avec **FedAvg LoGO** (Leave-one-gNB-out) + **FedProx** (μ=0.01) : chaque gNB entraîne un modèle local, les poids sont agrégés pour produire un modèle global par slice.

### 3. Optimisation MILP — Allocation des RBs

Le MILP minimise la fonction de coût **F(R)** :

```
F(R) = Σ_s [ 0.6 × pénalité_débit(s) + 0.4 × p_loss(s) ]
```

Où `p_loss(ρ)` est une sigmoïde calibrée sur les données Simu5G, et `ρ` est le taux de charge prédit par le LSTM.

Contrainte : `Σ_s R_s = R_MAX` (budget total de l'antenne).

### 4. Boucle fermée

```
t=0s  : mesure des KPIs → LSTM prédit ρ à t+15s
t=0s  : MILP calcule R_optimal pour t+15s  (<10ms)
t=15s : nouvelle allocation appliquée avant la surcharge
```

---

## Résultats détaillés

### Gains LSTM + MILP vs baseline (notebook 03)

| gNB | Gain moyen F(R) |
|---|---|
| Macro | +38.6% |
| Commerce | +15.2% |
| Industrie | +62.3% |
| **Global** | **+41.3%** |

### Comparaison des approches (notebook 02 vs 03)

| | MILP seul | LSTM + MILP |
|---|---|---|
| Quand il agit | Après la surcharge | Avant la surcharge |
| Gain moyen | ~20–25% | **41.3%** |
| Apport du LSTM | — | +17 points |

Le LSTM **double l'efficacité** du MILP grâce à l'anticipation.

### Apprentissage continu (notebook 04)

| Métrique | Début | Fin | Gain |
|---|---|---|---|
| MAE prédiction | 0.050 | 0.017 | −66% |
| Gain F(R) | 55.6% | 48.1% | — |
| Train loss | 244 | 62 | −75% |

FedAvg toutes les 5 fenêtres entre gNBs. 1272 updates effectués. **Gain global : 51.8%.**

---

## Installation et lancement

### Prérequis

```
Python 3.10+
PyTorch 2.0+ (CUDA recommandé)
OMNeT++ 6.0 + INET 4.6 + Simu5G 1.4.2  (pour les simulations)
```

### Dépendances Python

```bash
pip install -r requirements.txt
```

### Reproduire les résultats offline (notebooks)

```bash
cd work/simu5g/loss

# 1. Exploration des données
jupyter notebook 00_eda_dataset.ipynb

# 2. Entraînement LSTM (nécessite GPU, ~2-4h)
jupyter notebook 01_lstm_retrain.ipynb

# 3. Optimisation MILP + calibration
jupyter notebook 02_rb_optimization_F.ipynb

# 4. Boucle fermée statique → résultat 41.3%
jupyter notebook 03_closed_loop.ipynb

# 5. Boucle fermée avec apprentissage continu → résultat 51.8%
jupyter notebook 04_closed_loop_online.ipynb
```

### Lancer la comparaison live (Simu5G requis)

```bash
cd work/simu5g/simulation_pfe

# 3 modes en parallèle : baseline / MILP / LSTM+MILP
./run_comparison.sh GlobalSaturation 0

# Analyser les résultats
python compare_results.py comparison_results/GlobalSaturation_seed0
```

### Extraction des métriques Simu5G

```bash
python work/simu5g/extract_simu5g_metrics.py \
    --results work/simu5g/loss/resultat \
    --output  work/simu5g/loss/all_simu5g.csv
```

---

## Références

1. G. Nardini, G. Stea, A. Virdis, **"Simu5G — An OMNeT++ Library for End-to-End Performance Evaluation of 5G Networks"**, IEEE Access, vol. 8, 2020.
2. B. McMahan et al., **"Communication-Efficient Learning of Deep Networks from Decentralized Data"** (FedAvg), AISTATS 2017.
3. T. Li et al., **"Federated Optimization in Heterogeneous Networks"** (FedProx), MLSys 2020.
4. D. Bahdanau et al., **"Neural Machine Translation by Jointly Learning to Align and Translate"**, ICLR 2015.
5. **3GPP TS 23.501** — System architecture for the 5G System, V17.5.0, 2022.
6. **3GPP TS 22.261** — Service requirements for the 5G system, Release 17.
