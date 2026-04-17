# Network Slicing 5G — Boucle Fermée LSTM + MILP

**Projet de Fin d'Études (PFE)**  
Optimisation dynamique des ressources radio en réseau 5G hétérogène par prédiction LSTM et allocation MILP en boucle fermée.

---

## Résultats

| Approche | Réduction coût F(R) |
|---|---|
| Allocation fixe (baseline) | 0 % — référence |
| MILP seul oracle (nb02) | 20,7 % |
| **LSTM + MILP boucle fermée (nb03)** | **39,7 %** |
| LSTM + MILP + online learning (nb04) | 37,2 % global / **40,8 % régime établi** |

Évalué sur **762 fenêtres de test** (split 20 % final) issues de **26 scénarios** (12 réels + 14 synthétiques), 3 gNBs hétérogènes.

**Validation live Simu5G :** LSTM+MILP réduit les violations SLA URLLC de **−5,6 pp** (32,4 % → 26,9 %) vs 0 pp pour le MILP réactif seul.

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
│   │   ├── 03_closed_loop.ipynb           # Boucle fermée LSTM+MILP (39,7 %)
│   │   ├── 04_closed_loop_online.ipynb    # Boucle fermée + online learning (40,8 % régime)
│   │   ├── all_simu5g_trans.csv           # Dataset d'entraînement (255 812 lignes, 26 scénarios)
│   │   ├── create_rampup.py               # Génération scénarios de transition synthétiques
│   │   └── models_lstm_v3/                # Modèles LSTM finaux (Git LFS)
│   │       ├── model_final_{eMBB,URLLC,mMTC}.pt
│   │       └── scalers_final_{eMBB,URLLC,mMTC}.pkl
│   │
│   ├── simulation_pfe/                    # Boucle fermée temps réel (Simu5G)
│   │   ├── omnetpp_hetnet.ini             # Config OMNeT++ (scénarios dont ControllerDemo_v2)
│   │   ├── HetNetSlicing.ned              # Topologie réseau hétérogène
│   │   ├── network_config.xml             # Config réseau IP/5G
│   │   ├── sliceConfig*.xml               # Allocation RBs par gNB
│   │   ├── closed_loop_controller.py      # Contrôleur LSTM + MILP temps réel
│   │   ├── periodic_controller.py         # Contrôleur MILP réactif (comparaison)
│   │   ├── compare_results.py             # Analyse comparative des 3 modes
│   │   └── run_comparison.sh              # Lance baseline / MILP / LSTM+MILP
│   │
│   ├── output/                            # Graphiques et CSV des notebooks 03/04
│   ├── Simu5G/                            # Submodule Simu5G modifié
│   │   └── src/simu5g/control/            # SliceController + SliceResourceManager (custom)
│   └── extract_simu5g_metrics.py          # Extraction .vec/.sca → CSV (si re-simulation)
│
├── TECHNICAL_OVERVIEW.md                  # Documentation technique complète
├── requirements.txt
└── README.md
```

---

## Contexte

Un réseau 5G partage ses ressources radio (Resource Blocks) entre 3 types de slices :

| Slice | Usage | Latence max | Pkt loss max |
|---|---|---|---|
| **eMBB** | Streaming, vidéo HD | 60 ms | 1 % |
| **URLLC** | Robotique, chirurgie à distance | 25 ms | 0,001 % |
| **mMTC** | Capteurs IoT | 300 ms | 1 % |

Le réseau simule **3 gNBs hétérogènes** (Macro 50 RBs, Commerce 35 RBs, Industrie 25 RBs).

Avec une allocation fixe par défaut : **48 % de violations SLA pour eMBB, 61 % pour URLLC**.

---

## Approche

### 1. Collecte des données — Simu5G

Les données proviennent du simulateur **Simu5G** (OMNeT++ + INET), qui mesure latence, débit et perte de paquets end-to-end pour chaque slice. 12 scénarios réels + 14 scénarios de transition synthétiques (générés par `create_rampup.py`) → `all_simu5g_trans.csv` (**255 812 lignes, 26 scénarios**). C'est ce fichier qui est utilisé pour l'entraînement et l'évaluation.

### 2. Modèle LSTM — Prédiction de charge

Architecture **Seq2Seq avec attention de Bahdanau** :
- Entrée : 60 secondes × 15 features (throughput, latence, jitter, load, loss + dérivées)
- Sortie : prédiction sur les 15 prochaines secondes (throughput, latence, loss)
- Encodeur : BiLSTM 3 couches, hidden=256
- Décodeur : LSTM + Bahdanau Attention

Entraîné avec **FedAvg LoGO** (Leave-one-gNB-out) + **FedProx** (μ=0,01).

### 3. Optimisation MILP — Allocation des RBs

Le MILP minimise la fonction de coût **F(R)** :

```
F(R) = Σ_s [ 0.6 × pénalité_débit(s) + 0.4 × p_loss(ρ_s) ]
```

Où `p_loss(ρ)` est une sigmoïde calibrée par régression sur les données Simu5G.  
Résolu par `scipy.optimize.milp` en < 5 ms par fenêtre.

### 4. Boucle fermée

```
t=0s  : observation KPIs → LSTM prédit ρ à t+15s
t=0s  : MILP calcule R_optimal pour ρ prédit  (<5ms)
t=15s : nouvelle allocation appliquée avant la surcharge
```

---

## Résultats détaillés

### Gains LSTM + MILP vs baseline (notebook 03)

| gNB | Gain moyen F(R) |
|---|---|
| Macro | 35,0 % |
| Commerce | 14,9 % |
| Industrie | **62,1 %** |
| **Global** | **39,7 %** |

### Comparaison des approches

| | MILP seul oracle | LSTM + MILP |
|---|---|---|
| Quand il agit | Sur ρ observé | Sur ρ prédit (+15s) |
| Gain moyen | 20,7 % | **39,7 %** |
| Apport du LSTM | — | +19 points |

### Online learning (notebook 04)

| | Première moitié (warmup) | Deuxième moitié (régime) |
|---|---|---|
| Gain F(R) | 33,6 % | **40,8 %** |
| MAE prédiction | 0,0342 | 0,0272 |
| Train loss | 0,207 | 0,159 |

Train loss total : 0,183 → 0,006 (×29 de réduction). FedAvg toutes les 5 fenêtres pour eMBB.

### Validation live Simu5G (ControllerDemo_v2, t=300s)

| Mode | URLLC Viol% | Gain vs baseline |
|---|---|---|
| Baseline (allocation fixe) | 32,4 % | — |
| MILP périodique (réactif) | 32,4 % | +0,0 pp |
| **LSTM + MILP adaptatif** | **26,9 %** | **−5,6 pp** |

Le MILP réactif échoue car les violations URLLC (SLA : pkt_loss < 0,001 %) sont causées par des micro-rafales. Le LSTM anticipe 15 s à l'avance et pré-alloue avant la rafale.

---

## Installation et lancement

### Prérequis

```
Python 3.10+
PyTorch 2.0+ (CUDA recommandé)
OMNeT++ 6.0 + INET 4.6 + Simu5G 1.4.2  (pour la validation live uniquement)
```

### Dépendances Python

```bash
pip install -r requirements.txt
```

### Reproduire les résultats offline (notebooks)

```bash
cd work/simu5g/loss

jupyter notebook 00_eda_dataset.ipynb        # Exploration des données
jupyter notebook 01_lstm_retrain.ipynb       # Entraînement LSTM (~2-4h GPU)
jupyter notebook 02_rb_optimization_F.ipynb  # MILP seul + calibration
jupyter notebook 03_closed_loop.ipynb        # Boucle fermée → 39,7 %
jupyter notebook 04_closed_loop_online.ipynb # Online learning → 40,8 % régime
```

### Validation live (Simu5G requis)

```bash
cd work/simu5g/simulation_pfe

# Lance les 3 modes : baseline / MILP périodique / LSTM+MILP
./run_comparison.sh ControllerDemo_v2 0

# Les résultats s'affichent en fin de run
```

---

## Documentation

Le fichier [`TECHNICAL_OVERVIEW.md`](TECHNICAL_OVERVIEW.md) documente en détail chaque notebook, les choix d'architecture, les résultats par gNB et la validation live.

---

## Références

1. G. Nardini, G. Stea, A. Virdis, **"Simu5G — An OMNeT++ Library for End-to-End Performance Evaluation of 5G Networks"**, IEEE Access, vol. 8, 2020.
2. B. McMahan et al., **"Communication-Efficient Learning of Deep Networks from Decentralized Data"** (FedAvg), AISTATS 2017.
3. T. Li et al., **"Federated Optimization in Heterogeneous Networks"** (FedProx), MLSys 2020.
4. D. Bahdanau et al., **"Neural Machine Translation by Jointly Learning to Align and Translate"**, ICLR 2015.
5. **3GPP TS 23.501** — System architecture for the 5G System, V17.5.0, 2022.
6. **3GPP TS 22.261** — Service requirements for the 5G system, Release 17.
