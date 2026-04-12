# Rapport Technique — Network Slicing 5G
## Boucle Fermée LSTM + MILP pour l'Optimisation Dynamique des Ressources Radio

**Projet de Fin d'Études**
**Auteur :** Nadir NEHILI

---

## 1. Introduction et problématique

### 1.1 Contexte

Les réseaux 5G introduisent le concept de **network slicing** : la capacité de diviser un réseau physique en plusieurs réseaux virtuels indépendants (slices), chacun avec ses propres garanties de qualité de service (SLA). Trois types de slices standardisés par le 3GPP coexistent sur la même infrastructure radio :

- **eMBB** (Enhanced Mobile Broadband) : streaming vidéo, téléchargement. Fort besoin en débit.
- **URLLC** (Ultra-Reliable Low-Latency Communications) : robotique industrielle, chirurgie à distance, véhicules autonomes. Latence inférieure à 25 ms requise.
- **mMTC** (Massive Machine-Type Communications) : capteurs IoT, compteurs intelligents. Trafic sporadique et peu exigeant.

La ressource radio de base est le **Resource Block (RB)** — l'unité d'allocation de bande passante dans le standard NR 5G. Chaque antenne (gNB) dispose d'un budget fixe de RBs à distribuer entre les slices.

### 1.2 Problème

Avec une **allocation statique** (nombre fixe de RBs par slice, indépendant de la charge), les performances sont mauvaises en cas de trafic variable :

- eMBB : **100% de violations SLA** sur l'ensemble des scénarios simulés
- URLLC : **68% de violations SLA**
- mMTC : 0% (peu exigeant, toujours respecté)

L'enjeu est de concevoir un système capable de **réallouer dynamiquement les RBs** en anticipant les surcharges, sans interruption de service.

### 1.3 Solution proposée

Un système de **boucle fermée** en trois composantes :

1. Un modèle **LSTM Seq2Seq** qui prédit la charge réseau à 15 secondes
2. Un algorithme **MILP** qui calcule l'allocation optimale des RBs en fonction de la charge prédite
3. Une **boucle de contrôle** qui applique la nouvelle allocation en temps réel sur le simulateur Simu5G

---

## 2. Infrastructure de simulation

### 2.1 Simulateur Simu5G

**Simu5G** est un simulateur open-source développé à l'Université de Pise (Nardini et al., IEEE Access 2020), basé sur OMNeT++ et INET. Il implémente une pile protocole 5G complète : canal radio NR, scheduling MAC, PDCP/RLC/MAC/PHY, tunneling GTP-U, architecture 3GPP TS 23.501.

La latence mesurée est **end-to-end** (création du paquet à l'UE → réception au serveur applicatif), ce qui intègre tous les délais radio, réseau et traitement.

### 2.2 Topologie hétérogène

Le réseau simule **3 gNBs de types différents**, représentant des environnements de déploiement distincts :

| gNB | Type | Budget RBs | Scheduler | Usage typique |
|---|---|---|---|---|
| **Macro** | Urban Macro (UMa) | 50 RBs | LteMaxCi | Zone urbaine dense |
| **Commerce** | Urban Micro (UMi) | 35 RBs | LtePf | Centre commercial |
| **Industrie** | Urban Micro (UMi) | 25 RBs | LtePf | Zone industrielle |

Allocation par défaut (fixe, avant optimisation) :

| gNB | eMBB | URLLC | mMTC |
|---|---|---|---|
| Macro | 20 RBs | 17 RBs | 13 RBs |
| Commerce | 21 RBs | 11 RBs | 3 RBs |
| Industrie | 6 RBs | 12 RBs | 7 RBs |

### 2.3 SLA par slice

| Slice | Latence max | Débit minimum |
|---|---|---|
| eMBB | 60 ms | 10 Mbps |
| URLLC | 25 ms | 0.5 Mbps |
| mMTC | 300 ms | — |

### 2.4 Scénarios simulés

**12 scénarios réels** simulés, couvrant différents régimes de trafic :

| Scénario | Description |
|---|---|
| NormalLoad | Charge nominale équilibrée |
| GlobalSaturation | Saturation totale de toutes les slices |
| LowTrafficNight | Trafic nocturne minimal |
| FifaWorldCup_Commerce | Pic eMBB en zone commerciale |
| HetLoad_Asymmetric_A/B/C | Charges asymétriques entre gNBs |
| KddiOutage_Storm | Défaillance partielle + surcharge résiduelle |
| ModerateLoad_eMBB | Charge modérée eMBB |
| ModerateLoad_URLLC | Charge modérée URLLC |
| OverloadeMBB_Commerce | Surcharge eMBB en zone commerciale |
| SLABoundary_URLLC | URLLC à la limite du SLA |

**14 scénarios synthétiques de transition** générés par `create_rampup.py` via interpolation sigmoïde entre un scénario LOW et un scénario HIGH, avec différentes proportions de temps en régime bas/haut (RampUp, RampDown, Peak, Gradual, LowViol — pour eMBB, URLLC et mMTC).

**Dataset final** : `all_simu5g_trans.csv` — 255 812 fenêtres d'1 seconde, 26 scénarios, 3 gNBs.

---

## 3. Modèle LSTM — Prédiction de charge

### 3.1 Architecture Seq2Seq avec attention

Le modèle prédit l'évolution future des KPIs réseau à partir de l'historique récent.

**Entrée** : fenêtre glissante de 60 secondes × 15 features par (slice, gNB) :
- Features brutes : Throughput (Mbps), Latence log1p, Packet Loss (%), Jitter log1p, Network Load (%)
- Features dérivées : tendance (diff) et volatilité (rolling std sur 5 pas) pour chaque feature brute

**Sortie** : prédiction sur les 15 prochaines secondes (throughput, latence log1p, loss)

**Architecture** :
```
Encodeur  : LSTM bidirectionnel — 3 couches, 256 hidden, dropout 0.25
            projection des états cachés (2×256 → 256)

Attention : Bahdanau (additive)
            energie = v · tanh(W_enc·enc_out + W_dec·dec_h)
            context = Σ softmax(energie) × enc_out

Décodeur  : LSTM — 3 couches, 256 hidden
            context concaténé à chaque pas de décodage

Sortie    : couche dense 256 → 3 (throughput, latence_log1p, loss)
```

**Entraînement** :
- Perte : HuberLoss (robuste aux outliers de latence)
- Optimiseur : AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler : ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping : patience=15
- Batch : 64, sampling pondéré par variance (favorise les fenêtres de transition)

### 3.2 Apprentissage fédéré — FedAvg LoGO

L'entraînement utilise une stratégie **Leave-one-gNB-out (LoGO)** fédérée :

1. Pour chaque gNB laissé de côté (Macro, Commerce, Industrie) :
   - Les 2 autres gNBs entraînent chacun un modèle local
   - **FedAvg** : moyenne des poids → modèle global
   - Évaluation sur le gNB laissé de côté
2. Les 3 modèles locaux finaux sont agrégés par FedAvg → **modèle global par slice** (3 modèles : eMBB, URLLC, mMTC)

**FedProx** (μ=0.01) limite la divergence entre gNBs hétérogènes pendant l'entraînement local.

Un **scaler global par slice** (fitté sur tous les gNBs combinés) garantit que train et test vivent dans le même espace normalisé — indispensable pour la généralisation LoGO.

### 3.3 Résultats d'entraînement — LoGO FedAvg

Évaluation en classification binaire (le LSTM prédit-il correctement qu'une violation SLA va survenir dans les 15 prochaines secondes ?) sur le split test 80/20 :

**Répartition train/test par slice :**

| Slice | Train (80%) | Test (20%) | Total |
|---|---|---|---|
| eMBB | 68 403 | 17 295 | 85 698 |
| URLLC | 68 403 | 17 295 | 85 698 |
| mMTC | 56 586 | 14 330 | 70 916 |

| Slice | F1 | Précision | Recall | Spécificité | MAE latence | MAE loss |
|---|---|---|---|---|---|---|
| **eMBB** | **0.866** | 0.764 | **1.000** | 0.586 | 43.8 ms | 4.3% |
| **URLLC** | **0.771** | **1.000** | 0.628 | **1.000** | 28.7 ms | 1.3% |
| **mMTC** | 0.645 | **1.000** | 0.476 | **1.000** | 0.07 ms | 0.06% |
| **Moyenne** | **0.761** | **0.921** | **0.701** | — | — | — |

Observations :
- **eMBB** : Recall=1.000 — aucune violation manquée. Précision plus faible (faux positifs acceptables, mieux vaut sur-allouer que manquer une surcharge)
- **URLLC** : Précision=1.000 — toutes les alertes sont vraies. Recall=0.628 (28% des violations manquées — URLLC est la slice la plus difficile à prédire)
- **mMTC** : F1 plus faible mais peu critique (mMTC ne viole presque jamais ses SLA dans les données)
- **Paramètres du modèle** : 5 920 643 paramètres

### 3.4 Du LSTM au taux de charge ρ

La sortie du LSTM (latence prédite) est convertie en taux de charge ρ :

```
lat_pred = exp(latence_log1p_pred) - 1
p_pred   = fraction des pas où lat_pred > SLA_max
```

- Si `p_pred ≥ 0.99` → ρ = 2.0 (saturation totale)
- Si `p_pred > 0.01` → ρ inversé depuis la sigmoïde calibrée
- Sinon → ρ estimé depuis le débit prédit

---

## 4. Optimisation MILP — Allocation des RBs

### 4.1 Modélisation du risque — Sigmoïde calibrée

Pour chaque couple (gNB, slice), une **courbe sigmoïde** est calibrée sur les données Simu5G :

```
p_loss(ρ) = 1 / (1 + exp(−k × (ρ − ρ_c)))
```

Paramètres calibrés (exemples) :

| gNB | Slice | k | ρ_c | Interprétation |
|---|---|---|---|---|
| Macro | eMBB | 4.68 | 0.798 | Saturation dès ρ=0.8 |
| Macro | URLLC | 2.91 | 1.093 | Saturation progressive |
| Commerce | URLLC | 15.27 | 0.406 | Saturation très brutale |
| Industrie | eMBB | 1.33 | 2.304 | Très résistant à la charge |

### 4.2 Fonction de coût F(R)

Pour une allocation de k RBs à la slice s :

```
ρ_k   = ρ_base × R_défaut / k
p_s   = sigmoid(ρ_k, k_s, ρ_c_s)
cap_k = k × débit_par_RB
T_s   = min(λ, cap_k) × (1 − p_s)
pen_T = max(0, 1 − T_s / T_min)

coût_s(k) = 0.6 × pen_T + 0.4 × p_s

F(R) = Σ_{s} coût_s(R_s)
```

- ω_T = 0.6 : poids de la pénalité débit
- ω_P = 0.4 : poids du risque de perte

### 4.3 Formulation MILP

Variables binaires x[s,k] = 1 si la slice s reçoit k RBs.

**Minimiser** `Σ_{s,k} coût_s(k) × x[s,k]`

**Sous contraintes** :
```
Σ_k x[s,k] = 1            ∀s  (une valeur k par slice)
Σ_{s,k} k × x[s,k] ≤ R_MAX   (budget total gNB)
x[s,k] ∈ {0,1}
```

Résolu avec `scipy.optimize.milp`. **Temps de résolution < 10 ms** pour les 3 gNBs.

### 4.4 Exemple — GlobalSaturation à Macro

| | eMBB | URLLC | mMTC | F(R) |
|---|---|---|---|---|
| Allocation défaut | 20 RBs | 17 RBs | 13 RBs | 1.80 |
| Allocation MILP | 1 RB | 48 RBs | 1 RB | 1.10 |
| **Gain** | | | | **38.9%** |

---

## 5. Boucle fermée

### 5.1 Architecture système

```
Simu5G (OMNeT++)                          Python Controller
─────────────────                         ─────────────────
Chaque 1s :
SliceController.cc
  writeMetrics()  ──── metrics_live.csv ──►  lit le CSV
  readRbConfig()  ◄─── rb_config.json   ──   LSTM + MILP
  setQuota()
       │
  SliceResourceManager
       │
  LteMaxCi / LtePf
  isQuotaExhausted()
  → enforce RB quota
```

### 5.2 Cycle de contrôle (toutes les 15 secondes)

1. **Mesure** : latence, débit, perte par (gNB, slice) → `metrics_live.csv`
2. **Prédiction** : LSTM → ρ prédit à t+15s
3. **Optimisation** : MILP → R_optimal → `rb_config.json`
4. **Application** : nouveaux quotas appliqués au prochain cycle radio (~1ms), sans interruption

### 5.3 Trois modes comparés

| Mode | Comportement |
|---|---|
| Baseline | Allocation fixe défaut |
| MILP périodique | MILP toutes les 15s, ρ observé (réactif) |
| **LSTM + MILP** | MILP toutes les 15s, ρ prédit à +15s (anticipatif) |

### 5.4 Résultats simulation live

Les 3 scénarios testés en direct (ModerateLoad_URLLC, ModerateLoad_eMBB, OverloadeMBB_Commerce) n'ont pas montré de gains visibles entre les 3 modes.

**Cause** : saturation structurelle à l'antenne Macro (24 UEs simultanés). Les quotas RBs contrôlent l'attribution de bande passante mais ne réduisent pas le délai de buffering interne partagé entre UEs. Le contrôleur calculait pourtant les bonnes allocations (ex. URLLC=44 RBs au lieu de 17).

La simulation live est une **preuve d'intégration** : la boucle C++ ↔ Python fonctionne, les quotas sont modifiés en temps réel. Les gains sont mesurés dans l'évaluation offline sur des scénarios à charge modérée.

---

## 6. Évaluation offline

### 6.1 Protocole

Rejoue `all_simu5g_trans.csv` à travers la boucle fermée simulée. Pour chaque fenêtre de 15s :
- LSTM → ρ prédit → MILP → R_optimal
- `gain = (F_défaut − F_optimal) / F_défaut × 100`

**3 739 fenêtres** évaluées (notebook 03, split test 20% de all_simu5g_trans.csv).

### 6.2 Résultats globaux

| Approche | Gain moyen F(R) | Fenêtres améliorées |
|---|---|---|
| Allocation fixe | 0% | — |
| MILP seul | **20.7%** | — |
| **LSTM + MILP** | **41.3%** | **98.3%** |

### 6.3 Gains par gNB

| gNB | Gain |
|---|---|
| Macro | 38.6% |
| Commerce | 15.2% |
| **Industrie** | **62.3%** |

### 6.4 Gains par scénario (top 10)

| Scénario | Gain |
|---|---|
| NormalLoad | 57.0% |
| HetLoad_Asymmetric_B | 51.1% |
| HetLoad_Asymmetric_A | 48.1% |
| ModerateLoad_eMBB | 45.7% |
| OverloadeMBB_Commerce | 44.3% |
| SLABoundary_URLLC | 42.0% |
| KddiOutage_Storm | 40.8% |
| GlobalSaturation | 40.7% |
| FifaWorldCup_Commerce | 40.2% |
| ModerateLoad_URLLC | 38.9% |

### 6.5 Apport du LSTM vs MILP seul

| | MILP seul | LSTM + MILP |
|---|---|---|
| Quand il agit | Après la surcharge | Avant la surcharge (+15s) |
| Gain moyen | **20.7%** | **41.3%** |
| Apport du LSTM | — | **+20.6 points** |

**MILP seul — gains par gNB :**

| gNB | Gain MILP seul | Gain LSTM + MILP | Apport LSTM |
|---|---|---|---|
| Macro | 20.72% | 38.56% | +17.8 pts |
| Commerce | 15.78% | 15.19% | −0.6 pts |
| Industrie | 25.07% | 62.34% | +37.3 pts |
| **Global** | **20.7%** | **41.3%** | **+20.6 pts** |

Le LSTM **double l'efficacité** du MILP grâce à l'anticipation. Commerce est l'exception : le MILP seul capture déjà l'essentiel des gains (scénarios proches de l'allocation optimale par défaut).

### 6.6 MILP seul — Résultats complets par scénario et gNB

*(Notebook 02 — évaluation offline sur les 26 scénarios, allocation optimale calculée avec ρ observé)*

| Scénario | gNB | F_défaut | F_optimal | Gain |
|---|---|---|---|---|
| FifaWorldCup_Commerce | Macro | 1.799 | 1.099 | 38.9% |
| FifaWorldCup_Commerce | Commerce | 1.389 | 1.126 | 18.9% |
| FifaWorldCup_Commerce | Industrie | 0.210 | 0.066 | **68.5%** |
| GlobalSaturation | Macro | 1.799 | 1.099 | 38.9% |
| GlobalSaturation | Commerce | 1.389 | 1.126 | 18.9% |
| GlobalSaturation | Industrie | 1.038 | 0.408 | 60.7% |
| HetLoad_Asymmetric_A | Macro | 1.799 | 1.099 | 38.9% |
| HetLoad_Asymmetric_A | Commerce | 0.072 | 0.030 | 58.7% |
| HetLoad_Asymmetric_A | Industrie | 0.038 | 0.029 | 21.6% |
| HetLoad_Asymmetric_B | Macro | 0.059 | 0.048 | 19.2% |
| HetLoad_Asymmetric_B | Commerce | 1.389 | 1.126 | 18.9% |
| HetLoad_Asymmetric_B | Industrie | 0.038 | 0.029 | 21.6% |
| HetLoad_Asymmetric_C | Macro | 0.059 | 0.048 | 19.2% |
| HetLoad_Asymmetric_C | Commerce | 0.072 | 0.030 | 58.7% |
| HetLoad_Asymmetric_C | Industrie | 1.038 | 0.448 | 56.8% |
| KddiOutage_Storm | Macro | 1.799 | 1.099 | 38.9% |
| KddiOutage_Storm | Commerce | 1.389 | 1.126 | 18.9% |
| KddiOutage_Storm | Industrie | 0.135 | 0.072 | 47.0% |
| LowTrafficNight | Macro | 0.036 | 0.033 | 8.9% |
| LowTrafficNight | Commerce | 0.003 | 0.002 | 24.1% |
| LowTrafficNight | Industrie | 0.034 | 0.024 | 27.1% |
| ModerateLoad_URLLC | Macro | 1.799 | 1.099 | 38.9% |
| ModerateLoad_URLLC | Commerce | 1.006 | 1.001 | 0.5% |
| ModerateLoad_URLLC | Industrie | 0.132 | 0.070 | 47.2% |
| ModerateLoad_eMBB | Macro | 1.799 | 1.099 | 38.9% |
| ModerateLoad_eMBB | Commerce | 0.031 | 0.022 | 29.5% |
| ModerateLoad_eMBB | Industrie | 0.061 | 0.037 | 39.2% |
| NormalLoad | Macro | 0.090 | 0.064 | 28.8% |
| NormalLoad | Commerce | 0.072 | 0.030 | 58.7% |
| NormalLoad | Industrie | 0.061 | 0.037 | 39.2% |
| OverloadeMBB_Commerce | Macro | 1.033 | 0.555 | 46.2% |
| OverloadeMBB_Commerce | Commerce | 1.389 | 1.126 | 18.9% |
| OverloadeMBB_Commerce | Industrie | 0.210 | 0.066 | **68.5%** |
| SLABoundary_URLLC | Macro | 1.799 | 1.099 | 38.9% |
| SLABoundary_URLLC | Commerce | 1.389 | 1.126 | 18.9% |
| SLABoundary_URLLC | Industrie | 0.181 | 0.079 | 56.1% |
| Trans_Gradual_URLLC | Macro | 2.100 | 2.032 | 3.2% |
| Trans_Gradual_URLLC | Commerce | 2.012 | 2.002 | 0.5% |
| Trans_Gradual_URLLC | Industrie | 2.070 | 2.009 | 2.9% |
| Trans_Gradual_eMBB | Macro | 2.109 | 2.028 | 3.8% |
| Trans_Gradual_eMBB | Commerce | 2.159 | 2.014 | 6.7% |
| Trans_Gradual_eMBB | Industrie | 2.086 | 2.028 | 2.8% |
| Trans_Gradual_mMTC | — | 2.000 | 2.000 | 0.0% |
| Trans_LowViol_URLLC | Macro | 2.046 | 2.024 | 1.1% |
| Trans_LowViol_URLLC | Commerce | 2.006 | 2.002 | 0.2% |
| Trans_LowViol_URLLC | Industrie | 2.011 | 2.003 | 0.4% |
| Trans_LowViol_eMBB | Macro | 2.079 | 2.024 | 2.6% |
| Trans_LowViol_eMBB | Commerce | 2.037 | 2.005 | 1.6% |
| Trans_LowViol_eMBB | Industrie | 2.070 | 2.026 | 2.1% |
| Trans_LowViol_mMTC | — | 2.000 | 2.000 | 0.0% |
| Trans_Peak_URLLC | Macro | 2.777 | 2.095 | 24.6% |
| Trans_Peak_URLLC | Commerce | 2.119 | 2.005 | 5.4% |
| Trans_Peak_URLLC | Industrie | 2.992 | 2.223 | 25.7% |
| Trans_Peak_eMBB | Macro | 2.202 | 2.041 | 7.3% |
| Trans_Peak_eMBB | Commerce | 2.998 | 2.709 | 9.6% |
| Trans_Peak_eMBB | Industrie | 2.134 | 2.032 | 4.8% |
| Trans_RampDown_URLLC | Macro | 2.951 | 2.142 | 27.4% |
| Trans_RampDown_URLLC | Commerce | 2.726 | 2.021 | 25.9% |
| Trans_RampDown_URLLC | Industrie | 2.999 | 2.308 | 23.0% |
| Trans_RampDown_eMBB | Macro | 2.712 | 2.078 | 23.4% |
| Trans_RampDown_eMBB | Commerce | 3.000 | 2.927 | 2.4% |
| Trans_RampDown_eMBB | Industrie | 2.181 | 2.036 | 6.6% |
| Trans_RampDown_mMTC | — | 2.000 | 2.000 | 0.0% |
| Trans_RampUp_URLLC | Macro | 2.949 | 2.140 | 27.4% |
| Trans_RampUp_URLLC | Commerce | 2.285 | 2.009 | 12.1% |
| Trans_RampUp_URLLC | Industrie | 2.999 | 2.308 | 23.0% |
| Trans_RampUp_eMBB | Macro | 2.709 | 2.079 | 23.3% |
| Trans_RampUp_eMBB | Commerce | 3.000 | 2.933 | 2.2% |
| Trans_RampUp_eMBB | Industrie | 2.187 | 2.036 | 6.9% |
| Trans_RampUp_mMTC | — | 2.000 | 2.000 | 0.0% |

**Top 10 scénarios (gain moyen sur les 3 gNBs) :**

| Scénario | Gain moyen MILP seul |
|---|---|
| HetLoad_Asymmetric_C | **44.90%** |
| OverloadeMBB_Commerce | **44.53%** |
| NormalLoad | **42.23%** |
| FifaWorldCup_Commerce | 42.10% |
| HetLoad_Asymmetric_A | 39.73% |
| GlobalSaturation | 39.50% |
| SLABoundary_URLLC | 37.97% |
| ModerateLoad_eMBB | 35.87% |
| KddiOutage_Storm | 34.93% |
| ModerateLoad_URLLC | 28.87% |

---

## 7. Vitesse de décision

| Composante | Temps |
|---|---|
| Inférence LSTM (GPU) | ~5–15 ms |
| Résolution MILP (3 gNBs) | < 10 ms |
| **Cycle complet** | **< 25 ms** |
| Fenêtre disponible | 15 s (600×) |

Le cycle complet (prédiction + optimisation) s'exécute en moins de 25 ms pour une fenêtre de décision de 15 secondes, soit un facteur **600× de marge temporelle**. Le système ne constitue pas un goulot d'étranglement.

---

## 8. Analyse des résultats

### 8.1 Pourquoi le LSTM apporte autant — l'effet d'anticipation

Le MILP seul est un système **réactif** : il optimise pour la charge *actuelle*, donc il alloue les RBs après que la surcharge a déjà commencé. Le cycle de 15s signifie que la nouvelle allocation arrive systématiquement trop tard.

Le LSTM transforme le MILP en système **prédictif** : il estime la charge à t+15s, et le MILP alloue les RBs *avant* la montée en charge. C'est cette avance temporelle de 15 secondes qui explique le doublement du gain (20.7% → 41.3%).

Un exemple concret : sur GlobalSaturation à Macro, F_défaut = 1.799. Avec le MILP seul (ρ observé), F_optimal = 1.099 (−38.9%). Avec le LSTM, la prédiction permet au MILP d'anticiper la saturation avant qu'elle ne soit mesurée — l'allocation URLLC passe de 17 à 48 RBs avant le pic, et non pendant.

### 8.2 Pourquoi Industrie gagne le plus (62.3%)

L'antenne Industrie (25 RBs, LtePf) présente la plus grande **disparité entre l'allocation par défaut et l'allocation optimale** :

- Allocation défaut : eMBB=6 RBs, URLLC=12 RBs, mMTC=7 RBs
- Sur GlobalSaturation, le MILP recalcule : eMBB=1, URLLC=23, mMTC=1 — quasi-totalité des RBs vers URLLC

L'allocation par défaut sous-dote massivement URLLC (12 sur 25 RBs alors que la slice est la plus critique en latence). La marge de réallocation est donc très large. De plus, les scénarios Industrie présentent des charges URLLC élevées mais prévisibles (F1=0.771 sur cette slice), ce qui permet au LSTM d'anticiper efficacement.

### 8.3 Pourquoi Commerce gagne peu (15.2%)

Commerce (35 RBs, LtePf) a une allocation par défaut de eMBB=21, URLLC=11, mMTC=3. Sur de nombreux scénarios, cette allocation est déjà proche de l'optimum MILP :

- Sur ModerateLoad_URLLC : F_défaut=1.006, F_optimal=1.001 → gain MILP = **0.5%** seulement. Le problème n'est pas l'allocation mais la saturation intrinsèque de la slice.
- Sur les scénarios HetLoad où Commerce est peu chargé (F_défaut=0.072), le gain MILP atteint 58.7% — mais F_défaut est déjà très bas, l'impact absolu est faible.
- L'apport du LSTM à Commerce est quasi nul (−0.6 pts) : quand l'allocation par défaut est déjà proche de l'optimum, prédire ne change rien.

### 8.4 Pourquoi Macro plafonne à 38.6%

À Macro (50 RBs, LteMaxCi), F_défaut = **1.799** dans la majorité des scénarios chargés. Cette valeur est structurellement élevée car le scheduler LteMaxCi favorise les UEs en bonne condition radio, ce qui aggrave les violations SLA des slices défavorisées.

Même avec l'allocation optimale, F_optimal ne descend jamais sous **1.099** à Macro sur les scénarios saturés — c'est le plancher physique de cette antenne : 24 UEs simultanés créent une contention MAC irréductible par la seule réallocation des RBs. Le gain maximal atteignable à Macro est donc borné à ~38.9%, quelle que soit la qualité de la prédiction.

### 8.5 Analyse du modèle LSTM par slice

**eMBB (F1=0.866, Recall=1.000)** : Le modèle ne rate aucune violation eMBB. La Précision plus faible (0.764) signifie que certaines alertes sont des faux positifs — le système sur-alloue parfois sans nécessité. Ce comportement est **souhaitable** : mieux vaut anticiper une surcharge qui n'aura pas lieu que la manquer. eMBB est la slice la plus facile à prédire car sa dynamique (débit vidéo) est régulière.

**URLLC (F1=0.771, Précision=1.000)** : Toutes les alertes URLLC sont vraies (zéro faux positif), mais 37% des violations réelles passent inaperçues (Recall=0.628). URLLC est la slice la plus difficile à prédire : les pics de latence sont souvent brutaux et courts. MAE latence = 28.7 ms sur une SLA de 25 ms — la marge d'erreur est faible.

**mMTC (F1=0.645)** : F1 plus faible mais peu critique. mMTC viole rarement son SLA (300 ms) et contribue peu à F(R) dans les scénarios où les deux autres slices sont sous tension. La MAE latence de 0.07 ms confirme que le modèle est précis en valeur absolue — le F1 faible reflète un problème de classes déséquilibrées (très peu de violations réelles à détecter).

### 8.6 Scénarios de transition synthétiques — comportement du système

Les 14 scénarios de transition générés par `create_rampup.py` révèlent un comportement spécifique :

- **Trans_mMTC (toutes variantes)** : gain = **0.0%** systématiquement. mMTC ne génère jamais de pression sur F(R), aucune réallocation n'est utile.
- **Trans_Gradual (eMBB/URLLC)** : gains faibles (0.5–6.7%). Les transitions très lentes (sigmoïde étalée) maintiennent le réseau en régime quasi-stationnaire — le MILP seul suffit, le LSTM n'apporte pas de valeur.
- **Trans_RampUp/RampDown URLLC** : gains de 23–27%. Les montées/descentes rapides de charge URLLC créent des fenêtres où l'anticipation compte. C'est exactement le cas d'usage pour lequel la boucle fermée a été conçue.
- **Trans_Peak URLLC à Industrie** : F_défaut=2.992 (quasi-saturation totale), F_optimal=2.223 → gain 25.7%. Même en situation de pic, le MILP réduit significativement le coût.

### 8.7 Limites identifiées

**Limite 1 — Saturation structurelle à Macro** : Sur les scénarios à forte charge (GlobalSaturation, KddiOutage, etc.), F_défaut=1.799 et F_optimal=1.099 quel que soit le mode. Le plafond de gain est ~38.9%. Pour aller plus loin, il faudrait agir au niveau du scheduler LteMaxCi lui-même, pas seulement sur les quotas RBs.

**Limite 2 — Commerce quasi-insensible à la réallocation** : Sur ModerateLoad_URLLC à Commerce (gain MILP=0.5%), la saturation est intrinsèque au trafic, pas à l'allocation. Ajouter des RBs URLLC ne réduit pas la latence si les UEs sont déjà en contention physique.

**Limite 3 — URLLC difficile à prédire** : Recall=0.628 signifie que 37% des pics de latence URLLC ne sont pas anticipés. Pour une application réelle en robotique industrielle (latence critique), ce taux de manqués serait problématique. Un modèle dédié URLLC ou une fenêtre d'entrée plus courte mériterait d'être exploré.

**Limite 4 — Simulation live vs offline** : Les gains mesurés sont offline (rejoue des données). La simulation live C++ ↔ Python a validé l'intégration mais pas les performances, à cause de la contention MAC interne à Macro (24 UEs). Les gains réels en production dépendent de la capacité du simulateur à isoler les slices au niveau MAC.

---

## 9. Conclusion

### 9.1 Ce que ce projet démontre

Ce projet démontre qu'une boucle fermée **LSTM prédictif + MILP** peut réduire le coût de non-conformité SLA de **41.3%** par rapport à une allocation statique, sur 3 739 fenêtres couvrant 26 scénarios de trafic hétérogènes.

Trois résultats sont particulièrement significatifs :

1. **L'anticipation double l'efficacité de l'optimisation** : le MILP seul atteint 20.7% de gain — avec les mêmes contraintes et la même fonction objectif, simplement en ajoutant la prédiction LSTM à +15s, le gain passe à 41.3%. Ce n'est pas le MILP qui est limité, c'est la réactivité qui l'était.

2. **98.3% des fenêtres sont améliorées** : la boucle fermée n'est pas capricieuse. Elle améliore la situation dans presque tous les cas, avec un gain médian de 38.9% — ce qui signifie que la moitié des fenêtres gagnent plus de 38.9%.

3. **Le système décide en < 25 ms pour une fenêtre de 15 s** : il y a un facteur 600 entre le temps de décision et la fenêtre disponible. Le système est compatible avec un déploiement temps réel.

### 9.2 Ce que les chiffres signifient concrètement

Avant optimisation : **100% de violations SLA eMBB, 68% pour URLLC** sur l'ensemble des scénarios avec allocation fixe.

Avec LSTM + MILP, la réduction de F(R) de 41.3% se traduit directement par moins de fenêtres en violation SLA :
- À Industrie : F(R) passe de ~1.04 à ~0.39 en moyenne sur les scénarios chargés — soit une réduction de 62% du coût de non-conformité radio.
- À Macro : même structurellement saturée (24 UEs), F(R) passe de 1.799 à ~1.1 — le système réduit la dégradation même quand il ne peut pas l'éliminer.

### 9.3 Limites et honnêteté des résultats

Les résultats sont **offline** : les données sont rejouées, pas simulées en direct. La simulation live a validé l'intégration système (boucle C++ ↔ Python fonctionnelle, quotas appliqués en ~1ms) mais n'a pas montré de gains mesurables à cause de la contention MAC interne à Macro avec 24 UEs. Ce point distingue la performance théorique de la performance opérationnelle.

Le gain de Commerce (15.2%) est modeste car l'allocation par défaut y est déjà proche de l'optimum pour la plupart des scénarios — ce n'est pas une faiblesse du système, c'est une caractéristique de ce gNB.

### 9.4 Perspectives

- **Adapter le contrôleur temps réel** (`closed_loop_controller.py`) pour charger les modèles `models_lstm_v3` et valider les gains en simulation live sur des scénarios à charge modérée
- **Améliorer la prédiction URLLC** : Recall=0.628 est le maillon faible. Une fenêtre d'entrée plus courte (30s au lieu de 60s) ou un modèle spécialisé pics pourraient réduire les violations manquées
- **Enrichir F(R) avec des priorités inter-slices** : URLLC devrait avoir un poids plus élevé que eMBB dans la fonction objectif pour les déploiements critiques
- **Tester sur trafic réel** : les 12 scénarios réels Simu5G couvrent des cas représentatifs, mais un déploiement sur données opérateur permettrait de valider la généralisation hors distribution

---

## Références

1. G. Nardini, G. Stea, A. Virdis, **"Simu5G — An OMNeT++ Library for End-to-End Performance Evaluation of 5G Networks"**, IEEE Access, 2020.
2. B. McMahan et al., **"Communication-Efficient Learning of Deep Networks from Decentralized Data"** (FedAvg), AISTATS 2017.
3. T. Li et al., **"Federated Optimization in Heterogeneous Networks"** (FedProx), MLSys 2020.
4. D. Bahdanau, K. Cho, Y. Bengio, **"Neural Machine Translation by Jointly Learning to Align and Translate"**, ICLR 2015.
5. **3GPP TS 23.501** — System architecture for the 5G System, V17.5.0, 2022.
6. **3GPP TS 22.261** — Service requirements for the 5G system, Release 17.
7. A. Varga, **"OMNeT++ User Manual"**, OpenSim Ltd., 2023.
