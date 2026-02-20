# Guide simu5G - Network Slicing PFE

## Prerequis

- **Windows 10/11** avec opp_env
- **Python 3.10+** pour le script de conversion des traces

---

## Installation complete avec opp_env (Windows)

### 1. Installer opp_env

Telecharger et lancer l'installateur opp_env depuis :
https://github.com/omnetpp/opp_env/releases

Au premier lancement, un terminal Linux s'ouvre automatiquement.

### 2. Installer OMNeT++ et INET

Dans le terminal opp_env :

```bash
opp_env install inet-4.5.4
```

L'installation prend plusieurs minutes. Les fichiers sont dans :
```
/home/opp_env/default_workspace/
├── omnetpp-6.3.0/
└── inet-4.5.4/
```

### 3. Installer les dependances systeme

```bash
sudo apt-get update
sudo apt-get install -y build-essential ccache clang lld
```

### 4. Cloner simu5G

```bash
cd /c/Users/Utilisateur/Desktop/PFE_BigData/PFE_Bigdata/PFE-NetworkSlicing/PFE-NetworkSlicing/work/simu5g

git clone https://github.com/Unipisa/Simu5G.git
```

### 5. Activer l'environnement OMNeT++

**IMPORTANT : A faire a chaque nouvelle session opp_env !**

```bash
export OPP_ENV_VERSION=1
source /home/opp_env/default_workspace/omnetpp-6.3.0/setenv
```

### 6. Generer features.h

```bash
cd /c/Users/Utilisateur/Desktop/PFE_BigData/PFE_Bigdata/PFE-NetworkSlicing/PFE-NetworkSlicing/work/simu5g/Simu5G

opp_featuretool defines >src/simu5g/common/features.h
```

### 7. Generer le Makefile

```bash
cd src

opp_makemake -f --deep -O out \
    -KINET_PROJ=/home/opp_env/default_workspace/inet-4.5.4 \
    -DINET_IMPORT \
    -I. \
    -I/home/opp_env/default_workspace/inet-4.5.4/src \
    -L/home/opp_env/default_workspace/inet-4.5.4/src \
    -lINET
```

### 8. Corriger le conflit de nom (IMPORTANT)

Le build cree un dossier `Simu5G/` qui entre en conflit avec l'executable. Corriger :

```bash
sed -i 's/TARGET_NAME = Simu5G/TARGET_NAME = simu5g_run/' Makefile
```

### 9. Compiler

```bash
make -j4
```

La compilation prend plusieurs minutes. L'executable sera cree dans `../out/clang-release/src/simu5g_run`.

---

## Convertir les traces de trafic

**Dans un terminal PowerShell Windows** (pas opp_env) :

```powershell
cd C:\Users\Utilisateur\Desktop\PFE_BigData\PFE_Bigdata\PFE-NetworkSlicing\PFE-NetworkSlicing\work\simu5g

# Trouver le nom exact du fichier CSV
dir ..\output\processed_traffic_csv\

# Lancer la conversion
python convert_traces.py "..\output\processed_traffic_csv\part-00000-82fe7fdf-d2ce-406d-a204-27331a85b1b8-c000.csv" traces
```

Cela cree :
```
traces/
├── eMBB/flow_0.txt, flow_1.txt, ...
├── URLLC/flow_0.txt, ...
├── mMTC/flow_0.txt, ...
└── BestEffort/flow_0.txt, ...
```

---

## Configurer et lancer la simulation

### 1. Copier les fichiers de config

Dans opp_env :

```bash
cd /c/Users/Utilisateur/Desktop/PFE_BigData/PFE_Bigdata/PFE-NetworkSlicing/PFE-NetworkSlicing/work/simu5g

cp NetworkSlicing.ned Simu5G/simulations/NR/
cp omnetpp.ini Simu5G/simulations/NR/
cp sliceConfig.xml Simu5G/simulations/NR/
cp -r traces Simu5G/simulations/NR/
```

### 2. Lancer la simulation

```bash
cd Simu5G/simulations/NR

# Mode batch (sans interface graphique, plus rapide)
../../src/simu5g_run -u Cmdenv -f omnetpp.ini -c Normal

# Mode graphique (pour visualiser)
../../src/simu5g_run -u Qtenv -f omnetpp.ini -c Normal
```

Configurations disponibles :
| Config | Description |
|--------|-------------|
| `Normal` | Charge standard |
| `OverloadEmbb` | Surcharge video → violations SLA eMBB |
| `OverloadUrllc` | Surcharge temps-reel → violations latence URLLC |
| `OverloadAll` | Surcharge generale |

---

## Collecter les resultats

Les resultats sont generes dans `results/` au format `.vec` (vecteurs) et `.sca` (scalaires).

### Extraction des metriques pour comparaison M/M/1

**Dans PowerShell Windows** (pas opp_env) :

```powershell
cd C:\Users\Utilisateur\Desktop\PFE_BigData\PFE_Bigdata\PFE-NetworkSlicing\PFE-NetworkSlicing\work\simu5g

# Extraire les metriques simu5G vers format CSV compatible M/M/1
python extract_simu5g_metrics.py simulation_pfe/results ../output/simu5g_sla_dataset.csv
```

Le script `extract_simu5g_metrics.py` :
- Parse les fichiers .vec et .sca
- Agrege par fenetre de 1 seconde et par slice
- Calcule throughput, latence, jitter, packet loss
- Applique les seuils SLA et genere les labels `SLA_Violated_in_15s`
- Produit un CSV au meme format que `slice_sla_dataset`

### Export alternatif avec scavetool

```bash
cd Simu5G/simulations/NR

# Exporter toutes les metriques en CSV
scavetool export -o results.csv results/*.vec results/*.sca

# Filtrer par metrique specifique
scavetool export -f "name=~endToEndDelay:*" -o latency.csv results/*.vec
scavetool export -f "name=~throughput:*" -o throughput.csv results/*.vec
```

### Metriques cles

| Metrique | Description | Unite |
|----------|-------------|-------|
| `endToEndDelay:vector` | Latence bout-en-bout | secondes |
| `throughput:vector` | Debit | bps |
| `packetLoss:vector` | Paquets perdus | count |
| `jitter:vector` | Variation de latence | secondes |

---

## Troubleshooting

### `make: command not found`
```bash
sudo apt-get install -y build-essential
```

### `ccache: command not found`
```bash
sudo apt-get install -y ccache
```

### `clang++: command not found`
```bash
sudo apt-get install -y clang
```

### `invalid linker name in argument '-fuse-ld=lld'`
```bash
sudo apt-get install -y lld
```

### `cannot open output file .../Simu5G: Is a directory`
Le nom de l'executable entre en conflit avec un dossier. Corriger dans le Makefile :
```bash
sed -i 's/TARGET_NAME = Simu5G/TARGET_NAME = simu5g_run/' Makefile
rm -rf ../out
make -j4
```

### `This OMNeT++ installation cannot be used outside an opp_env shell`
Activer l'environnement :
```bash
export OPP_ENV_VERSION=1
source /home/opp_env/default_workspace/omnetpp-6.3.0/setenv
```

### `features.h: file not found`
Generer le fichier :
```bash
cd /path/to/Simu5G
opp_featuretool defines >src/simu5g/common/features.h
```

---

## Architecture simulee

```
UE eMBB [x4]    ──┐
UE URLLC [x4]   ──┼── 5G NR ──> gNodeB ──> UPF ──> Serveurs
UE mMTC [x2]    ──┤                                   (1 par slice)
UE BestEff [x2] ──┘
```

Le gNodeB repartit 50 Resource Blocks (100 MHz) entre les 4 slices.

---

## Fichiers du projet

| Fichier | Description |
|---------|-------------|
| `convert_traces.py` | Convertit les traces Spark → format simu5G |
| `extract_simu5g_metrics.py` | Extrait metriques vers format CSV M/M/1 |
| `NetworkSlicing.ned` | Topologie reseau (UEs, gNodeB, UPF, serveurs) |
| `omnetpp.ini` | Configuration simulation (scenarios, parametres radio) |
| `sliceConfig.xml` | Allocation des Resource Blocks par slice |
| `GUIDE.md` | Ce fichier |

---

## Workflow complet : Comparaison Approche A (M/M/1) vs B (simu5G)

### Etape 1 : Lancer toutes les simulations (dans opp_env)

```bash
cd /c/Users/Utilisateur/Desktop/PFE_BigData/PFE_Bigdata/PFE-NetworkSlicing/PFE-NetworkSlicing/work/simu5g/simulation_pfe

# Copier les fichiers de config mis a jour
cp ../NetworkSlicing.ned .
cp ../omnetpp.ini .
cp ../sliceConfig.xml .

# Lancer les 4 scenarios (10 minutes chacun)
../Simu5G/out/clang-release/src/simu5g_run -u Cmdenv -f omnetpp.ini -c Normal
../Simu5G/out/clang-release/src/simu5g_run -u Cmdenv -f omnetpp.ini -c OverloadEmbb
../Simu5G/out/clang-release/src/simu5g_run -u Cmdenv -f omnetpp.ini -c OverloadUrllc
../Simu5G/out/clang-release/src/simu5g_run -u Cmdenv -f omnetpp.ini -c OverloadAll
```

### Etape 2 : Extraire les metriques (dans PowerShell Windows)

```powershell
cd C:\Users\Utilisateur\Desktop\PFE_BigData\PFE_Bigdata\PFE-NetworkSlicing\PFE-NetworkSlicing\work\simu5g

python extract_simu5g_metrics.py simulation_pfe/results ../output/simu5g_sla_dataset.csv
```

### Etape 3 : Comparer avec le dataset M/M/1

Le dataset genere `simu5g_sla_dataset.csv` a le meme format que `slice_sla_dataset`.
Vous pouvez maintenant :
1. Entrainer le meme modele LSTM sur les deux datasets
2. Comparer les taux de violations SLA par slice
3. Analyser les differences de distribution de latence

### Resultats attendus

| Slice | M/M/1 (Approche A) | simu5G (Approche B) |
|-------|-------------------|---------------------|
| eMBB | ~34% violations | Variable selon charge |
| URLLC | ~0.03% violations | 5-15% avec surcharge |
| mMTC | ~98% violations | Variable |
| Best Effort | ~100% violations | Variable |

L'approche simu5G devrait produire des violations URLLC plus realistes
car elle simule la contention radio, les interferences et le scheduling.
