# Système de Maintenance Prédictive des Équipements FTTH par IA

Prédiction de **l'heure de chute des OLT** (incidents thermiques : surchauffe,
pannes de ventilateurs) sur le réseau Nokia supervisé via le 5520 AMS,
par Gradient Boosting.

## Alarmes thermiques exploitées (TL1)

| Rôle | TL1 | Description |
|---|---|---|
| Précurseur | `FANALM-1` | Panne d'un ventilateur du rack |
| Précurseur | `FANTRAYMISSING` | Tiroir de ventilation absent |
| Précurseur | `TMPHWARN` | Avertissement température haute |
| **Cible** | `HITEMP` | Seuil de température dépassé |
| **Cible** | `TMPH` | Alarme température trop haute |
| **Cible** | `MULTIFANFAILURE` | Pannes multiples de ventilateurs |
| **Cible** | `SHUTDOWN` | Arrêt d'urgence thermique |

**Objectif** : prédire, pour chaque OLT et **heure par heure**, la probabilité
d'une chute (incident thermique grave) dans les 6 prochaines heures.

## Seuils capteurs par OLT

Chaque OLT possède ses propres capteurs thermiques et seuils (fichiers
`temperature threshold *.csv` exportés de l'AMS, ex. DICAME-FX8 : 6 capteurs).
Ils sont intégrés comme features statiques : nombre de capteurs, température
max, marge minimale avant TCA et avant shutdown. Ajoutez les fichiers des
autres OLT dans le dossier du projet pour activer ces features.

## Pipeline

```
CSV AMS (UTF-16) ──► alarm_parser.py ──► features.py ──► train_model.py ──► predict.py ──► dashboard.py
                     normalisation       site × HEURE     Gradient           risque         dashboard
                     dates FR/relatives  fenêtres 1/6/    Boosting +         horaire        Flask ISOC
                                         24/168 h +       GroupKFold         par OLT
                                         seuils capteurs
```

## Utilisation

```powershell
# 1. Activer le venv local
.venv\Scripts\Activate.ps1

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Parser les exports AMS
python src\alarm_parser.py "data8May2026-17-15-47ALARMS_.csv" "data22Jun2026-10-48-17 (1).csv" "data22Jun2026-10-51-8 (2).csv" -o data\alarms_clean.pkl

# 4. Construire le dataset horaire (période dense + seuils capteurs)
python src\features.py --start 2026-04-01 --horizon 6

# 5. Entraîner le modèle
python src\train_model.py

# 6. Calculer les risques horaires par OLT
python src\predict.py

# 7. Backtest de démonstration (heure de chute anticipée)
python src\backtest.py --site SEGBE-FX4 --threshold 0.3

# 8. Lancer le dashboard  ->  http://127.0.0.1:5000
python src\dashboard.py
```

## Features (par OLT × heure)

- Compteurs glissants 1/6/24/168 h : alarmes ventilateurs, précurseurs,
  incidents graves, équipement infra (Rack/Slot/Subrack), Dying Gasp, volume total
- **Surtension d'alarmes** (`surge_1h/6h/24h`) : volume récent rapporté au
  rythme habituel du site — signal clé découvert dans les données (explosion
  du volume d'alarmes dans les heures précédant les chutes)
- Heures écoulées depuis le dernier précurseur / incident grave
- Seuils capteurs propres à l'OLT (marges TCA / shutdown)

## Modèle & validation

- `HistGradientBoostingClassifier` (scikit-learn), pondération des classes
  (68 positifs / 112 633 heures ≈ 0,06 %)
- Validation croisée **StratifiedGroupKFold par site** (aucune fuite train/test)
- Backtest leave-one-site-out (SEGBE-FX4) : montée du risque dès la veille au
  soir de la chute réelle, ~1,5 % de fausses alertes horaires
- Sorties : `models/thermal_model.joblib`, `models/metrics.json`,
  `data/predictions.json`

## Connexion à l'AMS (collecte automatique)

Le module [src/ams_collector.py](src/ams_collector.py) alimente le pipeline en continu.

### Mode 1 — `watch` (recommandé pour démarrer, sans credentials)
Dans l'AMS : *Administration → Scheduled Tasks → Alarm Export* → planifier un
export CSV toutes les 15 min vers un dossier partagé. Puis :

```powershell
python src\ams_collector.py --mode watch --folder "D:\exports_ams" --interval 900
```

Le collecteur détecte chaque nouvel export, l'ingère (déduplication),
reconstruit les features horaires et met à jour data/predictions.json
que le dashboard lit en direct (bouton « Actualiser »).

### Mode 2 — `api` (NBI REST du 5520 AMS)
Demander à l'admin AMS un **compte NBI en lecture seule**, puis :

```powershell
$env:AMS_URL  = "https://<ip-ams>:8443"
$env:AMS_USER = "nbi_readonly"
$env:AMS_PASS = "********"
python src\ams_collector.py --mode api --interval 900
```

Adapter `AMS_ALARM_PATH` dans le script à la version de votre AMS
(REST `/oms1350/data/plat/alarms`, RESTCONF `/restconf/data/fm:alarms`...
voir le guide constructeur « 5520 AMS Northbound Interface »).
Les identifiants restent dans des variables d'environnement, jamais dans le code.

### Démarrage automatique (Windows)
Planificateur de tâches → nouvelle tâche « Au démarrage » :
`<projet>\.venv\Scripts\python.exe src\ams_collector.py --mode watch --folder D:\exports_ams`

## Limites & pistes

- Seulement 50 incidents graves observés (avril–mai 2026) : collecter plus
  d'exports AMS améliorera nettement la précision horaire.
- Un seul fichier de seuils capteurs fourni (DICAME) : exporter ceux des
  autres OLT pour activer les features de marge thermique.
- Idéalement, intégrer la télémétrie température en continu (SNMP/AMS +
  InfluxDB, chapitre 4 du mémoire) pour un vrai calcul de RUL horaire.
