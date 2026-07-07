"""Collecteur AMS : alimente automatiquement le pipeline de prediction.

Deux modes d'integration avec le Nokia 5520 AMS :

MODE 1 - "watch" (operationnel immediatement, sans credentials API)
    L'AMS sait planifier des exports d'alarmes (Administration > Scheduled
    Tasks > Alarm Export) vers un dossier/FTP. Ce script surveille ce dossier,
    ingere chaque nouveau CSV, reconstruit les features et rescored les OLT.

    python src\\ams_collector.py --mode watch --folder "D:\\exports_ams" --interval 900

MODE 2 - "api" (NBI REST du 5520 AMS)
    L'AMS expose une interface Northbound (NBI). Selon la version :
      - REST/JSON  : https://<ams>:8443/oms1350/data/plat/alarms   (OMS)
      - RESTCONF   : https://<ams>:8443/restconf/data/fm:alarms
      - SOAP/XML   : AlarmRetrieval du NBI 3GPP CORBA/SOAP
    Adaptez AMS_ALARM_PATH a votre version (voir le guide "5520 AMS NBI").
    Les identifiants sont lus dans les variables d'environnement
    AMS_URL, AMS_USER, AMS_PASS (jamais en dur dans le code).

    $env:AMS_URL  = "https://10.x.x.x:8443"
    $env:AMS_USER = "nbi_readonly"
    $env:AMS_PASS = "********"        # compte en LECTURE SEULE dediee au NBI
    python src\\ams_collector.py --mode api --interval 900

Dans les deux modes, apres chaque ingestion le script relance :
    features -> predictions -> data/predictions.json (lu par le dashboard).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
STATE_FILE = DATA / "collector_state.json"

# Chemin de l'endpoint alarmes : A ADAPTER a la version de votre AMS
AMS_ALARM_PATH = "/oms1350/data/plat/alarms"


# --------------------------------------------------------------------------- #
#  MODE API : interrogation du NBI REST de l'AMS                              #
# --------------------------------------------------------------------------- #
def fetch_alarms_api() -> pd.DataFrame | None:
    """Recupere les alarmes actives + historiques via le NBI REST de l'AMS."""
    import requests

    url = os.environ.get("AMS_URL")
    user = os.environ.get("AMS_USER")
    pwd = os.environ.get("AMS_PASS")
    if not all([url, user, pwd]):
        print("[api] definissez AMS_URL / AMS_USER / AMS_PASS (compte NBI lecture seule)")
        return None

    session = requests.Session()
    session.verify = True  # garder la verification TLS ; fournir le CA interne si besoin

    # 1) authentification (token de session) -------------------------------
    #    Selon la version AMS : POST /oms1350/data/plat/session ou Basic Auth.
    auth = session.post(
        f"{url}/oms1350/data/plat/session",
        json={"login": user, "password": pwd},
        timeout=30,
    )
    auth.raise_for_status()
    token = auth.json().get("token", "")
    session.headers["Authorization"] = f"Bearer {token}"

    # 2) recuperation paginee des alarmes -----------------------------------
    rows, offset, page = [], 0, 1000
    while True:
        r = session.get(
            f"{url}{AMS_ALARM_PATH}",
            params={"offset": offset, "limit": page},
            timeout=60,
        )
        r.raise_for_status()
        batch = r.json().get("items", r.json() if isinstance(r.json(), list) else [])
        if not batch:
            break
        rows.extend(batch)
        offset += page
        if len(batch) < page:
            break

    if not rows:
        return None

    # 3) mapping JSON NBI -> colonnes de l'export CSV AMS -------------------
    #    (adapter les cles au schema de votre version)
    df = pd.DataFrame([{
        "Severity": a.get("severity") or a.get("perceivedSeverity"),
        "Event Time": a.get("eventTime") or a.get("raisedTime"),
        "Cleared Time": a.get("clearedTime"),
        "Source Name": a.get("sourceName") or a.get("objectName"),
        "Mnemonic": a.get("mnemonic"),
        "TL1 Alarm Condition": a.get("tl1Condition") or a.get("alarmCondition"),
        "Probable Cause": a.get("probableCause"),
        "Specific Problem": a.get("specificProblem"),
        "Service Affecting": a.get("serviceAffecting"),
    } for a in rows])
    print(f"[api] {len(df):,} alarmes recuperees depuis {url}")
    return df


# --------------------------------------------------------------------------- #
#  MODE WATCH : surveillance d'un dossier d'exports planifies                 #
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"seen": []}


def save_state(state: dict) -> None:
    DATA.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def watch_new_files(folder: Path, state: dict) -> list[Path]:
    files = sorted(folder.glob("*.csv"))
    return [f for f in files if f.name not in state["seen"]]


# --------------------------------------------------------------------------- #
#  Ingestion + re-prediction                                                  #
# --------------------------------------------------------------------------- #
def ingest(frames: list[pd.DataFrame], reference: datetime) -> None:
    """Normalise les nouvelles alarmes et les fusionne dans alarms_clean.pkl."""
    from alarm_parser import USECOLS, load_alarms, parse_times

    # ecrire un CSV temporaire au meme format que les exports puis reutiliser
    # le parseur officiel garantit une normalisation identique
    tmp = DATA / f"ams_pull_{reference:%Y%m%d-%H%M%S}.csv"
    pd.concat(frames)[USECOLS[:9]].assign(**{"History Time": ""}).to_csv(
        tmp, index=False, encoding="utf-16"
    )
    new = load_alarms(tmp, reference=reference)
    tmp.unlink()

    clean_path = DATA / "alarms_clean.pkl"
    if clean_path.exists():
        old = pd.read_pickle(clean_path)
        full = pd.concat([old, new], ignore_index=True)
    else:
        full = new
    full = full.drop_duplicates(subset=["ts", "site", "entity", "tl1", "Specific Problem"])
    full = full.sort_values("ts")
    full.to_pickle(clean_path)
    print(f"[ingest] +{len(new):,} alarmes -> total {len(full):,}")


def repredict(horizon: int, start: str) -> None:
    """Reconstruit features + predictions (meme logique que la CLI)."""
    import glob

    from features import build_features, build_hourly
    from predict import build_heatmap, build_history, score_sites
    from sensors import load_sensor_features

    alarms = pd.read_pickle(DATA / "alarms_clean.pkl")
    alarms = alarms[alarms["ts"] >= pd.Timestamp(start)]
    sensor_files = glob.glob(str(BASE / "temperature threshold*.csv"))
    sensors = load_sensor_features(sensor_files) if sensor_files else None

    ds = build_features(build_hourly(alarms), horizon=horizon, sensors=sensors)
    ds.to_pickle(DATA / "dataset.pkl")

    model = BASE / "models" / "thermal_model.joblib"
    scores, hz = score_sites(DATA / "dataset.pkl", model)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "horizon_h": hz,
        "sites": json.loads(scores.to_json(orient="records", date_format="iso")),
        "history": build_history(DATA / "dataset.pkl", model),
        "heatmap": build_heatmap(DATA / "alarms_clean.pkl"),
    }
    (DATA / "predictions.json").write_text(json.dumps(payload))

    top = scores.head(3)[["site", "risk_proba", "risk_level"]]
    print(f"[predict] {len(scores)} OLT rescores. Top risques :")
    for _, r in top.iterrows():
        print(f"          {r['site']:<20} {r['risk_proba']:6.1%}  {r['risk_level']}")


def run_cycle(args) -> None:
    now = datetime.now()
    if args.mode == "api":
        df = fetch_alarms_api()
        if df is None or df.empty:
            return
        ingest([df], reference=now)
    else:  # watch
        folder = Path(args.folder)
        state = load_state()
        new_files = watch_new_files(folder, state)
        if not new_files:
            print(f"[watch] {now:%H:%M:%S} aucun nouvel export dans {folder}")
            return
        from alarm_parser import load_alarms
        for f in new_files:
            print(f"[watch] nouvel export : {f.name}")
            alarms = load_alarms(f)
            clean = DATA / "alarms_clean.pkl"
            full = pd.concat([pd.read_pickle(clean), alarms]) if clean.exists() else alarms
            full = full.drop_duplicates(
                subset=["ts", "site", "entity", "tl1", "Specific Problem"]
            ).sort_values("ts")
            full.to_pickle(clean)
            state["seen"].append(f.name)
        save_state(state)

    repredict(args.horizon, args.start)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collecteur AMS -> pipeline de prediction")
    ap.add_argument("--mode", choices=["watch", "api"], default="watch")
    ap.add_argument("--folder", default=str(DATA / "incoming"),
                    help="Dossier surveille (mode watch) ou l'AMS depose ses exports")
    ap.add_argument("--interval", type=int, default=900,
                    help="Periode de collecte en secondes (defaut 900 = 15 min)")
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--once", action="store_true", help="Un seul cycle puis sortie")
    args = ap.parse_args()

    if args.mode == "watch":
        Path(args.folder).mkdir(parents=True, exist_ok=True)
        print(f"[collector] mode watch : {args.folder} (toutes les {args.interval}s)")
    else:
        print(f"[collector] mode api : {os.environ.get('AMS_URL', 'AMS_URL non defini')}")

    while True:
        try:
            run_cycle(args)
        except Exception as e:  # continuer la collecte malgre une erreur ponctuelle
            print(f"[erreur] {type(e).__name__}: {e}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
