"""Inference horaire : risque de chute de chaque OLT dans les prochaines heures.

Produit :
  - le classement des OLT par probabilite de chute (fenetre de N heures),
  - l'historique horaire du risque par site (tendance predictive),
  - la heatmap du chassis (alarmes thermiques par slot LT).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd


def risk_level(p: float, threshold: float) -> str:
    if p >= 0.7:
        return "CRITIQUE"
    if p >= 0.4:
        return "ELEVE"
    if p >= max(threshold, 0.1):
        return "SURVEILLANCE"
    return "NORMAL"


def score_sites(dataset_path: str | Path, model_path: str | Path) -> tuple[pd.DataFrame, int]:
    bundle = joblib.load(model_path)
    model, feats, thr = bundle["model"], bundle["features"], bundle["threshold"]

    ds = pd.read_pickle(dataset_path)
    horizon = int(ds.attrs.get("horizon_h", 6))
    latest = ds.sort_values("hour_ts").groupby("site").tail(1).copy()
    latest["risk_proba"] = model.predict_proba(latest[feats])[:, 1]
    latest["risk_level"] = latest["risk_proba"].apply(lambda p: risk_level(p, thr))

    # heure estimee de chute = derniere heure observee + horizon (borne haute)
    latest["fall_window_end"] = latest["hour_ts"] + pd.Timedelta(hours=horizon)

    cols = [
        "site", "hour_ts", "fall_window_end", "risk_proba", "risk_level",
        "fan_alarms_24h", "precursor_24h", "severe_168h",
        "hours_since_precursor", "hours_since_severe", "total_alarms_24h",
        "temp_max", "margin_tca_min", "margin_shutdown_min",
    ]
    out = latest[cols].sort_values("risk_proba", ascending=False).reset_index(drop=True)
    return out, horizon


def build_history(dataset_path: str | Path, model_path: str | Path,
                  days: int = 14) -> dict:
    """Tendance predictive horaire par site sur les N derniers jours."""
    bundle = joblib.load(model_path)
    model, feats = bundle["model"], bundle["features"]

    ds = pd.read_pickle(dataset_path)
    # fenetre relative a la derniere heure observee DE CHAQUE site
    site_max = ds.groupby("site")["hour_ts"].transform("max")
    ds = ds[ds["hour_ts"] >= site_max - pd.Timedelta(days=days)].copy()
    ds["proba"] = model.predict_proba(ds[feats])[:, 1]

    history: dict[str, list] = {}
    for site, g in ds.sort_values("hour_ts").groupby("site"):
        history[site] = [
            {"d": h.strftime("%d/%m %Hh"), "p": round(float(p), 4)}
            for h, p in zip(g["hour_ts"], g["proba"])
        ]
    return history


def build_alarm_types(alarms_path: str | Path) -> dict:
    """Types d'alarmes thermiques par site : occurrences et derniere date."""
    al = pd.read_pickle(alarms_path)
    th = al[al["is_thermal"]]
    out: dict[str, list] = {}
    for (site, tl1), g in th.groupby(["site", "tl1"]):
        out.setdefault(site, []).append({
            "tl1": tl1,
            "count": int(len(g)),
            "last": g["ts"].max().strftime("%d/%m/%Y %H:%M"),
            "severe": bool(g["is_severe"].iloc[0]),
            "problem": str(g["Specific Problem"].iloc[-1]),
        })
    for site in out:
        out[site].sort(key=lambda x: (x["severe"], x["count"]), reverse=True)
    return out


def build_heatmap(alarms_path: str | Path) -> dict:
    """Heatmap chassis : alarmes thermiques par slot LT + alarmes fans rack."""
    al = pd.read_pickle(alarms_path)
    lt = al["entity"].str.extract(r"LT(\d+)", expand=False)
    al["lt"] = pd.to_numeric(lt, errors="coerce")

    n_slots = al.groupby("site")["lt"].max().dropna().astype(int)
    slot_thermal = (
        al[al["is_thermal"] & al["lt"].notna()]
        .groupby(["site", "lt"]).size()
    )
    fan_alarms = (
        al[al["tl1"].isin({"FANALM-1", "MULTIFANFAILURE", "FANTRAYMISSING"})]
        .groupby("site").size()
    )

    heatmap: dict[str, dict] = {}
    for site in n_slots.index:
        counts = {}
        if site in slot_thermal.index.get_level_values(0):
            counts = {str(int(k)): int(v) for k, v in slot_thermal.loc[site].items()}
        heatmap[site] = {
            "slots": int(n_slots[site]),
            "slot_alarms": counts,
            "fan_alarms": int(fan_alarms.get(site, 0)),
        }
    return heatmap


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Risque horaire de chute par OLT")
    ap.add_argument("-d", "--dataset", default="data/dataset.pkl")
    ap.add_argument("-m", "--model", default="models/thermal_model.joblib")
    ap.add_argument("-a", "--alarms", default="data/alarms_clean.pkl")
    ap.add_argument("-o", "--out", default="data/predictions.json")
    ap.add_argument("--history-days", type=int, default=14)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    scores, horizon = score_sites(args.dataset, args.model)
    history = build_history(args.dataset, args.model, days=args.history_days)
    heatmap = build_heatmap(args.alarms)
    alarm_types = build_alarm_types(args.alarms)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "horizon_h": horizon,
        "sites": json.loads(scores.to_json(orient="records", date_format="iso")),
        "history": history,
        "heatmap": heatmap,
        "alarm_types": alarm_types,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload))

    print(f"[ok] {len(scores)} OLT scores (horizon {horizon}h) -> {args.out}\n")
    print(f"=== TOP {args.top} OLT A RISQUE DE CHUTE ===")
    show = scores.head(args.top)[[
        "site", "hour_ts", "risk_proba", "risk_level",
        "fan_alarms_24h", "precursor_24h", "severe_168h", "temp_max", "margin_tca_min",
    ]]
    with pd.option_context("display.width", 170):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
