"""Feature engineering HORAIRE : predit a quelle heure un OLT risque de tomber.

Grain : (site OLT, heure). Cible : au moins un incident thermique grave
(HITEMP / SHUTDOWN / MULTIFANFAILURE / TMPH) sur le site dans les
`horizon` prochaines HEURES.

Integre les seuils des capteurs thermiques propres a chaque OLT
(fichiers "temperature threshold *.csv") comme features statiques.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sensors import SENSOR_FEATURES, load_sensor_features

WINDOWS_H = (1, 6, 24, 168)  # fenetres glissantes en heures (1h, 6h, 1j, 7j)
_BASE = ("fan_alarms", "precursor", "severe", "infra_eqp", "dg", "total_alarms")

FEATURE_COLUMNS: list[str] = (
    [f"{b}_{w}h" for b in _BASE for w in WINDOWS_H]
    + [
        "hours_since_precursor",
        "hours_since_severe",
        "precursor_ratio_24h",
        "cum_precursors",
        "cum_severe",
        "hour",
        "dayofweek",
        # surtension d'alarmes : volume recent rapporte au rythme habituel du
        # site (invariant d'echelle -> comparable d'un OLT a l'autre)
        "surge_1h",
        "surge_6h",
        "surge_24h",
        "dg_ratio_24h",
    ]
    + SENSOR_FEATURES
)


def build_hourly(alarms: pd.DataFrame) -> pd.DataFrame:
    """Agrege les alarmes par (site, heure) avec grille horaire continue."""
    df = alarms.copy()
    df["hour_ts"] = df["ts"].dt.floor("h")
    df["is_fan"] = df["tl1"].isin({"FANALM-1", "MULTIFANFAILURE", "FANTRAYMISSING"})

    grouped = df.groupby(["site", "hour_ts"]).agg(
        total_alarms=("ts", "size"),
        fan_alarms=("is_fan", "sum"),
        precursors=("is_precursor", "sum"),
        severe=("is_severe", "sum"),
        infra_eqp=("is_infra_eqp", "sum"),
        dg=("is_dg", "sum"),
    ).reset_index()

    out = []
    for site, g in grouped.groupby("site"):
        g = g.set_index("hour_ts").sort_index()
        idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.drop(columns="site").reindex(idx, fill_value=0)
        g.index.name = "hour_ts"
        g["site"] = site
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def build_features(hourly: pd.DataFrame, horizon: int = 6,
                   sensors: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features glissantes horaires + cible : chute de l'OLT sous `horizon` heures."""
    frames = []
    for site, g in hourly.groupby("site"):
        g = g.sort_values("hour_ts").set_index("hour_ts")
        f = pd.DataFrame(index=g.index)

        for w in WINDOWS_H:
            r = g.rolling(f"{w}h")
            f[f"fan_alarms_{w}h"] = r["fan_alarms"].sum()
            f[f"precursor_{w}h"] = r["precursors"].sum()
            f[f"severe_{w}h"] = r["severe"].sum()
            f[f"infra_eqp_{w}h"] = r["infra_eqp"].sum()
            f[f"dg_{w}h"] = r["dg"].sum()
            f[f"total_alarms_{w}h"] = r["total_alarms"].sum()

        hour_num = pd.Series(np.arange(len(g)), index=g.index, dtype=float)
        last_prec = hour_num.where(g["precursors"] > 0).ffill()
        last_sev = hour_num.where(g["severe"] > 0).ffill()
        f["hours_since_precursor"] = (hour_num - last_prec).fillna(99999.0)
        f["hours_since_severe"] = (hour_num - last_sev).fillna(99999.0)

        f["precursor_ratio_24h"] = (
            f["precursor_24h"] / f["total_alarms_24h"].replace(0, np.nan)
        ).fillna(0.0)
        f["cum_precursors"] = g["precursors"].cumsum()
        f["cum_severe"] = g["severe"].cumsum()
        f["hour"] = f.index.hour
        f["dayofweek"] = f.index.dayofweek

        # surtension : volume recent vs rythme habituel du site
        eps = 1.0
        f["surge_1h"] = f["total_alarms_1h"] / (f["total_alarms_24h"] / 24 + eps)
        f["surge_6h"] = f["total_alarms_6h"] / (f["total_alarms_168h"] / 28 + eps)
        f["surge_24h"] = f["total_alarms_24h"] / (f["total_alarms_168h"] / 7 + eps)
        f["dg_ratio_24h"] = (
            f["dg_24h"] / f["total_alarms_24h"].replace(0, np.nan)
        ).fillna(0.0)

        # cible : incident grave sur l'OLT dans les `horizon` prochaines heures
        future_sev = (
            g["severe"][::-1].rolling(f"{horizon}h").sum()[::-1] - g["severe"]
        )
        f["target"] = (future_sev > 0).astype(int)

        f["site"] = site
        frames.append(f.reset_index())

    ds = pd.concat(frames, ignore_index=True)

    # seuils capteurs propres a chaque OLT (NaN si fichier non fourni ;
    # HistGradientBoosting gere les NaN nativement)
    if sensors is not None and not sensors.empty:
        ds = ds.merge(sensors, on="site", how="left")
    else:
        for c in SENSOR_FEATURES:
            ds[c] = np.nan

    # retire les dernieres heures de chaque site (cible non observable)
    cutoff = ds.groupby("site")["hour_ts"].transform("max") - pd.Timedelta(hours=horizon)
    ds = ds[ds["hour_ts"] <= cutoff].reset_index(drop=True)
    ds.attrs["horizon_h"] = horizon
    return ds


def main() -> None:
    import argparse, glob

    ap = argparse.ArgumentParser(description="Construit le dataset ML horaire")
    ap.add_argument("-i", "--input", default="data/alarms_clean.pkl")
    ap.add_argument("-o", "--out", default="data/dataset.pkl")
    ap.add_argument("--horizon", type=int, default=6,
                    help="Horizon de prediction en HEURES (defaut: 6)")
    ap.add_argument("--start", default="2026-04-01",
                    help="Ignore les alarmes avant cette date (periode dense)")
    ap.add_argument("--sensors", default="temperature threshold*.csv",
                    help="Glob des fichiers de seuils capteurs par OLT")
    args = ap.parse_args()

    alarms = pd.read_pickle(args.input)
    if args.start:
        alarms = alarms[alarms["ts"] >= pd.Timestamp(args.start)]
    print(f"[features] {len(alarms):,} alarmes, {alarms['site'].nunique()} sites "
          f"| fichiers: {sorted(alarms['file_source'].unique())}")

    sensor_files = glob.glob(args.sensors)
    sensors = load_sensor_features(sensor_files) if sensor_files else None

    hourly = build_hourly(alarms)
    ds = build_features(hourly, horizon=args.horizon, sensors=sensors)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ds.to_pickle(args.out)
    pos = int(ds["target"].sum())
    print(f"[ok] {len(ds):,} lignes (site-heure) -> {args.out}")
    print(f"     positifs (chute OLT sous {args.horizon}h): {pos} "
          f"({100 * pos / len(ds):.4f} %)")


if __name__ == "__main__":
    main()
