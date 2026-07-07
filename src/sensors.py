"""Chargement des seuils thermiques des capteurs par OLT.

Format d'export AMS (ex: "temperature threshold dica.csv") :
    Thermal Sensors:DICAME-FX8
    ,Sensor
    ,,
    ,,,Id,Temperature [C],Low TCA...,High TCA...,Low Shutdown...,High Shutdown...
    ,,,R1.S1.LT6.1,68,80,85,90,95
    ...
Chaque OLT possede son propre nombre de capteurs et ses propres seuils.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SENSOR_FEATURES = ["n_sensors", "temp_max", "margin_tca_min", "margin_shutdown_min"]


def parse_threshold_file(path: str | Path) -> tuple[str, pd.DataFrame] | None:
    """Retourne (site, table capteurs) ou None si format inconnu."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or "Thermal Sensors" not in lines[0]:
        return None
    site = lines[0].split(":", 1)[1].strip().rstrip(",")

    rows = []
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        # ligne capteur : ,,,R1.S1.LT6.1,68,80,85,90,95
        if len(parts) >= 9 and parts[3].startswith("R") and "." in parts[3]:
            try:
                rows.append({
                    "sensor_id": parts[3],
                    "temp": float(parts[4]),
                    "low_tca": float(parts[5]),
                    "high_tca": float(parts[6]),
                    "low_shutdown": float(parts[7]),
                    "high_shutdown": float(parts[8]),
                })
            except ValueError:
                continue
    if not rows:
        return None
    return site, pd.DataFrame(rows)


def load_sensor_features(paths: list[str | Path]) -> pd.DataFrame:
    """Features statiques par site : nb capteurs et marges avant seuils.

    - margin_tca_min      : plus petite marge (High TCA - temperature actuelle)
                            -> proche de 0 ou negative = alarme imminente
    - margin_shutdown_min : plus petite marge avant l'arret d'urgence de l'OLT
    """
    records = {}
    for p in paths:
        parsed = parse_threshold_file(p)
        if parsed is None:
            print(f"[sensors] format inconnu, ignore : {p}")
            continue
        site, sensors = parsed
        records[site] = {
            "n_sensors": len(sensors),
            "temp_max": sensors["temp"].max(),
            "margin_tca_min": (sensors["high_tca"] - sensors["temp"]).min(),
            "margin_shutdown_min": (sensors["high_shutdown"] - sensors["temp"]).min(),
        }
        print(f"[sensors] {site}: {len(sensors)} capteurs | temp max "
              f"{records[site]['temp_max']:.0f}C | marge TCA min "
              f"{records[site]['margin_tca_min']:.0f}C | marge shutdown min "
              f"{records[site]['margin_shutdown_min']:.0f}C")

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "site"
    return df.reset_index()
