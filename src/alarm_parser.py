"""Chargement et normalisation des exports d'alarmes Nokia 5520 AMS (CSV UTF-16).

Gere les formats de dates mixtes :
  - absolu francais : "6 mai 2026 11:15:43 UTC"
  - relatif : "Today at 06:45:33 UTC", "Yesterday at ...", "N minutes ago"
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- constantes
USECOLS = [
    "Severity", "Event Time", "Cleared Time", "Source Name",
    "Mnemonic", "TL1 Alarm Condition", "Probable Cause",
    "Specific Problem", "Service Affecting",
]

# Alarmes thermiques : precurseurs (signaux faibles) et incidents graves (cible)
PRECURSOR_TL1 = {"FANALM-1", "FANTRAYMISSING", "TMPHWARN"}
SEVERE_TL1 = {"HITEMP", "SHUTDOWN", "MULTIFANFAILURE", "TMPH"}
THERMAL_TL1 = PRECURSOR_TL1 | SEVERE_TL1

MONTHS = {
    "janv": 1, "jan": 1, "fevr": 2, "feb": 2, "fev": 2, "mars": 3, "mar": 3,
    "avr": 4, "apr": 4, "mai": 5, "may": 5, "juin": 6, "jun": 6,
    "juil": 7, "jul": 7, "aout": 8, "aug": 8, "sept": 9, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_ABS_RE = r"(\d{1,2}) ([a-z]+)\.? (\d{4}) (\d{1,2}):(\d{2}):(\d{2})"
_REL_RE = r"(today|yesterday) at (\d{1,2}):(\d{2}):(\d{2})"
_AGO_RE = r"(\d+)\s+(second|minute|hour|day)s? ago"
_FNAME_RE = re.compile(r"data(\d{1,2})([A-Za-z]+)(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})")


def infer_reference_from_filename(path: str | Path) -> datetime | None:
    """Deduit la date d'export depuis un nom type dataDDMonYYYY-HH-MM-SS."""
    m = _FNAME_RE.search(Path(path).name)
    if not m:
        return None
    day, mon, year, hh, mm, ss = m.groups()
    month = MONTHS.get(mon[:4].lower()) or MONTHS.get(mon[:3].lower())
    if not month:
        return None
    return datetime(int(year), month, int(day), int(hh), int(mm), int(ss))


def _normalize_text(s: pd.Series) -> pd.Series:
    return (
        s.fillna("")
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.strip()
    )


def parse_times(series: pd.Series, reference: datetime | None = None) -> pd.Series:
    """Convertit la colonne de temps AMS en datetime (vectorise)."""
    s = _normalize_text(series)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # --- dates absolues -----------------------------------------------------
    abs_parts = s.str.extract(_ABS_RE)
    mask_abs = abs_parts[0].notna()
    if mask_abs.any():
        p = abs_parts[mask_abs]
        month = (
            p[1].str.slice(0, 4).map(MONTHS)
            .fillna(p[1].str.slice(0, 3).map(MONTHS))
        )
        ok = month.notna()
        comp = pd.DataFrame({
            "year": p[2][ok].astype(int),
            "month": month[ok].astype(int),
            "day": p[0][ok].astype(int),
            "hour": p[3][ok].astype(int),
            "minute": p[4][ok].astype(int),
            "second": p[5][ok].astype(int),
        })
        out.loc[comp.index] = pd.to_datetime(comp, errors="coerce")

    # --- reference pour les dates relatives ---------------------------------
    if reference is None:
        valid = out.dropna()
        reference = valid.max().to_pydatetime() if not valid.empty else datetime.now()
    ref_day = reference.replace(hour=0, minute=0, second=0, microsecond=0)

    # --- today / yesterday ---------------------------------------------------
    rel_parts = s.str.extract(_REL_RE)
    mask_rel = rel_parts[0].notna() & out.isna()
    if mask_rel.any():
        p = rel_parts[mask_rel]
        base = p[0].map({"today": ref_day, "yesterday": ref_day - timedelta(days=1)})
        delta = pd.to_timedelta(
            p[1].astype(int) * 3600 + p[2].astype(int) * 60 + p[3].astype(int),
            unit="s",
        )
        out.loc[p.index] = pd.to_datetime(base) + delta

    # --- "N minutes ago" ------------------------------------------------------
    ago_parts = s.str.extract(_AGO_RE)
    mask_ago = ago_parts[0].notna() & out.isna()
    if mask_ago.any():
        p = ago_parts[mask_ago]
        secs = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}
        delta = pd.to_timedelta(p[0].astype(int) * p[1].map(secs), unit="s")
        out.loc[p.index] = pd.Timestamp(reference) - delta

    return out


def load_alarms(path: str | Path, reference: datetime | None = None) -> pd.DataFrame:
    """Charge un export AMS et retourne un DataFrame normalise."""
    path = Path(path)
    if reference is None:
        reference = infer_reference_from_filename(path)

    df = pd.read_csv(path, encoding="utf-16", usecols=USECOLS, dtype=str)
    df["ts"] = parse_times(df["Event Time"], reference)
    df = df.dropna(subset=["ts"]).copy()

    # Source Name -> type / site / entite  (ex: "Rack:KLIKAME-FX16:R1")
    parts = df["Source Name"].str.split(":", n=2, expand=True)
    df["source_type"] = parts[0].str.strip()
    df["site"] = parts[1].str.strip() if 1 in parts.columns else ""
    df["entity"] = parts[2].str.strip() if 2 in parts.columns else ""
    df = df[df["site"].notna() & (df["site"] != "")]

    tl1 = df["TL1 Alarm Condition"].fillna("").str.strip().str.upper()
    df["tl1"] = tl1
    df["is_precursor"] = tl1.isin(PRECURSOR_TL1)
    df["is_severe"] = tl1.isin(SEVERE_TL1)
    df["is_thermal"] = tl1.isin(THERMAL_TL1)
    df["is_dg"] = tl1.eq("DG")  # Dying Gasp (instabilite alimentation)
    df["is_infra_eqp"] = (
        df["source_type"].isin(["Rack", "Slot", "Subrack", "NE System", "SFP Inventory"])
        & df["Probable Cause"].fillna("").str.contains("Equipment", case=False)
    )
    df["severity"] = df["Severity"].fillna("").str.strip()
    df["file_source"] = path.name
    return df[[
        "ts", "site", "source_type", "entity", "severity", "tl1",
        "Probable Cause", "Specific Problem", "Service Affecting",
        "is_precursor", "is_severe", "is_thermal", "is_dg", "is_infra_eqp",
        "file_source",
    ]]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Parse les exports d'alarmes AMS")
    ap.add_argument("files", nargs="+", help="Fichiers CSV AMS (UTF-16)")
    ap.add_argument("-o", "--out", default="data/alarms_clean.pkl")
    args = ap.parse_args()

    frames = []
    for f in args.files:
        print(f"[parse] {f} ...")
        d = load_alarms(f)
        print(f"        {len(d):,} alarmes | {d['ts'].min()} -> {d['ts'].max()} "
              f"| thermiques: {int(d['is_thermal'].sum())}")
        frames.append(d)

    full = pd.concat(frames, ignore_index=True).sort_values("ts")
    full = full.drop_duplicates(subset=["ts", "site", "entity", "tl1", "Specific Problem"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    full.to_pickle(out)
    print(f"[ok] {len(full):,} alarmes -> {out}")
    print(f"     sites: {full['site'].nunique()} | thermiques: {int(full['is_thermal'].sum())} "
          f"(graves: {int(full['is_severe'].sum())}, precurseurs: {int(full['is_precursor'].sum())})")


if __name__ == "__main__":
    main()
