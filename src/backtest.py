"""Backtest horaire : preuve que le systeme anticipe l'HEURE de chute d'un OLT.

Principe (leave-one-site-out, honnete) :
  1. on retire completement un OLT du jeu d'entrainement,
  2. on entraine le modele sur tous les autres OLT,
  3. on rejoue la chronologie horaire de l'OLT retire,
  4. on verifie que le risque monte AVANT l'heure reelle de l'incident.

Usage :
  python src\\backtest.py --site SEGBE-FX4
"""
from __future__ import annotations

import pandas as pd

from features import FEATURE_COLUMNS
from train_model import make_model, sample_weights, usable_features


def backtest_site(ds: pd.DataFrame, site: str, alert_threshold: float = 0.4) -> None:
    horizon = int(ds.attrs.get("horizon_h", 6))
    site_df = ds[ds["site"] == site].sort_values("hour_ts")
    if site_df.empty:
        print(f"[!] Site inconnu : {site}")
        print("    Sites avec incidents graves :",
              ", ".join(sorted(ds[ds["target"] == 1]["site"].unique())))
        return

    train = ds[ds["site"] != site]
    feats = usable_features(train[FEATURE_COLUMNS])
    model = make_model()
    model.fit(train[feats], train["target"],
              sample_weight=sample_weights(train["target"]))

    site_df = site_df.assign(
        risque=model.predict_proba(site_df[feats])[:, 1]
    )
    incidents = site_df[site_df["target"] == 1]

    print("=" * 74)
    print(f"BACKTEST HORAIRE LEAVE-ONE-SITE-OUT : {site} (horizon {horizon}h)")
    print("(le modele n'a JAMAIS vu cet OLT pendant l'entrainement)")
    print("=" * 74)

    if incidents.empty:
        print("Aucun incident grave sur cet OLT dans la periode etudiee.")
    else:
        # heure reelle de la premiere chute = premiere heure avec incident
        # dans la fenetre => l'incident survient a first_flag + horizon max
        first_flag = incidents["hour_ts"].min()
        window = site_df[
            (site_df["hour_ts"] >= first_flag - pd.Timedelta(hours=24))
            & (site_df["hour_ts"] <= first_flag + pd.Timedelta(hours=horizon + 2))
        ]
        print(f"\nPremiere heure signalant la chute (fenetre {horizon}h) : {first_flag}")
        print(f"\n{'Heure':<18} {'Risque':<9} {'':<32} {'Alerte':<11} {'Chute reelle <'}{horizon}h")
        print("-" * 78)
        for _, r in window.iterrows():
            alert = ">>> ALERTE" if r["risque"] >= alert_threshold else ""
            real = "OUI" if r["target"] == 1 else ""
            bar = "#" * int(r["risque"] * 30)
            print(f"{r['hour_ts']:%d/%m %H:%M}     {r['risque']:6.1%}  {bar:<32} {alert:<11} {real}")

        alerts = site_df[site_df["risque"] >= alert_threshold]
        if not alerts.empty and alerts["hour_ts"].min() <= first_flag:
            lead = (first_flag - alerts["hour_ts"].min()).total_seconds() / 3600
            print(f"\n[RESULTAT] Premiere alerte a {alerts['hour_ts'].min():%d/%m %H:%M}, "
                  f"soit {lead:.0f} heure(s) AVANT la fenetre de chute.")
        else:
            print("\n[RESULTAT] Pas d'alerte anticipee au seuil choisi.")

    calm = site_df[site_df["target"] == 0]
    fa_n = int((calm["risque"] >= alert_threshold).sum())
    fa = fa_n / len(calm) if len(calm) else 0.0
    print(f"Fausses alertes sur heures calmes : {fa:.2%} ({fa_n}/{len(calm)} heures)")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Backtest horaire par OLT")
    ap.add_argument("--site", default="SEGBE-FX4")
    ap.add_argument("-i", "--input", default="data/dataset.pkl")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="Seuil d'alerte (defaut : 0.4)")
    args = ap.parse_args()

    ds = pd.read_pickle(args.input)
    backtest_site(ds, args.site, args.threshold)


if __name__ == "__main__":
    main()
