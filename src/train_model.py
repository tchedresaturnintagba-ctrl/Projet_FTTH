"""Entrainement du modele Gradient Boosting de prediction d'incidents thermiques.

Evaluation par validation croisee groupee par site (les jours d'un meme site ne
sont jamais repartis entre train et test -> pas de fuite de donnees), puis
entrainement final sur l'ensemble du dataset.
Sortie : models/thermal_model.joblib + models/metrics.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from features import FEATURE_COLUMNS


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=300,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42,
    )


def sample_weights(y: pd.Series) -> np.ndarray:
    pos = max(int(y.sum()), 1)
    w = float((len(y) - pos) / pos)
    return np.where(y == 1, w, 1.0)


def usable_features(X: pd.DataFrame) -> list[str]:
    """Ecarte les colonnes constantes (ex: seuils capteurs disponibles pour
    un seul OLT) qui font planter le binning de sklearn et n'apportent rien."""
    kept = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    dropped = sorted(set(X.columns) - set(kept))
    if dropped:
        print(f"[train] colonnes constantes ignorees : {', '.join(dropped)}")
    return kept


def best_threshold(y_true, proba) -> float:
    """Seuil qui maximise le F1 sur la courbe precision/rappel."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def train(ds: pd.DataFrame, out_dir: Path, n_splits: int = 5) -> dict:
    feats = usable_features(ds[FEATURE_COLUMNS])
    X = ds[feats]
    y = ds["target"]
    groups = ds["site"]

    # --- validation croisee out-of-fold, groupee par site -------------------
    n_splits = min(n_splits, int(y.sum())) or 2
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(ds))
    for tr_idx, te_idx in cv.split(X, y, groups):
        m = make_model()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              sample_weight=sample_weights(y.iloc[tr_idx]))
        oof[te_idx] = m.predict_proba(X.iloc[te_idx])[:, 1]

    thr = best_threshold(y, oof) if y.sum() else 0.5
    y_pred = (oof >= thr).astype(int)

    metrics = {
        "validation": f"StratifiedGroupKFold({n_splits}) par site (out-of-fold)",
        "rows": int(len(ds)),
        "sites": int(groups.nunique()),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, oof)) if y.nunique() > 1 else None,
        "average_precision": float(average_precision_score(y, oof))
        if y.nunique() > 1 else None,
        "threshold": thr,
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "report": classification_report(y, y_pred, output_dict=True, zero_division=0),
    }

    # --- modele final entraine sur tout le dataset --------------------------
    final = make_model()
    final.fit(X, y, sample_weight=sample_weights(y))

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": feats, "threshold": thr},
                out_dir / "thermal_model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Entraine le modele thermique")
    ap.add_argument("-i", "--input", default="data/dataset.pkl")
    ap.add_argument("-o", "--out", default="models")
    args = ap.parse_args()

    ds = pd.read_pickle(args.input)
    print(f"[train] {len(ds):,} lignes | positifs: {int(ds['target'].sum())}")
    m = train(ds, Path(args.out))

    print(f"[ok] modele -> {args.out}/thermal_model.joblib")
    print(f"     validation       : {m['validation']}")
    print(f"     ROC-AUC          : {m['roc_auc']}")
    print(f"     Avg Precision    : {m['average_precision']}")
    print(f"     Seuil optimal    : {m['threshold']:.4f}")
    print(f"     Matrice confusion: {m['confusion_matrix']}")
    rep = m["report"].get("1", {})
    print(f"     Classe 1 -> precision {rep.get('precision', 0):.2f} | "
          f"rappel {rep.get('recall', 0):.2f} | f1 {rep.get('f1-score', 0):.2f}")


if __name__ == "__main__":
    main()
