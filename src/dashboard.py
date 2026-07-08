"""Dashboard ISOC - Supervision predictive thermique des OLT FTTH (Flask)."""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from auth import ensure_default_admin, get_secret_key, login_required, verify

BASE = Path(__file__).resolve().parent.parent
PREDICTIONS = BASE / "data" / "predictions.json"
METRICS = BASE / "models" / "metrics.json"

app = Flask(__name__)
app.secret_key = get_secret_key()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=8 * 3600,  # 8 h de session
)

_temp_pwd = ensure_default_admin()
if _temp_pwd:
    print("=" * 62)
    print("  PREMIER LANCEMENT : compte 'admin' cree.")
    print(f"  Mot de passe temporaire : {_temp_pwd}")
    print("  Changez-le :  python src\\auth.py add admin")
    print("=" * 62)


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    username = ""
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if verify(username, password):
            session.clear()
            session["user"] = username
            session.permanent = True
            target = request.args.get("next") or url_for("index")
            if not target.startswith("/"):  # anti open-redirect
                target = url_for("index")
            return redirect(target)
        error = "Identifiant ou mot de passe incorrect."
    return render_template("login.html", error=error, username=username)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    data = load_json(PREDICTIONS)
    metrics = load_json(METRICS)
    sites = data.get("sites", [])
    counts = {
        "CRITIQUE": sum(1 for s in sites if s["risk_level"] == "CRITIQUE"),
        "ELEVE": sum(1 for s in sites if s["risk_level"] == "ELEVE"),
        "SURVEILLANCE": sum(1 for s in sites if s["risk_level"] == "SURVEILLANCE"),
        "NORMAL": sum(1 for s in sites if s["risk_level"] == "NORMAL"),
    }
    return render_template(
        "index.html",
        sites=sites,
        counts=counts,
        generated_at=data.get("generated_at", "-"),
        horizon=data.get("horizon_h", 6),
        metrics=metrics,
        history=data.get("history", {}),
        heatmap=data.get("heatmap", {}),
        alarm_types=data.get("alarm_types", {}),
        current_user=session.get("user", ""),
    )


@app.route("/api/predictions")
@login_required
def api_predictions():
    return jsonify(load_json(PREDICTIONS))


@app.route("/api/refresh", methods=["POST"])
@login_required
def api_refresh():
    """Recalcule les predictions a partir du dernier dataset/modele."""
    import json as _json
    from datetime import datetime

    from predict import build_alarm_types, build_heatmap, build_history, score_sites

    dataset = BASE / "data" / "dataset.pkl"
    model = BASE / "models" / "thermal_model.joblib"
    alarms = BASE / "data" / "alarms_clean.pkl"
    if not dataset.exists() or not model.exists():
        return jsonify({"ok": False, "message": "Dataset ou modele introuvable"}), 400

    scores, horizon = score_sites(dataset, model)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "horizon_h": horizon,
        "sites": _json.loads(scores.to_json(orient="records", date_format="iso")),
        "history": build_history(dataset, model),
        "heatmap": build_heatmap(alarms),
        "alarm_types": build_alarm_types(alarms),
    }
    PREDICTIONS.write_text(_json.dumps(payload))
    return jsonify({"ok": True, "message": f"{len(scores)} OLT rescores (horizon {horizon}h)"})


@app.route("/api/metrics")
@login_required
def api_metrics():
    return jsonify(load_json(METRICS))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
