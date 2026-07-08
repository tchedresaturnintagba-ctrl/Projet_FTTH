"""Authentification du dashboard ISOC.

- Utilisateurs stockes dans data/users.json avec mots de passe HACHES
  (scrypt via werkzeug) - jamais de mot de passe en clair.
- Cle secrete de session generee aleatoirement et persistee dans
  data/secret_key (a proteger comme un credential).

Gestion des comptes (CLI) :
    python src\\auth.py add <utilisateur>       # cree/modifie (mot de passe saisi masque)
    python src\\auth.py remove <utilisateur>
    python src\\auth.py list
"""
from __future__ import annotations

import json
import secrets
from functools import wraps
from pathlib import Path

from flask import redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE = Path(__file__).resolve().parent.parent
USERS_FILE = BASE / "data" / "users.json"
SECRET_FILE = BASE / "data" / "secret_key"


# ------------------------------------------------------------------ stockage
def load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}


def save_users(users: dict) -> None:
    USERS_FILE.parent.mkdir(exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))


def get_secret_key() -> str:
    """Cle de session persistante (generee au premier lancement)."""
    if SECRET_FILE.exists():
        return SECRET_FILE.read_text().strip()
    key = secrets.token_hex(32)
    SECRET_FILE.parent.mkdir(exist_ok=True)
    SECRET_FILE.write_text(key)
    return key


def ensure_default_admin() -> str | None:
    """Cree un compte admin au premier lancement et retourne son mot de passe
    temporaire (affiche UNE seule fois dans la console)."""
    users = load_users()
    if users:
        return None
    temp = secrets.token_urlsafe(9)
    users["admin"] = {"hash": generate_password_hash(temp), "role": "admin"}
    save_users(users)
    return temp


# ------------------------------------------------------------------ logique
def verify(username: str, password: str) -> bool:
    user = load_users().get(username)
    return bool(user and check_password_hash(user["hash"], password))


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user" not in session:
            if request.path.startswith("/api/"):
                return {"ok": False, "message": "Authentification requise"}, 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


# ------------------------------------------------------------------ CLI
def main() -> None:
    import argparse
    import getpass

    ap = argparse.ArgumentParser(description="Gestion des comptes du dashboard ISOC")
    ap.add_argument("action", choices=["add", "remove", "list"])
    ap.add_argument("username", nargs="?")
    args = ap.parse_args()

    users = load_users()
    if args.action == "list":
        for name, u in users.items():
            print(f"  {name} ({u.get('role', 'user')})")
        if not users:
            print("  (aucun compte)")
    elif args.action == "add":
        if not args.username:
            ap.error("utilisateur requis")
        pwd = getpass.getpass(f"Mot de passe pour '{args.username}' : ")
        if len(pwd) < 8:
            raise SystemExit("[!] 8 caracteres minimum.")
        if pwd != getpass.getpass("Confirmer : "):
            raise SystemExit("[!] Les mots de passe ne correspondent pas.")
        users[args.username] = {
            "hash": generate_password_hash(pwd),
            "role": users.get(args.username, {}).get("role", "user"),
        }
        save_users(users)
        print(f"[ok] compte '{args.username}' enregistre.")
    elif args.action == "remove":
        if not args.username:
            ap.error("utilisateur requis")
        if users.pop(args.username, None) is None:
            raise SystemExit(f"[!] compte '{args.username}' introuvable.")
        save_users(users)
        print(f"[ok] compte '{args.username}' supprime.")


if __name__ == "__main__":
    main()
