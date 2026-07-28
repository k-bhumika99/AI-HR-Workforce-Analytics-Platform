"""
Lightweight authentication for the AI HR Workforce Analytics Platform.

Users are stored in a local JSON file (data/users.json) with salted
password hashes (Werkzeug's generate_password_hash / check_password_hash).
This is intentionally simple — no external auth service required — but
still keeps plaintext passwords out of storage.
"""

import os
import re
import json
import uuid
from datetime import datetime
from functools import wraps

from flask import session, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _ensure_store():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)


def load_users() -> dict:
    _ensure_store()
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_users(users: dict):
    _ensure_store()
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def find_user_by_email(email: str):
    users = load_users()
    email = email.strip().lower()
    for username, u in users.items():
        if u.get("email", "").strip().lower() == email:
            return username, u
    return None, None


def create_user(name: str, email: str, password: str):
    """Returns (ok: bool, error: str|None)."""
    name = (name or "").strip()
    email = (email or "").strip().lower()
    password = password or ""

    if not name:
        return False, "Please enter your full name."
    if not EMAIL_RE.match(email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    existing_user, _ = find_user_by_email(email)
    if existing_user:
        return False, "An account with this email already exists."

    users = load_users()
    username = email  # email doubles as the unique username/key
    users[username] = {
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    save_users(users)
    return True, None


def verify_user(email: str, password: str):
    """Returns the user dict (with 'username' key added) or None."""
    username, user = find_user_by_email(email)
    if not user:
        return None
    if not check_password_hash(user.get("password_hash", ""), password or ""):
        return None
    return {**user, "username": username}


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            flash("Please sign in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped
