"""Password hashing, JWT tokens and the role/clearance model."""

from __future__ import annotations

import hashlib
import hmac
import os
import time

import jwt

ROLES = ("reader", "editor", "admin")
ACCESS_TAGS = ("public", "internal", "restricted")

# Classifier sensitivity label -> default access tag on the document.
SENSITIVITY_TO_TAG = {"public": "public", "internal": "internal", "confidential": "restricted"}

# Default clearances given to a new user of each role (admins can change them per user).
DEFAULT_CLEARANCES = {
    "reader": ["public", "internal"],
    "editor": ["public", "internal", "restricted"],
    "admin": ["public", "internal", "restricted"],
}

# What each role is allowed to do.
PERMISSIONS = {
    "reader": {"query"},
    "editor": {"query", "add_doc"},
    "admin": {"query", "add_doc", "remove_doc", "manage_users", "view_audit"},
}


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"pbkdf2${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000)
    return hmac.compare_digest(digest.hex(), digest_hex)


def make_token(username: str, secret: str, ttl_s: int = 12 * 3600) -> str:
    now = int(time.time())
    return jwt.encode({"sub": username, "iat": now, "exp": now + ttl_s}, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> str | None:
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        return None


def can(role: str, permission: str) -> bool:
    return permission in PERMISSIONS.get(role, set())


def readable_tags(user: dict) -> list[str]:
    """Access tags this user may retrieve. Admins always see everything."""
    if user["role"] == "admin":
        return list(ACCESS_TAGS)
    return [t for t in user["clearances"] if t in ACCESS_TAGS]
