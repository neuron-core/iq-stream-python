"""Stream tokens, as the server's IQ\\Auth\\Auth mints and checks them."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import secrets
from typing import cast

from ._protocol import Scope

_BASE64URL = re.compile(r"[A-Za-z0-9_-]+")


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _from_b64url(text: str) -> bytes:
    if not _BASE64URL.fullmatch(text):
        raise ValueError("not base64url")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def new_stream_id() -> str:
    """128 random bits as base64url: 22 characters. Never reuse an id for a second answer."""
    return b64url(secrets.token_bytes(16))


def claims(app_key: str, stream_id: str, exp: int) -> str:
    """The first half of a token: compact JSON in the order app_key, stream, exp, slashes unescaped. The three
    languages write this same form, so that their tokens compare byte for byte."""
    text = json.dumps({"app_key": app_key, "stream": stream_id, "exp": exp}, separators=(",", ":"), ensure_ascii=False)
    return b64url(text.encode("utf-8"))


def _signature(secret: bytes, scope: Scope, body: str) -> str:
    return b64url(hmac.new(secret, f"{scope}.{body}".encode("ascii"), hashlib.sha256).digest())


def sign(secret: bytes, scope: Scope, body: str) -> str:
    return f"{body}.{_signature(secret, scope, body)}"


def verify(secret: bytes, scope: Scope, app_key: str, stream_id: str, token: str, now: float) -> bool:
    """True when the token carries this app's signature for this scope and stream and has not expired."""
    body, dot, signature = token.partition(".")
    if not body or not dot or not signature or "." in signature or not _BASE64URL.fullmatch(body):
        return False
    if not hmac.compare_digest(_signature(secret, scope, body).encode("ascii"), signature.encode("ascii", "replace")):
        return False
    try:
        parsed: object = json.loads(_from_b64url(body))
    except (ValueError, binascii.Error):
        return False
    if not isinstance(parsed, dict):
        return False
    fields = cast("dict[str, object]", parsed)
    exp = fields.get("exp")
    return (
        fields.get("app_key") == app_key
        and fields.get("stream") == stream_id
        and isinstance(exp, (int, float))
        and not isinstance(exp, bool)
        and exp > now
    )
