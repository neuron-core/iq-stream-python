"""Tokens, tickets, ids and the client's configuration."""

from __future__ import annotations

import asyncio
import pickle
import re
import ssl
from pathlib import Path

import pytest
from fakes import STREAM, Harness, answer, closed_fail, open_

from iq_stream import FrameTooLarge, IQClient, IQError, Ticket, new_stream_id

# The vector of core-client.md, produced by the server's own IQ\Auth\Auth::mint.
APP_KEY = "demo-app"
SECRET = "test-app-secret-0123456789abcdef0123"
EXP = 2_000_000_000
BODY = "eyJhcHBfa2V5IjoiZGVtby1hcHAiLCJzdHJlYW0iOiJRMmhoZEMxemRISmxZVzB0TURBd01RIiwiZXhwIjoyMDAwMDAwMDAwfQ"
VECTOR = {
    "pub": f"{BODY}.K9667Y9dJdNfJLyDD8CQ1Dtt5Kgjmc5vjreIFPX1ltQ",
    "sub": f"{BODY}.axHyIkb73vpsdXhSCIuvasYjNTZEcVNnVqaCbckYL5U",
}
URL = "http://iq.test:8080"
# A self-signed CA certificate, as a company's internal CA would be: only its subject is looked at.
INTERNAL_CA = """\
-----BEGIN CERTIFICATE-----
MIIBoDCCAUegAwIBAgIUcfHlikupAZnVTsvO3NTwqsY1XuIwCgYIKoZIzj0EAwIw
JTEjMCEGA1UEAwwaaXEtc3RyZWFtIGludGVybmFsIHRlc3QgQ0EwIBcNMjYxMTE0
MjAzNDI4WhgPMjEyNjEwMjEyMDM0MjhaMCUxIzAhBgNVBAMMGmlxLXN0cmVhbSBp
bnRlcm5hbCB0ZXN0IENBMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEV33O524w
uFAMlr4d3exYKAiJYtdj/IgcGqxAZiO/4Mtun21r1vkcgFjzKR/Fpg7sXGfFCZ1d
BwzYg8NPzR7es6NTMFEwHQYDVR0OBBYEFMAoXigtWJrCpp9sw3tqr1LxIwXkMB8G
A1UdIwQYMBaAFMAoXigtWJrCpp9sw3tqr1LxIwXkMA8GA1UdEwEB/wQFMAMBAf8w
CgYIKoZIzj0EAwIDRwAwRAIgXT7WT7TDI47dcnzMg1Cy5uPkMd5YIJS2rtbAGwll
OdsCIFa1IVPEtitt/PLEyPaoXH4xMl3xZ9ra50Im1FQdo/Vz
-----END CERTIFICATE-----
"""
ENV = ["IQ_STREAM_URL", "IQ_STREAM_APP_KEY", "IQ_STREAM_APP_SECRET", "IQ_STREAM_PUBLIC_URL", "IQ_STREAM_TRUST_ENV"]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def at(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    def set_time(seconds: float) -> None:
        monkeypatch.setattr("iq_stream._client.time.time", lambda: seconds)

    return set_time


def client(**options: object) -> IQClient:
    settings: dict[str, object] = {"url": URL, "app_key": APP_KEY, "app_secret": SECRET, **options}
    return IQClient(**settings)  # type: ignore[arg-type]


def test_token_reproduces_the_servers_vector_byte_for_byte(at) -> None:  # type: ignore[no-untyped-def]
    at(EXP - 3600)
    iq = client()
    assert iq.token("pub", STREAM) == VECTOR["pub"]
    assert iq.token("sub", STREAM) == VECTOR["sub"]
    assert iq.token("sub", STREAM, ttl=3600) == VECTOR["sub"]
    at(EXP - 59.1)
    assert iq.token("sub", STREAM, ttl=60) == VECTOR["sub"], "exp is floored"


def test_verify_accepts_its_own_tokens_and_nothing_else(at) -> None:  # type: ignore[no-untyped-def]
    at(EXP - 3600)
    iq = client()
    assert iq.verify("sub", VECTOR["sub"], STREAM) is True
    assert iq.verify("pub", VECTOR["pub"], STREAM) is True
    assert iq.verify("pub", VECTOR["sub"], STREAM) is False, "a sub token never passes as pub"
    assert iq.verify("sub", VECTOR["pub"], STREAM) is False
    assert iq.verify("sub", VECTOR["sub"], "another-stream-000001") is False, "another stream"
    changed = f"{BODY}.b{VECTOR['sub'][len(BODY) + 2 :]}"
    assert iq.verify("sub", changed, STREAM) is False, "a changed signature"
    assert iq.verify("sub", f"x{VECTOR['sub']}", STREAM) is False, "changed claims"
    for junk in ["", "a", "a.b", "a.b.c", f"{VECTOR['sub']}.x", ".", f"{BODY}.", f"{BODY}.é", None, 42]:
        assert iq.verify("sub", junk, STREAM) is False  # type: ignore[arg-type]
    assert client(app_secret="another-app-secret-0123456789abcdef").verify("sub", VECTOR["sub"], STREAM) is False
    assert client(app_key="other-app").verify("sub", iq.token("sub", STREAM), STREAM) is False
    with pytest.raises(ValueError):
        iq.verify("admin", VECTOR["sub"], STREAM)  # type: ignore[arg-type]
    at(EXP)
    assert iq.verify("sub", VECTOR["sub"], STREAM) is False, "expired"


def test_subscribe_url_and_ticket_use_the_public_address(at) -> None:  # type: ignore[no-untyped-def]
    at(EXP - 3600)
    iq = client(public_url="https://stream.example.com/")
    assert iq.subscribe_url(STREAM) == f"https://stream.example.com/{STREAM}?token={VECTOR['sub']}"
    ticket = iq.ticket(STREAM)
    assert ticket == Ticket(STREAM, VECTOR["sub"], f"https://stream.example.com/{STREAM}?token={VECTOR['sub']}", EXP)
    assert ticket.to_dict() == {"streamId": STREAM, "token": VECTOR["sub"], "url": ticket.url, "expiresAt": EXP}
    assert iq.ticket(STREAM, ttl=60).expires_at == EXP - 3540

    made = iq.ticket()
    assert re.fullmatch(r"[A-Za-z0-9_-]{22}", made.stream_id)
    assert iq.verify("sub", made.token, made.stream_id) is True
    assert made.url == f"https://stream.example.com/{made.stream_id}?token={made.token}"
    assert client().subscribe_url(STREAM) == f"{URL}/{STREAM}?token={VECTOR['sub']}"


def test_stream_ids_are_128_random_bits_and_are_validated() -> None:
    iq = client()
    ids = {iq.new_stream_id() for _ in range(200)} | {new_stream_id() for _ in range(200)}
    assert len(ids) == 400
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{22}", stream_id) for stream_id in ids)
    for bad in ["short", "x" * 129, "has space in it 0000000", "slash/in/it/0000000000", "", 42, None]:
        for call in (
            lambda bad=bad: iq.token("sub", bad),
            lambda bad=bad: iq.ticket(bad) if bad is not None else iq.ticket(42),  # type: ignore[arg-type]
            lambda bad=bad: iq.publisher(bad),
            lambda bad=bad: iq.reserve(bad),
            lambda bad=bad: iq.close_stream(bad),
        ):
            with pytest.raises(ValueError):
                call()
    with pytest.raises(ValueError):
        iq.token("admin", STREAM)  # type: ignore[arg-type]
    for ttl in (0, -1, float("nan"), float("inf"), True, "60"):
        with pytest.raises(ValueError):
            iq.token("sub", STREAM, ttl=ttl)  # type: ignore[arg-type]


def test_the_constructor_validates_its_options() -> None:
    with pytest.raises(IQError):
        client(app_secret="x" * 31)
    client(app_secret="x" * 32)
    client(app_secret="é" * 16)  # bytes, not characters
    for options in (
        {"app_key": "has space"},
        {"app_key": ""},
        {"url": "ftp://iq.test"},
        {"url": "iq.test:8080"},
        {"url": "http://"},
        {"url": "http://iq.test:port"},
        {"url": "http://iq.test/?a=b"},
        {"url": "unix:///run/iq/iq.sock"},
        {"url": "http+unix:///run/iq/iq.sock"},
        {"keepalive": 0},
        {"keepalive": -1},
        {"min_open": -1},
        {"max_frame_bytes": 0},
        {"max_frame_bytes": 1.5},
        {"retry_budget": float("nan")},
        {"retry_budget": float("inf")},
        {"takeover_budget": float("inf")},
        {"min_open": float("inf")},
        {"keepalive": float("inf")},
        {"keepalive": float("-inf")},
        {"queue_bytes": 0},
        {"on_error": "ignore"},
        {"error_handler": "not callable"},
    ):
        with pytest.raises(IQError):
            client(**options)
    iq = client(keepalive=None, min_open=0, url="https://iq.example.com/prefix/")
    assert iq.url == "https://iq.example.com/prefix"
    try:
        client(app_secret="too-short-secret")
    except IQError as error:
        assert "too-short-secret" not in str(error), "the secret is never in a message"


def test_options_default_to_the_environment(monkeypatch: pytest.MonkeyPatch, at) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(IQError):
        IQClient()
    with pytest.raises(IQError):
        IQClient("http://iq.test")
    monkeypatch.setenv("IQ_STREAM_URL", "http://iq.internal:8080/")
    monkeypatch.setenv("IQ_STREAM_APP_KEY", APP_KEY)
    monkeypatch.setenv("IQ_STREAM_APP_SECRET", SECRET)
    at(EXP - 3600)
    iq = IQClient()
    assert (iq.url, iq.public_url, iq.app_key) == ("http://iq.internal:8080", "http://iq.internal:8080", APP_KEY)
    assert iq.token("sub", STREAM) == VECTOR["sub"]
    monkeypatch.setenv("IQ_STREAM_PUBLIC_URL", "https://stream.example.com")
    assert IQClient().public_url == "https://stream.example.com"
    explicit = IQClient("http://other:1", public_url="https://public.example.com")
    assert (explicit.url, explicit.public_url) == ("http://other:1", "https://public.example.com")


@pytest.mark.parametrize(
    ("url", "trust_env"),
    [
        ("http://127.0.0.1:8080", False),
        ("http://localhost:8080", False),
        ("http://api.localhost", False),
        ("http://iq-stream:8080", False),  # a single-label name: a Compose or Kubernetes service
        ("http://10.1.2.3", False),
        ("http://192.168.0.10", False),
        ("http://172.16.0.1", False),
        ("http://169.254.1.1", False),
        ("http://[::1]:8080", False),
        ("http://[fd00::1]", False),
        ("https://stream.example.com", True),
        ("http://8.8.8.8", True),
    ],
)
def test_trust_env_is_off_for_loopback_and_private_addresses(url: str, trust_env: bool) -> None:
    assert client(url=url).trust_env is trust_env


def test_trust_env_follows_iq_stream_trust_env_and_the_option(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQ_STREAM_TRUST_ENV", "1")
    assert client(url="http://127.0.0.1").trust_env is True
    monkeypatch.setenv("IQ_STREAM_TRUST_ENV", "0")
    assert client(url="https://stream.example.com").trust_env is False
    assert client(url="https://stream.example.com", trust_env=True).trust_env is True
    monkeypatch.delenv("IQ_STREAM_TRUST_ENV")
    assert client(url="http://127.0.0.1", trust_env=True).trust_env is True


def test_the_ca_bundle_of_ssl_cert_file_counts_on_a_private_network_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = tmp_path / "internal-ca.pem"
    bundle.write_text(INTERNAL_CA, encoding="ascii")
    monkeypatch.setenv("SSL_CERT_FILE", str(bundle))
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    for url in ("https://10.0.0.5:8443", "https://iq:8443", "https://stream.example.com"):
        iq = client(url=url)
        context = iq._core.connector._ssl()  # pyright: ignore[reportPrivateUsage]
        subjects = [dict(field[0] for field in cert["subject"]) for cert in context.get_ca_certs()]  # type: ignore[index]
        assert {"commonName": "iq-stream internal test CA"} in subjects, f"{url}, trust_env={iq.trust_env}"


def test_a_public_url_that_is_not_an_http_address_is_refused_for_tickets() -> None:
    with pytest.raises(IQError, match="public_url"):
        client(public_url="stream.example.com").subscribe_url(STREAM)
    assert (
        client(public_url="https://stream.example.com")
        .subscribe_url(STREAM)
        .startswith(f"https://stream.example.com/{STREAM}?token=")
    )


def test_the_secret_does_not_show_when_the_client_is_printed() -> None:
    iq = client()
    assert SECRET not in repr(iq)
    assert SECRET not in str(iq)
    assert repr(iq) == f"IQClient(url={URL!r}, app_key={APP_KEY!r})"


def test_the_client_holds_no_connection_and_no_thread_until_it_needs_one() -> None:
    import threading

    before = threading.active_count()
    iq = client(verify=ssl.create_default_context())
    iq.ticket()
    assert threading.active_count() == before
    assert iq._core.connector._sync is None  # pyright: ignore[reportPrivateUsage]


def test_errors_carry_their_details_and_survive_pickling() -> None:
    error = IQError("IQ stream x: refused (HTTP 503)", stream_id="x", status=503, code="unavailable")
    copy = pickle.loads(pickle.dumps(error))
    assert (str(copy), copy.stream_id, copy.status, copy.code) == (str(error), "x", 503, "unavailable")
    plain = IQError("message")
    assert (plain.stream_id, plain.status, plain.code) == (None, None, None)
    too_large = pickle.loads(pickle.dumps(FrameTooLarge(1025, 1024)))
    assert isinstance(too_large, FrameTooLarge)
    assert (too_large.size, too_large.limit, str(too_large)) == (
        1025,
        1024,
        "The line is 1025 bytes; the limit is 1024.",
    )


@pytest.mark.parametrize("kind", ["sync", "async"])
async def test_reserve_sends_one_empty_post_and_close_stream_a_close_line(kind: str) -> None:
    h = Harness(kind)
    iq, transport = h.client, h.transport

    async def reserve(stream_id: str = STREAM):  # type: ignore[no-untyped-def]
        return iq.reserve(stream_id) if kind == "sync" else await iq.areserve(stream_id)

    async def close_stream(stream_id: str = STREAM, outcome: str = "fail"):  # type: ignore[no-untyped-def]
        if kind == "sync":
            return await asyncio.to_thread(iq.close_stream, stream_id, outcome)  # type: ignore[arg-type]
        return await iq.aclose_stream(stream_id, outcome)  # type: ignore[arg-type]

    transport.on_post = lambda body, n: open_(0)
    assert (await reserve()).last_event_id == 0
    transport.on_post = lambda body, n: answer(409, error="publisher_active")
    assert (await reserve()).error == "publisher_active", "the run has started already"
    assert transport.posts == ["", ""]
    assert all(iq.verify("pub", token, STREAM) for token in transport.post_tokens)

    transport.on_post = lambda body, n: closed_fail(2)
    assert (await close_stream()).outcome == "fail"
    transport.on_post = lambda body, n: answer(200, lastEventId=2, closed=True, outcome="end")
    assert (await close_stream(outcome="end")).outcome == "end"
    transport.on_post = lambda body, n: answer(409, error="stream_closed", outcome="end")
    already = await close_stream()
    assert (already.status, already.closed, already.outcome, already.error) == (409, True, "end", "stream_closed")
    assert transport.posts[2:] == ['{"close":"fail"}\n', '{"close":"end"}\n', '{"close":"fail"}\n']
    with pytest.raises(ValueError):
        await close_stream(outcome="done")

    transport.on_post = lambda body, n: answer(401, error="unauthorized")
    with pytest.raises(IQError) as refused:
        await reserve()
    assert (refused.value.status, refused.value.code, refused.value.stream_id) == (401, "unauthorized", STREAM)
    with pytest.raises(IQError):
        await close_stream()
    transport.on_post = lambda body, n: answer(409, error="stream_closed", outcome="end")
    with pytest.raises(IQError) as closed:
        await reserve()
    assert closed.value.code == "stream_closed"


@pytest.mark.parametrize("kind", ["sync", "async"])
async def test_close_stream_waits_for_a_publisher_still_attached_within_retry_budget(kind: str) -> None:
    async def close_stream(h: Harness):  # type: ignore[no-untyped-def]
        if kind == "sync":
            return await asyncio.to_thread(h.client.close_stream, STREAM)
        return await h.client.aclose_stream(STREAM)

    h = Harness(kind, retry_budget=1)
    h.transport.on_post = lambda body, n: answer(409, error="publisher_active") if n < 2 else closed_fail(4)
    assert (await close_stream(h)).closed is True
    assert len(h.transport.posts) == 3

    stuck = Harness(kind, retry_budget=0.3)
    stuck.transport.on_post = lambda body, n: answer(409, error="publisher_active")
    with pytest.raises(IQError) as raised:
        await close_stream(stuck)
    assert raised.value.code == "publisher_active"

    once = Harness(kind, retry_budget=0)
    once.transport.on_post = lambda body, n: answer(503, error="unavailable")
    with pytest.raises(IQError):
        await close_stream(once)
    assert once.transport.timeouts == [10.0], "retry_budget=0: one request, with its full timeout"
