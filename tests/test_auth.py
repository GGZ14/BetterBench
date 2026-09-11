"""Bearer authentication: sent on the wire when configured, 401 without it on a
gated server, and the key never lands in any result the run produced."""
from __future__ import annotations

import json

from betterbench.client import get_model_context, stream_chat_sync
from betterbench.cli import main

SEKRET = "sekret-bearer-key-42"


def _chat(url: str, **kw):
    return stream_chat_sync(url, "mock", [{"role": "user", "content": "say hi"}],
                            max_tokens=8, temperature=0.0, **kw)


def test_no_key_against_a_gated_server_is_a_401(server):
    r = _chat(server(api_key=SEKRET))
    assert not r.ok
    assert "401" in (r.error or "")


def test_misrouted_key_is_a_401_too(server):
    r = _chat(server(api_key=SEKRET), api_key="wrong")
    assert not r.ok
    assert "401" in (r.error or "")


def test_correct_key_gets_through(server):
    assert _chat(server(api_key=SEKRET), api_key=SEKRET).ok


def test_models_probe_authenticates_too(server):
    """The context-window probe starts a prefill sweep; a silent 401 there
    would read as 'unknown max context' and let every depth be rejected
    mid-sweep instead of skipped up front."""
    url = server(api_key=SEKRET, max_ctx=4096)
    assert get_model_context(url, "mock", api_key=SEKRET) == 4096
    assert get_model_context(url, "mock") is None


def _spy_headers(monkeypatch):
    import betterbench.client as client

    seen: list[dict] = []
    orig = client.http.client.HTTPConnection.request

    def spy(self, method, path, body=None, headers={}):
        seen.append({k.lower(): v for k, v in (headers or {}).items()})
        return orig(self, method, path, body, headers)

    monkeypatch.setattr(client.http.client.HTTPConnection, "request", spy)
    return seen


def test_without_a_key_no_authorization_header_is_sent(server, monkeypatch):
    seen = _spy_headers(monkeypatch)
    _chat(server())
    assert seen, "no request went out"
    assert all("authorization" not in h for h in seen)


def test_with_a_key_the_bearer_header_is_sent(server, monkeypatch):
    seen = _spy_headers(monkeypatch)
    _chat(server(api_key=SEKRET), api_key=SEKRET)
    assert seen
    assert any(h.get("authorization") == "Bearer " + SEKRET for h in seen)


def test_run_uses_the_key_and_never_records_it(server, tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"warmup": 0, "prefill_warmup": 0,
                               "prefill_runs": 1, "prefill_depths": [300]}))
    out = tmp_path / "r.json"
    main(["run", "--endpoint", server(api_key=SEKRET), "--model", "mock",
          "--config", str(cfg), "--out", str(out), "--no-html", "--prefill",
          "--api-key", SEKRET])
    text = out.read_text()
    assert SEKRET not in text                      # key stays off the disk
    assert json.loads(text)["prefill"]


def test_env_var_fills_in_for_a_missing_flag(server, tmp_path, monkeypatch):
    monkeypatch.setenv("BETTERBENCH_API_KEY", SEKRET)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"warmup": 0, "prefill_warmup": 0,
                               "prefill_runs": 1, "prefill_depths": [300]}))
    out = tmp_path / "r.json"
    main(["run", "--endpoint", server(api_key=SEKRET), "--model", "mock",
          "--config", str(cfg), "--out", str(out), "--no-html", "--prefill"])
    assert json.loads(out.read_text())["prefill"]


def test_ab_authenticates_each_endpoint_with_its_own_key(server, tmp_path):
    """A/B keys are per-endpoint: broken pairing here means every pair drops
    (a 401'd call fails the pair), so >0 pairs proves the plumbing end to end."""
    url_a = server(api_key="key-a-secret")
    url_b = server(api_key="key-b-secret")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"warmup": 0,
                               "ab_min_pairs": 1, "ab_max_pairs": 2}))
    out = tmp_path / "ab.json"
    main(["ab", "--endpoint-a", url_a, "--endpoint-b", url_b, "--model", "mock",
          "--config", str(cfg), "--out", str(out),
          "--api-key-a", "key-a-secret", "--api-key-b", "key-b-secret"])
    text = out.read_text()
    assert json.loads(text)["pairs"] >= 1
    assert "key-a-secret" not in text and "key-b-secret" not in text
