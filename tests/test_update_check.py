"""The release check: correct about versions, cached, silent on every failure,
never on the wire while a measurement is being timed, and switchable off."""
from __future__ import annotations

import json
import time

import pytest

from betterbench import update as up
from betterbench.cli import main

FAST = {"warmup": 0, "runs_per_category": 1,
        "concurrency_levels": [1], "concurrency_requests": 1,
        "prefill_depths": [200], "prefill_runs": 1, "prefill_warmup": 0}


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    """Turn the check back on (conftest disables it everywhere) and point its
    cache at a temp home, so no test can touch the real ~/.betterbench."""
    monkeypatch.delenv("BETTERBENCH_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    monkeypatch.setattr(up, "_thread", None)
    monkeypatch.setattr(up, "_latest", None)
    return tmp_path / "bb" / up.CACHE_NAME


# --------------------------------------------------------------------------- #
# Version comparison
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag,expected", [
    ("v0.4.1", (0, 4, 1)), ("0.4.1", (0, 4, 1)), (" v1.2 ", (1, 2)),
    ("0.5.0rc1", None), ("v0.4.0-hotfix", None), ("nightly", None), ("", None),
])
def test_only_plain_release_tags_parse(tag, expected):
    """A prerelease read leniently as its base version would announce an rc as
    the latest release."""
    assert up._version_tuple(tag) == expected


@pytest.mark.parametrize("latest,current,newer", [
    ("0.5.0", "0.4.0", True), ("v0.4.1", "0.4.0", True),
    ("0.10.0", "0.9.0", True),        # not a string sort
    ("0.4", "0.4.0", False),          # padded: 0.4 == 0.4.0
    ("0.4.0", "0.4.0", False), ("0.4.0", "0.5.0", False),
    ("0.5.0rc1", "0.4.0", False),     # unparseable is never "newer"
    ("garbage", "0.4.0", False),
])
def test_is_newer(latest, current, newer):
    assert up._is_newer(latest, current) is newer


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def test_a_fresh_cache_is_used_without_any_network(enabled, monkeypatch):
    enabled.parent.mkdir(parents=True)
    enabled.write_text(json.dumps({"checked_at": time.time(), "latest": "9.9.9"}))
    monkeypatch.setattr(up, "_fetch_latest", _never_called)
    up.start()
    up.settle(timeout=5)
    assert "9.9.9" in (up.notice() or "")


def test_a_stale_cache_is_refetched(enabled, monkeypatch):
    enabled.parent.mkdir(parents=True)
    enabled.write_text(json.dumps({"checked_at": time.time() - up.CACHE_TTL_S - 1,
                                   "latest": "0.0.1"}))
    monkeypatch.setattr(up, "_fetch_latest", lambda: "9.9.9")
    up.start()
    up.settle(timeout=5)
    assert "9.9.9" in (up.notice() or "")
    assert json.loads(enabled.read_text())["latest"] == "9.9.9"


def test_a_repo_with_no_release_tags_is_cached_too(enabled, monkeypatch):
    """`""` is a real answer. Caching only hits would re-request on every
    single invocation for a repo that has never been tagged."""
    monkeypatch.setattr(up, "_fetch_latest", lambda: None)
    up.start()
    up.settle(timeout=5)
    assert up.notice() is None
    assert json.loads(enabled.read_text())["latest"] == ""

    monkeypatch.setattr(up, "_fetch_latest", _never_called)
    up.start()
    up.settle(timeout=5)
    assert up.notice() is None


def test_a_corrupt_cache_is_just_rechecked(enabled, monkeypatch):
    enabled.parent.mkdir(parents=True)
    enabled.write_text("{not json at all")
    monkeypatch.setattr(up, "_fetch_latest", lambda: "9.9.9")
    up.start()
    up.settle(timeout=5)
    assert "9.9.9" in (up.notice() or "")


# --------------------------------------------------------------------------- #
# Every failure is silence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("boom", [
    OSError("network unreachable"),        # offline
    TimeoutError("timed out"),             # a proxy swallowing it
    ValueError("Expecting value"),         # garbage where JSON should be
])
def test_a_failed_check_is_silent(enabled, monkeypatch, capsys, boom):
    def raise_it():
        raise boom

    monkeypatch.setattr(up, "_fetch_latest", raise_it)
    up.start()
    up.settle(timeout=5)
    assert up.notice() is None
    assert capsys.readouterr().err == ""


def test_an_unwritable_cache_does_not_break_the_check(enabled, monkeypatch):
    """A read-only home is not this feature's business to complain about."""
    enabled.parent.parent.mkdir(parents=True, exist_ok=True)
    enabled.parent.write_text("a file where the home directory should be")
    monkeypatch.setattr(up, "_fetch_latest", lambda: "9.9.9")
    up.start()
    up.settle(timeout=5)
    assert "9.9.9" in (up.notice() or "")      # still reported, just not cached


# --------------------------------------------------------------------------- #
# Wiring into the CLI
# --------------------------------------------------------------------------- #
def test_a_run_reports_the_update_after_the_results(server, tmp_path, monkeypatch,
                                                    capsys, enabled):
    monkeypatch.setattr(up, "_fetch_latest", lambda: "v99.0.0")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(FAST))
    main(["run", "--endpoint", server(), "--model", "mock", "--config", str(cfg),
          "--out", str(tmp_path / "r.json"), "--no-html", "--prefill"])
    out = capsys.readouterr().out
    assert "newer BetterBench is available: v99.0.0" in out
    assert out.index("wrote ") < out.index("newer BetterBench")   # after the run


def test_the_check_is_settled_before_the_first_measured_request(server, tmp_path,
                                                                monkeypatch, enabled):
    """A benchmark may not have an HTTPS request to github.com in flight while
    it is timing a request to the endpoint under test."""
    import betterbench.cli as cli

    in_flight, overlapped = [], []

    def slow_fetch():
        in_flight.append(1)
        time.sleep(0.2)
        in_flight.pop()
        return "9.9.9"

    real = cli.prefill_sweep

    def watched(*a, **k):
        overlapped.append(bool(in_flight))
        return real(*a, **k)

    monkeypatch.setattr(up, "_fetch_latest", slow_fetch)
    monkeypatch.setattr(cli, "prefill_sweep", watched)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(FAST))
    main(["run", "--endpoint", server(), "--model", "mock", "--config", str(cfg),
          "--out", str(tmp_path / "r.json"), "--no-html", "--prefill"])
    assert overlapped == [False]


def test_the_flag_switches_the_check_off(server, tmp_path, monkeypatch, capsys,
                                         enabled):
    monkeypatch.setattr(up, "_fetch_latest", _never_called)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(FAST))
    main(["run", "--endpoint", server(), "--model", "mock", "--config", str(cfg),
          "--out", str(tmp_path / "r.json"), "--no-html", "--prefill",
          "--no-update-check"])
    assert "newer BetterBench" not in capsys.readouterr().out
    assert not enabled.exists()          # not even a cache file


def test_the_env_var_switches_the_check_off(enabled, monkeypatch):
    monkeypatch.setenv("BETTERBENCH_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(up, "_fetch_latest", _never_called)
    up.start()
    up.settle(timeout=5)
    assert up.notice() is None


def _never_called():
    raise AssertionError("the network was reached when it should not have been")
