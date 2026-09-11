"""Default output location: every run gets its own, versioned directory under
$BETTERBENCH_HOME (default ~/.betterbench)/runs/ — never the cwd, and a
second run in the same second gets `-2` instead of overwriting. An explicit
--out always wins."""
from __future__ import annotations

import json
from pathlib import Path

from betterbench.cli import main
from betterbench import runs as runs_mod

FAST = {"warmup": 0, "runs_per_category": 1,
        "concurrency_levels": [1], "concurrency_requests": 1,
        "prefill_depths": [200], "prefill_runs": 1, "prefill_warmup": 0}


def _run_prefill(server, home: Path, monkeypatch, *flags):
    monkeypatch.setenv("BETTERBENCH_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "cfg.json"
    cfg.write_text(json.dumps(FAST))
    main(["run", "--endpoint", server(), "--model", "mock",
          "--config", str(cfg), "--prefill", *flags])


def test_home_env_moves_the_base(monkeypatch, tmp_path):
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    assert runs_mod.betterbench_home() == tmp_path / "bb"


def test_default_home_is_the_dotfolder(monkeypatch):
    monkeypatch.delenv("BETTERBENCH_HOME", raising=False)
    assert runs_mod.betterbench_home() == Path.home() / ".betterbench"


def test_slug_sanitizes_and_lowercases():
    assert runs_mod.slug("Qwen3-30B / mx-FP8 (v2)") == "qwen3-30b-mx-fp8-v2"
    assert runs_mod.slug("!!!") == ""


def test_run_without_out_writes_into_a_versioned_run_dir(server, tmp_path, monkeypatch):
    home = tmp_path / "bb"
    _run_prefill(server, home, monkeypatch)
    run_dirs = list((home / "runs").iterdir())
    assert len(run_dirs) == 1
    d = run_dirs[0]
    assert d.name.startswith("20")               # timestamp prefix
    assert d.name.endswith("-mock")                # model slug
    assert (d / "results.json").exists()
    assert (d / "results.html").exists()            # report beside the result


def test_name_flag_replaces_the_slug(server, tmp_path, monkeypatch):
    home = tmp_path / "bb"
    _run_prefill(server, home, monkeypatch, "--name", "mxfp4-flip")
    d = next(iter((home / "runs").iterdir()))
    assert d.name.endswith("-mxfp4-flip")


def test_two_runs_in_one_second_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    monkeypatch.setattr(runs_mod.time, "strftime", lambda fmt, *a: "20260101-000000")
    first = runs_mod.allocate_run_dir("Qwen3.8")
    second = runs_mod.allocate_run_dir("Qwen3.8")
    assert first.name != second.name
    assert second.name == "20260101-000000-qwen3-8-2"


def test_explicit_out_still_wins(server, tmp_path, monkeypatch):
    home = tmp_path / "bb"
    monkeypatch.setenv("BETTERBENCH_HOME", str(home))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(FAST))
    out = tmp_path / "elsewhere" / "my.json"
    main(["run", "--endpoint", server(), "--model", "mock",
          "--config", str(cfg), "--out", str(out), "--no-html", "--prefill"])
    assert out.exists()
    assert not (home / "runs").exists()            # nothing else was created


def test_ab_also_writes_a_run_dir_default(server, tmp_path, monkeypatch):
    url_a = server()
    url_b = server()
    home = tmp_path / "bb"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BETTERBENCH_HOME", str(home))
    cfg = home / "cfg.json"
    cfg.write_text(json.dumps({"warmup": 0, "ab_min_pairs": 1,
                               "ab_max_pairs": 2}))
    main(["ab", "--endpoint-a", url_a, "--endpoint-b", url_b, "--model", "mock",
          "--config", str(cfg)])
    run_dirs = list((home / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "ab.json").exists()


def test_report_and_compare_no_longer_scatter_files_in_the_cwd(server, tmp_path, monkeypatch):
    """Regression: `main()` used to mkdir a stray `results/x/` in the cwd
    before *any* command, so even `report file.json` (no --out) dirtied the
    working directory. Now only an explicit --out pre-creates a path."""
    r = tmp_path / "bb" / "r.json"
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(FAST))
    main(["run", "--endpoint", server(), "--model", "mock",
          "--config", str(cfg), "--out", str(r), "--no-html", "--prefill"])
    cwd = tmp_path / "dirty-cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    main(["report", str(r)])
    main(["compare", str(r), str(r)])
    assert list(cwd.iterdir()) == []
