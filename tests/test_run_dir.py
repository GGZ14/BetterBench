"""Default output location: every run gets its own, versioned directory under
$BETTERBENCH_HOME (default ~/.betterbench)/runs/ — never the cwd, and a
second run in the same second gets `-2` instead of overwriting. An explicit
--out always wins."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

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


def test_relative_home_is_explicitly_cwd_based_and_warns_once(monkeypatch, capsys, tmp_path):
    """A relative BETTERBENCH_HOME is resolved against the cwd explicitly and
    warns on stderr exactly once — the first call resolves it the same way
    a bare relative path already would, so behaviour is unchanged, but the
    cwd-reliance becomes visible (without --out, runs land in the cwd)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BETTERBENCH_HOME", "bb")
    monkeypatch.setattr(runs_mod, "_warned_relative_home", False)
    first = runs_mod.betterbench_home()
    second = runs_mod.betterbench_home()
    assert first == tmp_path / "bb"          # under the cwd at call time
    assert first.is_absolute()
    assert first == second
    err = capsys.readouterr().err
    assert "BETTERBENCH_HOME is relative" in err
    assert "./bb/runs/" in err
    assert err.count("warning:") == 1        # once, even after two calls


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


def test_slug_caps_long_labels_with_a_sha_suffix():
    long = runs_mod.slug("q" * 300)
    assert long.startswith("q" * 64)              # capped on the sanitised slug
    tail = long[len(long) - 8:]
    assert len(long) == 64 + 1 + len(tail)       # 64 + '-' + short hash
    int(tail, 16)                                 # hex
    # short slugs are uncapped and hash-free: 'qwen3-480b-instruct'
    assert runs_mod.slug("qwen3-480b-instruct") == "qwen3-480b-instruct"
    assert runs_mod.slug("q" * 64) == "q" * 64


def test_slug_transliterates_unicode():
    assert runs_mod.slug("Módèle-XL") == "modele-xl"
    assert runs_mod.slug("Über-Übung (v2)") == "uber-ubung-v2"
    assert runs_mod.slug("Módèle-XL (mx-fp8)") == "modele-xl-mx-fp8"
    # cap + hash path still works on a >SLUG_MAX input that normalises
    # from (longer) unicode text — 'é'*200 -> 'e'*200
    long = runs_mod.slug("é" * 200)
    assert long.startswith("e" * runs_mod.SLUG_MAX)
    tail = long[len(long) - 8:]
    assert len(long) == runs_mod.SLUG_MAX + 1 + len(tail)
    int(tail, 16)                             # hex



def test_two_runs_in_one_second_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    monkeypatch.setattr(runs_mod.time, "strftime", lambda fmt, *a: "20260101-000000")
    first = runs_mod.allocate_run_dir("Qwen3.8")
    second = runs_mod.allocate_run_dir("Qwen3.8")
    assert first.name != second.name
    assert second.name == "20260101-000000-qwen3-8-2"


def test_precreated_suffix_is_skipped_not_reused(monkeypatch, tmp_path):
    """The mkdir outcome is the check: a pre-existing <stamp>-<slug>-2 must
    never be handed back (never reuses an existing path); the free -3 is
    taken instead."""
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    monkeypatch.setattr(runs_mod.time, "strftime", lambda fmt, *a: "20260101-000000")
    runs_root = tmp_path / "bb" / "runs"
    runs_root.mkdir(parents=True)
    (runs_root / "20260101-000000-qwen3-8-2").mkdir()  # left by a prior run; -3 stays free
    first = runs_mod.allocate_run_dir("Qwen3.8")
    second = runs_mod.allocate_run_dir("Qwen3.8")
    assert first.name == "20260101-000000-qwen3-8"
    assert second.name == "20260101-000000-qwen3-8-3"   # skipped the pre-created -2


def test_concurrent_allocations_are_race_safe(monkeypatch, tmp_path):
    """Two processes racing on the same second + label: each must get a
    unique directory and none may raise. With the old check-then-act
    loop both threads could pass exists() before either mkdir'd, and the
    loser of mkdir(exist_ok=False) died with an uncaught
    FileExistsError."""
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    monkeypatch.setattr(runs_mod.time, "strftime", lambda fmt, *a: "20260101-000000")
    names, errors = [], []
    lock = threading.Lock()
    rounds = 3
    for _ in range(rounds):
        barrier = threading.Barrier(2, timeout=10)
        workdirs = []

        def work():
            try:
                barrier.wait()
                workdirs.append(runs_mod.allocate_run_dir("Qwen3.8"))
            except Exception as e:  # noqa: BLE001 - any failure is the bug
                with lock:
                    errors.append(e)

        ts = [threading.Thread(target=work) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=10)
            assert not t.is_alive()
        with lock:
            names.extend(p.name for p in workdirs)
    assert errors == []
    assert len(names) == 2 * rounds
    assert len(set(names)) == len(names)      # every directory unique
    assert all((tmp_path / "bb" / "runs" / n).is_dir() for n in names)


def test_a_300_char_model_name_still_gets_a_run_dir(monkeypatch, tmp_path):
    """300-char ids used to make allocate_run_dir raise a raw
    OSError: [Errno 36] File name too long; the slug is now capped."""
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    d = runs_mod.allocate_run_dir("m" * 300)
    assert d.is_dir()
    assert len(d.name) < 300              # shorter than the input


def test_similar_long_labels_get_different_dir_names(monkeypatch, tmp_path):
    """Same 64-char prefix, different tail -> the hash suffix disambiguates."""
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    a = runs_mod.allocate_run_dir("p" * 64 + "left" + "r" * 40)
    b = runs_mod.allocate_run_dir("p" * 64 + "right" + "r" * 39)
    assert a.is_dir() and b.is_dir()
    assert a.name.split("-", 2)[2] != b.name.split("-", 2)[2]


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


def test_a_validation_failure_leaves_no_run_dir(server, tmp_path, monkeypatch):
    """A run that dies in validation (every phase switched off) must not
    have pre-allocated its run directory for nothing — the old code
    populated `~/.../runs/<stamp>-mock/` empty before the phase check
    fired. That in itself is an empty directory that nobody cleans up.
    Allocation now happens at write time, so a failed run leaves nothing."""
    home = tmp_path / "bb"
    monkeypatch.setenv("BETTERBENCH_HOME", str(home))
    home.mkdir(parents=True, exist_ok=True)
    cfg = home / "cfg.json"
    cfg.write_text(json.dumps({**FAST, "run_single_stream": False,
                               "run_prefill": False, "run_concurrency": False}))
    with pytest.raises(SystemExit):
        main(["run", "--endpoint", server(), "--model", "mock",
              "--config", str(cfg)])
    assert not (home / "runs").exists()        # no (empty) run directory at all


def test_a_missing_corpus_run_leaves_no_run_dir(server, tmp_path, monkeypatch):
    """Same regression from the corpus side: `--corpus <empty>` fails right
    before the first request is ever sent. That empty run directory was used
    to be created and left behind here too."""
    home = tmp_path / "bb"
    monkeypatch.setenv("BETTERBENCH_HOME", str(home))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(FAST))
    empty = tmp_path / "no-corpus"
    empty.mkdir()
    with pytest.raises(SystemExit):
        main(["run", "--endpoint", server(), "--model", "mock",
              "--config", str(cfg), "--corpus", str(empty), "--decode"])
    assert not (home / "runs").exists()


def test_plan_run_dir_names_without_creating(monkeypatch, tmp_path):
    """plan_run_dir answers 'where will this run's directory be' without
    creating anything — it's the pre-browser-preview of the write-time
    allocation, and must not have the same side effect."""
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    monkeypatch.setattr(runs_mod.time, "strftime", lambda fmt, *a: "20260101-000000")
    p = runs_mod.plan_run_dir("Qwen3.8", "mxfp4-flip")
    assert p == tmp_path / "bb" / "runs" / "20260101-000000-mxfp4-flip"
    assert not p.parent.exists()              # not even the runs root
    # repeated planning is a pure read (can't bump into a -2 suffix)
    assert p == runs_mod.plan_run_dir("Qwen3.8", "mxfp4-flip")


def test_plan_and_allocate_agree_name(monkeypatch, tmp_path):
    """What the preview promised, allocation will deliver (absent a race): same
    stamp + slug + no suffix when nothing exists yet."""
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path / "bb"))
    monkeypatch.setattr(runs_mod.time, "strftime", lambda fmt, *a: "20260101-000000")
    plan = runs_mod.plan_run_dir("Qwen3.8")
    got = runs_mod.allocate_run_dir("Qwen3.8")
    assert got.name == plan.name             # no -2: nothing was pre-empting
    assert got.is_dir()
