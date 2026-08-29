"""`run --decode / --prefill / --concurrency` measure only the phase asked for."""
from __future__ import annotations

import json

import pytest

from betterbench.cli import main

# Small enough that the whole matrix of phase selections runs in seconds.
FAST = {"warmup": 0, "runs_per_category": 1,
        "concurrency_levels": [1, 2], "concurrency_requests": 2,
        "prefill_depths": [200, 400], "prefill_runs": 1, "prefill_warmup": 0}

SECTIONS = ("single_stream", "prefill", "concurrency")


def _run(server, tmp_path, *flags, cfg=None):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({**FAST, **(cfg or {})}))
    out = tmp_path / "r.json"
    main(["run", "--endpoint", server(), "--model", "mock",
          "--config", str(cfg_path), "--out", str(out), "--no-html", *flags])
    return json.loads(out.read_text())


@pytest.mark.parametrize("flag,section", [
    ("--decode", "single_stream"),
    ("--prefill", "prefill"),
    ("--concurrency", "concurrency"),
])
def test_one_phase_flag_runs_only_that_phase(server, tmp_path, flag, section):
    res = _run(server, tmp_path, flag)
    assert res[section]
    assert [s for s in SECTIONS if s in res] == [section]


def test_phase_flags_combine(server, tmp_path):
    res = _run(server, tmp_path, "--decode", "--prefill")
    assert [s for s in SECTIONS if s in res] == ["single_stream", "prefill"]


def test_no_flags_still_runs_everything(server, tmp_path):
    from betterbench.report import render_markdown
    res = _run(server, tmp_path)
    assert all(res.get(s) for s in SECTIONS)
    # A complete run's header keeps passes/cat and gains no phase caveat.
    header = render_markdown(res).splitlines()[3]
    assert "passes/cat" in header and "phases" not in header


def test_prefill_only_needs_no_corpus(server, tmp_path):
    """The depth sweep builds its own prompts, so an empty corpus is fine."""
    empty = tmp_path / "empty-corpus"
    empty.mkdir()
    res = _run(server, tmp_path, "--prefill", "--corpus", str(empty))
    assert res["prefill"] and "single_stream" not in res


def test_decode_only_with_an_empty_corpus_still_errors(server, tmp_path):
    empty = tmp_path / "empty-corpus2"
    empty.mkdir()
    with pytest.raises(SystemExit):
        _run(server, tmp_path, "--decode", "--corpus", str(empty))


def test_selecting_and_disabling_the_same_phase_is_an_error(server, tmp_path):
    with pytest.raises(SystemExit):
        _run(server, tmp_path, "--prefill", "--no-prefill")


def test_config_file_can_switch_a_phase_off(server, tmp_path):
    """A `run_*: false` in --config used to be overwritten by the CLI default."""
    res = _run(server, tmp_path, cfg={"run_concurrency": False})
    assert "concurrency" not in res and res["single_stream"]


def test_phase_flags_beat_the_config_file(server, tmp_path):
    res = _run(server, tmp_path, "--concurrency", cfg={"run_concurrency": False})
    assert [s for s in SECTIONS if s in res] == ["concurrency"]


def test_a_phase_only_result_renders(server, tmp_path):
    """The report must not assume a single-stream section is present."""
    from betterbench.html_report import render_html
    from betterbench.report import render_markdown
    res = _run(server, tmp_path, "--prefill")
    md = render_markdown(res)
    assert "Prompt processing (prefill) sweep" in md
    assert "Single-stream" not in md
    # passes/cat measures the phase that did not run; the header says so instead.
    assert "passes/cat" not in md and "**phases**: prefill" in md
    assert "Prompt processing throughput by depth" in render_html(res)
