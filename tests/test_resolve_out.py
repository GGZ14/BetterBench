import pytest

from betterbench.cli import _resolve_out


def test_out_refuses_existing_dir(tmp_path):
    # Regression: reusing a --out that already exists as a directory wrote nothing
    # and left the PREVIOUS run's results.json there to be quoted as fresh.
    with pytest.raises(SystemExit):
        _resolve_out(str(tmp_path), "m", None, "results.json")


def test_out_plain_path_wins(tmp_path):
    f = tmp_path / "r.json"
    assert _resolve_out(str(f), "m", None, "results.json") == f
