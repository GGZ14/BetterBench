"""Result files are written atomically: text is put into a temp file in
the same directory, fsync'd, and renamed onto the target. A run that
dies mid-write must never leave behind a half-written file that a
reader would mistake for complete."""
from __future__ import annotations

import os

import pytest

from betterbench.cli import _atomic_write


def test_atomic_write_creates_parents_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "a" / "b" / "results.json"
    _atomic_write(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"
    assert list(p.parent.iterdir()) == [p]        # no .tmp residue, no extra files


def test_atomic_write_overwrites_completely(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("a shorter old file that used to be here")
    _atomic_write(p, "xy")                          # shorter than the old file
    assert p.read_text(encoding="utf-8") == "xy"   # no old tail bleeding through


def test_atomic_write_keeps_the_old_file_when_the_write_fails(tmp_path, monkeypatch):
    """A failure mid-write (disk full here, simulated with a fsync raise)
    must not touch the existing target: the half-written bytes live in the
    gone .tmp, and the on-disk result stays exactly what it was before."""
    p = tmp_path / "results.json"
    p.write_text("old complete results")

    def boom(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        _atomic_write(p, "partial junk that must never land on the target")
    assert p.read_text(encoding="utf-8") == "old complete results"
    assert not [f for f in p.parent.iterdir() if f.name.endswith(".tmp")]  # tmp cleaned up


def test_atomic_write_cleanups_the_tmp_when_replacement_fails(tmp_path, monkeypatch):
    """Even when the rename itself fails, the .tmp in the same directory must
    not be left dangling: it holds bytes that belong nowhere."""
    p = tmp_path / "r.json"
    monkeypatch.setattr(os, "replace", lambda a, b: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        _atomic_write(p, "data")
    assert not [f for f in p.parent.iterdir() if f.name.endswith(".tmp")]
