"""Tests for universe build-age sidecar and reuse-window logic."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import GetFMPData.build_stock_universe as bsu
from GetFMPData.build_stock_universe import universe_age_days, write_universe_meta
from util.data_pull.pull import get_universe

UNIVERSE = pd.DataFrame({"symbol": ["AAA", "BBB"], "exchange": ["NYSE", "NASDAQ"]})


def _write_universe(tmp_path, age_days=None):
    csv_path = tmp_path / "universe.csv"
    UNIVERSE.to_csv(csv_path, index=False)
    if age_days is not None:
        built = datetime.now(timezone.utc) - timedelta(days=age_days)
        meta = {"built_at_utc": built.isoformat(), "rows": len(UNIVERSE)}
        csv_path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return csv_path


def test_age_none_when_csv_missing(tmp_path):
    assert universe_age_days(tmp_path / "nope.csv") is None


def test_age_none_when_sidecar_missing(tmp_path):
    csv_path = _write_universe(tmp_path)
    assert universe_age_days(csv_path) is None


def test_age_none_when_sidecar_corrupt(tmp_path):
    csv_path = _write_universe(tmp_path)
    csv_path.with_suffix(".meta.json").write_text("not json")
    assert universe_age_days(csv_path) is None


def test_write_meta_roundtrip(tmp_path):
    csv_path = _write_universe(tmp_path)
    write_universe_meta(csv_path, len(UNIVERSE))
    age = universe_age_days(csv_path)
    assert age is not None and 0 <= age < 0.01
    meta = json.loads(csv_path.with_suffix(".meta.json").read_text())
    assert meta["rows"] == len(UNIVERSE)


def _patch_default(monkeypatch, csv_path, build_called):
    def fake_build(client_kwargs=None, **kwargs):
        build_called.append(True)
        return UNIVERSE
    monkeypatch.setattr(bsu, "DEFAULT_OUT", csv_path)
    monkeypatch.setattr(bsu, "build_stock_universe", fake_build)


def test_get_universe_reuses_fresh_csv(tmp_path, monkeypatch):
    csv_path = _write_universe(tmp_path, age_days=10)
    build_called = []
    _patch_default(monkeypatch, csv_path, build_called)
    df = get_universe(None, max_age_days=60)
    assert not build_called
    assert list(df["symbol"]) == ["AAA", "BBB"]


def test_get_universe_rebuilds_stale_csv(tmp_path, monkeypatch):
    csv_path = _write_universe(tmp_path, age_days=61)
    build_called = []
    _patch_default(monkeypatch, csv_path, build_called)
    get_universe(None, max_age_days=60)
    assert build_called


def test_get_universe_rebuilds_when_age_unknown(tmp_path, monkeypatch):
    csv_path = _write_universe(tmp_path)  # no sidecar
    build_called = []
    _patch_default(monkeypatch, csv_path, build_called)
    get_universe(None, max_age_days=60)
    assert build_called


@pytest.mark.parametrize("max_age", [0, None])
def test_get_universe_always_rebuilds_without_window(tmp_path, monkeypatch, max_age):
    csv_path = _write_universe(tmp_path, age_days=1)
    build_called = []
    _patch_default(monkeypatch, csv_path, build_called)
    get_universe(None, max_age_days=max_age)
    assert build_called


def test_explicit_csv_bypasses_age_check(tmp_path, monkeypatch):
    csv_path = _write_universe(tmp_path)  # no sidecar, would be "stale"
    build_called = []
    _patch_default(monkeypatch, csv_path, build_called)
    df = get_universe(csv_path, max_age_days=60)
    assert not build_called
    assert len(df) == 2
