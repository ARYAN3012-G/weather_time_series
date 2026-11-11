"""
Pipeline integrity tests — run before any training:   python -m pytest tests -q

They check the scientific guarantees of the v3 pipeline:
  * duplicate weather site removed
  * SPEI calibration uses ONLY the reference (train) period and is frozen
  * inputs never contain the target day or later (genuine future forecasting)
  * historical SPEI in the input comes from input days only
  * purged split: no day shared between train / val / test samples
  * spatial folds are disjoint
  * baselines use only information available at forecast time
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from dataprep.spei import SPEICalibrator, accumulate, water_balance  # noqa: E402
from features.sequence_builder import build_sequences_all_fields, build_sequences_for_field, load_prepared  # noqa: E402
from features.splits import make_masks, purged_periods, spatial_folds  # noqa: E402
from models.baselines import climatology, markov, persistence  # noqa: E402

HAVE_PREPARED = all((config.PREPARED_DIR / f"{f}.csv").exists() for f in config.FIELDS)
needs_data = pytest.mark.skipif(not HAVE_PREPARED, reason="run src/dataprep/prepare_data.py first")


@pytest.fixture(scope="module")
def frames():
    return load_prepared()


# ---------------------------------------------------------------- sites
def test_duplicate_removed_from_config():
    assert "lepakshi" not in config.FIELDS and "kurnool_pulliah" not in config.FIELDS
    assert len(config.FIELDS) == 6
    assert set(config.FIELD_TO_CLUSTER) == set(config.FIELDS)


@needs_data
def test_no_duplicate_prepared_files():
    stems = {p.stem for p in config.PREPARED_DIR.glob("*.csv")}
    assert stems == set(config.FIELDS)


@needs_data
def test_no_identical_weather_between_prepared_sites(frames):
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A = frames[a].set_index(config.DATE_COLUMN)["Precipitation_Flux"]
            B = frames[b].set_index(config.DATE_COLUMN)["Precipitation_Flux"]
            idx = A.index.intersection(B.index)
            assert (A[idx] == B[idx]).mean() < 0.99, f"{a} and {b} look duplicated"


# ---------------------------------------------------------------- SPEI calibration
def _synthetic(seed=0, years=(1979, 2026)):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
    doy = dates.day_of_year.to_numpy()
    p = rng.gamma(0.6, 4 + 3 * np.sin(2 * np.pi * doy / 365), len(dates))
    pet = 4 + 1.5 * np.cos(2 * np.pi * doy / 365) + rng.normal(0, 0.3, len(dates))
    return pd.DataFrame({config.DATE_COLUMN: dates, "Precipitation_Flux": p,
                         "ReferenceET_PenmanMonteith_FAO56": pet})


def test_spei_calibration_ignores_future_years():
    df = _synthetic()
    d = accumulate(water_balance(df), config.SPEI_SCALE_DAYS)
    cal1 = SPEICalibrator(30, 1979, 2011).fit(df[config.DATE_COLUMN], d)

    df2 = df.copy()
    future = df2[config.DATE_COLUMN].dt.year > 2011
    df2.loc[future, "Precipitation_Flux"] *= 5          # drastically change val/test years
    d2 = accumulate(water_balance(df2), config.SPEI_SCALE_DAYS)
    cal2 = SPEICalibrator(30, 1979, 2011).fit(df2[config.DATE_COLUMN], d2)

    assert cal1.to_dict()["params"] == cal2.to_dict()["params"], "calibration used post-2011 data"
    s1 = cal1.transform(df[config.DATE_COLUMN], d)
    s2 = cal2.transform(df2[config.DATE_COLUMN], d2)
    ref = (df[config.DATE_COLUMN].dt.year <= 2010).to_numpy()   # avoid accumulation spill-over at boundary
    np.testing.assert_allclose(s1[ref], s2[ref])


def test_spei_reference_period_is_standardised():
    df = _synthetic(1)
    d = accumulate(water_balance(df), 30)
    cal = SPEICalibrator(30, 1979, 2011).fit(df[config.DATE_COLUMN], d)
    s = cal.transform(df[config.DATE_COLUMN], d)
    ref = (df[config.DATE_COLUMN].dt.year <= 2011) & s.notna()
    assert abs(s[ref].mean()) < 0.1 and abs(s[ref].std() - 1) < 0.1


def test_spei_calibrator_roundtrip():
    df = _synthetic(2)
    d = accumulate(water_balance(df), 30)
    cal = SPEICalibrator(30, 1979, 2011).fit(df[config.DATE_COLUMN], d)
    cal2 = SPEICalibrator.from_dict(cal.to_dict())
    np.testing.assert_allclose(cal.transform(df[config.DATE_COLUMN], d), cal2.transform(df[config.DATE_COLUMN], d))


@needs_data
def test_prepared_spei_matches_frozen_calibration(frames):
    import json
    calib = json.loads(config.SPEI_CALIBRATION_FILE.read_text())
    for name, df in frames.items():
        cal = SPEICalibrator.from_dict(calib["sites"][name])
        s = cal.transform(df[config.DATE_COLUMN], df["WB30"])
        np.testing.assert_allclose(s.to_numpy(), df["SPEI"].to_numpy(), atol=1e-6)
        np.testing.assert_array_equal(config.classify_spei(df["SPEI"]), df["label"])


# ---------------------------------------------------------------- sequences / leakage
@needs_data
@pytest.mark.parametrize("h", config.HORIZONS)
def test_inputs_strictly_before_target(frames, h):
    df = frames[config.FIELDS[0]]
    X, y, meta = build_sequences_for_field(df, config.FEATURE_SETS["weather_spei"],
                                           config.SEQUENCE_LENGTH_DAYS, h)
    assert ((meta.target_date - meta.last_input_date).dt.days == h).all()
    assert ((meta.last_input_date - meta.input_start).dt.days == config.SEQUENCE_LENGTH_DAYS - 1).all()

    by_date = df.set_index(config.DATE_COLUMN)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(meta), 50, replace=False):
        days = pd.date_range(meta.input_start[i], meta.last_input_date[i], freq="D")
        expected = by_date.loc[days, config.FEATURE_SETS["weather_spei"]].to_numpy(np.float32)
        np.testing.assert_allclose(X[i], expected, rtol=1e-6)
        assert y[i] == by_date.loc[meta.target_date[i], "label"]
        # historical SPEI channel ends at day t, never at the target day
        assert X[i, -1, -1] == pytest.approx(by_date.loc[meta.last_input_date[i], "SPEI"], rel=1e-5)


@needs_data
@pytest.mark.parametrize("h", config.HORIZONS)
def test_doy_features_use_input_dates_only(frames, h):
    """Day-of-year channels must encode each INPUT day's date, never the target date."""
    df = frames[config.FIELDS[0]]
    cols = config.FEATURE_SETS["weather_spei_doy"]
    X, _, meta = build_sequences_for_field(df, cols, config.SEQUENCE_LENGTH_DAYS, h)
    i_sin, i_cos = cols.index("DOY_sin"), cols.index("DOY_cos")

    def enc(dates):
        a = 2 * np.pi * (pd.DatetimeIndex(dates).day_of_year.to_numpy() - 1) / 365.25
        return np.sin(a), np.cos(a)

    rng = np.random.default_rng(1)
    for i in rng.choice(len(meta), 50, replace=False):
        days = pd.date_range(meta.input_start[i], meta.last_input_date[i], freq="D")
        s, c = enc(days)
        np.testing.assert_allclose(X[i, :, i_sin], s, atol=1e-6)
        np.testing.assert_allclose(X[i, :, i_cos], c, atol=1e-6)
        # the target date's encoding must not appear as the last input day
        ts, tc = enc([meta.target_date[i]])
        assert not (np.isclose(X[i, -1, i_sin], ts[0], atol=1e-6) and np.isclose(X[i, -1, i_cos], tc[0], atol=1e-6))
    # the other channels are unchanged by adding the date features
    X_base, _, _ = build_sequences_for_field(df, config.FEATURE_SETS["weather_spei"], config.SEQUENCE_LENGTH_DAYS, h)
    np.testing.assert_array_equal(X[:, :, :len(cols) - 2], X_base)


def test_default_feature_sets_fixed_before_rerun():
    assert config.DEFAULT_FEATURE_SETS == ["weather", "weather_spei", "spei_only", "weather_spei_doy",
                                           "weather_spei_doy_wb"]
    for fs in ("weather", "weather_spei", "spei_only"):
        assert not set(config.DOY_FEATURES) & set(config.FEATURE_SETS[fs])


@needs_data
def test_horizon_zero_rejected(frames):
    with pytest.raises(ValueError):
        build_sequences_for_field(frames[config.FIELDS[0]], config.WEATHER_FEATURES, 30, 0)


# ---------------------------------------------------------------- splits
@needs_data
@pytest.mark.parametrize("h", config.HORIZONS)
def test_purged_split_has_no_shared_days(frames, h):
    df = frames[config.FIELDS[0]]
    _, _, meta = build_sequences_for_field(df, config.WEATHER_FEATURES, config.SEQUENCE_LENGTH_DAYS, h)
    per = purged_periods(meta)
    tr, va, te = meta[per == "train"], meta[per == "val"], meta[per == "test"]
    assert tr.target_date.max() < va.input_start.min()
    assert va.target_date.max() < te.input_start.min()
    assert tr.target_date.dt.year.max() <= config.TRAIN_END_YEAR
    assert te.input_start.dt.year.min() > config.VAL_END_YEAR
    assert (per == "purged").sum() > 0


@pytest.mark.parametrize("mode", ["loco", "loso"])
def test_spatial_folds_disjoint_and_complete(mode):
    folds = spatial_folds(mode)
    covered = set()
    for _, tr, te in folds:
        assert not set(tr) & set(te)
        assert set(tr) | set(te) == set(config.FIELDS)
        covered |= set(te)
    assert covered == set(config.FIELDS)


def test_masks_respect_spatial_holdout():
    meta = pd.DataFrame({
        "field": ["a", "a", "b", "b"],
        "input_start": pd.to_datetime(["2000-01-01", "2020-01-01", "2000-01-01", "2020-01-01"]),
        "target_date": pd.to_datetime(["2000-03-01", "2020-03-01", "2000-03-01", "2020-03-01"]),
    })
    m = make_masks(meta, ["a"], ["b"])
    assert m["train"].tolist() == [True, False, False, False]
    assert m["test"].tolist() == [False, False, False, True]


# ---------------------------------------------------------------- baselines
def test_baselines_use_only_past_information():
    meta_tr = pd.DataFrame({"label_t": [0, 0, 1, 2, 2],
                            "target_date": pd.to_datetime(["2000-01-05"] * 5)})
    y_tr = np.array([0, 1, 1, 2, 2])
    meta_te = pd.DataFrame({"label_t": [2, 0], "target_date": pd.to_datetime(["2020-01-05", "2020-07-01"])})
    p = persistence(meta_te)
    assert p.argmax(1).tolist() == [2, 0]
    mk = markov(meta_tr, y_tr, meta_te)
    assert np.allclose(mk.sum(1), 1) and mk[0].argmax() == 2
    cl = climatology(meta_tr, y_tr, meta_te)
    assert np.allclose(cl.sum(1), 1)


@needs_data
def test_wb_ensemble_never_sees_the_days_it_forecasts(frames):
    """Changing the weather on the forecast days (t, t+h] must not change the water-balance
    ensemble for samples outside the calibration years (VAL/TEST). Training samples share the
    frozen 1979-2011 SPEI calibration that also defines their labels and SPEI inputs; their own
    year is additionally left out of the ensemble members."""
    from models.baselines import wb_ensemble
    f = config.FIELDS[0]
    h = 15
    _, _, meta = build_sequences_all_fields({f: frames[f]}, "spei_only", h)
    pick = meta.iloc[[len(meta) - 3000, len(meta) - 500]].reset_index(drop=True)   # two TEST-period samples
    assert (pd.to_datetime(pick["last_input_date"]).dt.year > config.SPEI_CALIB_END).all()
    p1 = wb_ensemble(pick, {f: frames[f]})
    for i in range(len(pick)):
        t = pd.to_datetime(pick["last_input_date"].iloc[i])
        g = frames[f].copy()
        m = (g[config.DATE_COLUMN] > t) & (g[config.DATE_COLUMN] <= t + pd.Timedelta(days=h))
        g.loc[m, "Precipitation_Flux"] = g.loc[m, "Precipitation_Flux"] + 50.0
        p2 = wb_ensemble(pick.iloc[[i]].reset_index(drop=True), {f: g})
        assert np.allclose(p1[i], p2[0])


@needs_data
def test_hybrid_feature_set_adds_three_constant_channels(frames):
    f = config.FIELDS[0]
    X, y, meta = build_sequences_all_fields({f: frames[f]}, "weather_spei_doy", 15)
    Xw, yw, metaw = build_sequences_all_fields({f: frames[f]}, "weather_spei_doy_wb", 15)
    assert Xw.shape == X.shape[:2] + (X.shape[2] + 3,)
    assert np.array_equal(y, yw) and np.allclose(Xw[:, :, :-3], X)
    assert np.allclose(Xw[:, 0, -3:], Xw[:, -1, -3:]) and np.allclose(Xw[:, 0, -3:].sum(1), 1.0)
