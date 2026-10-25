import numpy as np
import pandas as pd
import pytest

from ahr import ahr, pipeline, synthetic, trigger


# ----------------------------------------------------------------------------
# AHR model
# ----------------------------------------------------------------------------


def test_activity_model_recovers_slope_and_ahr():
    rng = np.random.default_rng(0)
    t = np.arange(0, 3600, 5.0)
    act = np.clip(0.1 * np.sin(2 * np.pi * t / 600) + 0.1 + 0.02 * rng.standard_normal(len(t)), 0, None)
    hr = 65 + 50 * act + rng.standard_normal(len(t))
    stress = (t > 2400) & (t < 3000)
    hr[stress] += 10
    calib = t < 1800
    df = pd.DataFrame({"t": t, "activity": act, "mean_hr": hr})
    out, model = ahr.add_ahr_columns(df, calib, lags_s=())
    assert model.coef_[0] == pytest.approx(65, abs=1)
    assert model.coef_[1] == pytest.approx(50, abs=5)
    assert out.loc[stress, "ahr"].mean() == pytest.approx(10, abs=1)
    assert abs(out.loc[~stress, "ahr"].mean()) < 1


def test_lag_terms_use_only_past_activity():
    t = np.arange(0, 100, 5.0)
    a = np.zeros_like(t)
    a[10] = 1.0  # t = 50
    X = ahr.activity_design(t, a, lags_s=(10.0,))
    assert X[10, 1] == 1.0 and X[12, 2] == 1.0
    assert np.nansum(X[:12, 2]) == 0.0  # lagged column never sees the future
    assert np.isnan(X[0, 2])


def test_slopes_are_nonnegative():
    t = np.arange(0, 600, 5.0)
    act = np.linspace(0, 0.1, len(t))
    hr = 70 - 30 * act  # spurious negative relation
    model = ahr.ActivityHRModel(lags_s=()).fit(t, act, hr, np.ones(len(t), bool))
    assert model.coef_[1] == pytest.approx(0.0, abs=1e-6)


def test_ahr_separates_stress_from_walking_on_synthetic_recording():
    rec = synthetic.simulate(seed=1, artifact_gain=0.0)
    cfg = pipeline.FeatureConfig()
    f = pipeline.extract_features(rec.ppg, rec.fs_ppg, rec.acc, rec.fs_acc, causal=True, cfg=cfg)
    lab = pipeline.window_labels(f["t"], rec.labels, rec.label_fs, cfg.win_s)
    # calibrate on baseline plus the first walking bout so the activity slope is identifiable
    calib = (lab["label"] == 1).to_numpy() | ((f["t"] > 1440) & (f["t"] <= 1560)).to_numpy()
    out, model = ahr.add_ahr_columns(f, calib)
    assert sum(model.coef_[1:]) > 10
    pure = (lab["label_purity"] == 1).to_numpy()
    stress = pure & (lab["label"] == 2).to_numpy()
    walking = (out["activity"] > 0.2).to_numpy()
    assert out.loc[stress, "ahr"].mean() > 4
    # walking raises raw HR more than stress, but AHR much less
    assert out.loc[walking, "hr_rel"].mean() > out.loc[stress, "hr_rel"].mean()
    assert out.loc[walking, "ahr"].mean() < out.loc[walking, "hr_rel"].mean() - 5


# ----------------------------------------------------------------------------
# Trigger
# ----------------------------------------------------------------------------


def _frame(score, step=5.0, quality=None, motion=None):
    n = len(score)
    return pd.DataFrame(
        {
            "t": np.arange(1, n + 1) * step,
            "ahr": score,
            "quality_ok": np.ones(n, bool) if quality is None else quality,
            "motion": np.zeros(n, bool) if motion is None else motion,
        }
    )


def test_trigger_requires_sustained_elevation():
    score = np.r_[np.zeros(10), np.full(5, 20.0), np.zeros(10), np.full(10, 20.0)]
    df = _frame(score)
    prompts = trigger.simulate(df, trigger.TriggerConfig(threshold=10, min_duration_s=30, refractory_s=600))
    # first run lasts 20 s (< 30 s) -> no prompt; second run fires 30 s after it starts
    assert prompts.tolist() == [df["t"][25 + 6]]


@pytest.mark.parametrize("refractory_min", [10, 20, 30])
def test_trigger_respects_refractory_period(refractory_min):
    df = _frame(np.full(2 * 3600 // 5, 25.0))  # constantly elevated for 2 h
    cfg = trigger.TriggerConfig(threshold=10, min_duration_s=30, refractory_s=refractory_min * 60)
    prompts = trigger.simulate(df, cfg)
    assert len(prompts) >= 2
    assert np.all(np.diff(prompts) >= refractory_min * 60)
    assert len(prompts) == int((7200 - 35) // (refractory_min * 60)) + 1


def test_trigger_blocks_motion_and_poor_quality():
    score = np.full(40, 25.0)
    motion = np.zeros(40, bool)
    motion[::5] = True  # a motion window every 25 s breaks every run
    cfg = trigger.TriggerConfig(threshold=10, min_duration_s=30, refractory_s=0)
    assert len(trigger.simulate(_frame(score, motion=motion), cfg)) == 0
    quality = ~motion
    assert len(trigger.simulate(_frame(score, quality=quality), cfg)) == 0
    assert len(trigger.simulate(_frame(score, quality=quality), trigger.TriggerConfig(10, 30, 0, require_quality=False))) > 0


def test_trigger_decisions_use_no_future_data():
    rng = np.random.default_rng(1)
    score = 10 + 8 * rng.standard_normal(1000)
    df = _frame(score)
    cfg = trigger.TriggerConfig(threshold=12, min_duration_s=15, refractory_s=300)
    full = trigger.simulate(df, cfg)
    for cut in (200, 500, 800):
        altered = df.copy()
        altered.loc[cut:, "ahr"] = rng.uniform(-50, 50, len(df) - cut)  # rewrite the future
        t_cut = df["t"][cut - 1]
        alt = trigger.simulate(altered, cfg)
        np.testing.assert_array_equal(full[full <= t_cut], alt[alt <= t_cut])
        np.testing.assert_array_equal(full[full <= t_cut], trigger.simulate(df.iloc[:cut], cfg))


def test_trigger_rejects_out_of_order_windows():
    trig = trigger.StreamingTrigger(trigger.TriggerConfig())
    trig.update(10.0, 0.0)
    with pytest.raises(ValueError):
        trig.update(5.0, 0.0)


def test_causal_feature_pipeline_uses_no_future_samples():
    rec = synthetic.simulate([synthetic.Segment(1, 300, 0, 0.0), synthetic.Segment(0, 120, 0, 0.3)], seed=3)
    full = pipeline.extract_features(rec.ppg, rec.fs_ppg, rec.acc, rec.fs_acc, causal=True)
    cut_s = 200.0
    part = pipeline.extract_features(
        rec.ppg[: int(cut_s * rec.fs_ppg)], rec.fs_ppg, rec.acc[: int(cut_s * rec.fs_acc)], rec.fs_acc, causal=True
    )
    cols = ["mean_hr", "rmssd", "activity", "template_corr", "spectral_entropy", "quality_ok"]
    pd.testing.assert_frame_equal(full.loc[: len(part) - 1, cols], part[cols], check_exact=False, atol=1e-9)
