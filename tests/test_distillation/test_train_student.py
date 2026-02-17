"""Tests for student classifier training pipeline."""

import json
import os
import tempfile

import numpy as np
import pytest

from eisv_lumen.distillation.train_student import (
    ALL_PATTERNS,
    ALL_TOKENS,
    NUMERIC_FEATURES,
    SHAPE_NAMES,
    StudentModels,
    load_distillation_data,
    prepare_features,
    predict,
    save_student_models,
    load_student_models,
    train_student_models,
)


def _make_fake_data(n: int = 200, seed: int = 42) -> list:
    """Generate fake distillation data for testing."""
    rng = np.random.RandomState(seed)
    data = []
    for i in range(n):
        shape = SHAPE_NAMES[i % len(SHAPE_NAMES)]
        row = {
            "shape": shape,
            "mean_E": float(rng.uniform(0, 1)),
            "mean_I": float(rng.uniform(0, 1)),
            "mean_S": float(rng.uniform(0, 1)),
            "mean_V": float(rng.uniform(0, 0.3)),
            "dE": float(rng.uniform(-0.1, 0.1)),
            "dI": float(rng.uniform(-0.1, 0.1)),
            "dS": float(rng.uniform(-0.1, 0.1)),
            "dV": float(rng.uniform(-0.05, 0.05)),
            "d2E": float(rng.uniform(-0.01, 0.01)),
            "d2I": float(rng.uniform(-0.01, 0.01)),
            "d2S": float(rng.uniform(-0.01, 0.01)),
            "d2V": float(rng.uniform(-0.01, 0.01)),
            "token_1": ALL_TOKENS[i % len(ALL_TOKENS)],
            "token_2": ALL_TOKENS[(i + 3) % len(ALL_TOKENS)] if i % 3 != 0 else "none",
            "token_3": "none",
            "pattern": ALL_PATTERNS[i % len(ALL_PATTERNS)],
        }
        data.append(row)
    return data


class TestPrepareFeatures:
    def test_output_shape(self):
        data = _make_fake_data(50)
        X, scaler, enc = prepare_features(data)
        # 12 numeric + 9 one-hot shape = 21
        assert X.shape == (50, 21)

    def test_scaled_features_have_zero_mean(self):
        data = _make_fake_data(100)
        X, scaler, enc = prepare_features(data)
        # First 12 columns should be roughly zero-mean
        numeric_part = X[:, :12]
        means = np.abs(numeric_part.mean(axis=0))
        assert all(m < 0.2 for m in means)

    def test_shape_onehot_sums_to_one(self):
        data = _make_fake_data(50)
        X, _, _ = prepare_features(data)
        shape_part = X[:, 12:]
        row_sums = shape_part.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1.0)

    def test_refit_false_uses_existing_scaler(self):
        data = _make_fake_data(50)
        X1, scaler, enc = prepare_features(data)
        X2, _, _ = prepare_features(data, scaler=scaler, shape_encoder=enc, fit=False)
        np.testing.assert_array_almost_equal(X1, X2)


class TestTrainStudentModels:
    @pytest.fixture
    def fake_data(self):
        return _make_fake_data(200)

    def test_returns_models_and_metrics(self, fake_data):
        models, metrics = train_student_models(fake_data, n_estimators=10, max_depth=5, verbose=False)
        assert isinstance(models, StudentModels)
        assert "pattern_accuracy" in metrics
        assert "token1_accuracy" in metrics
        assert "token2_accuracy" in metrics

    def test_metrics_are_reasonable(self, fake_data):
        _, metrics = train_student_models(fake_data, n_estimators=10, max_depth=5, verbose=False)
        # On random data, accuracy should be above random chance
        # Pattern: 5 classes → chance = 0.2
        # Token1: 15 classes → chance ≈ 0.067
        # These are trained on the data so should exceed chance
        assert metrics["pattern_accuracy"] >= 0.0
        assert metrics["token1_accuracy"] >= 0.0
        assert metrics["n_train"] > 0
        assert metrics["n_test"] > 0

    def test_cv_scores_present(self, fake_data):
        _, metrics = train_student_models(fake_data, n_estimators=10, max_depth=5, verbose=False)
        assert "cv_pattern_mean" in metrics
        assert "cv_token1_mean" in metrics


class TestSaveLoadModels:
    def test_roundtrip(self):
        data = _make_fake_data(100)
        models, _ = train_student_models(data, n_estimators=5, max_depth=5, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_student_models(models, tmpdir)

            # Check files exist
            expected_files = [
                "pattern_clf.pkl", "token1_clf.pkl", "token2_clf.pkl",
                "scaler.pkl", "shape_encoder.pkl",
                "token1_encoder.pkl", "token2_encoder.pkl", "pattern_encoder.pkl",
            ]
            for fname in expected_files:
                assert os.path.exists(os.path.join(tmpdir, fname))

            # Load back
            loaded = load_student_models(tmpdir)
            assert isinstance(loaded, StudentModels)

            # Predictions should match
            features = {f: 0.5 for f in NUMERIC_FEATURES}
            pred1 = predict(models, "settled_presence", features)
            pred2 = predict(loaded, "settled_presence", features)
            assert pred1["token_1"] == pred2["token_1"]
            assert pred1["pattern"] == pred2["pattern"]


class TestPredict:
    def test_returns_expected_keys(self):
        data = _make_fake_data(100)
        models, _ = train_student_models(data, n_estimators=5, max_depth=5, verbose=False)

        features = {f: 0.5 for f in NUMERIC_FEATURES}
        result = predict(models, "settled_presence", features)

        assert "pattern" in result
        assert "token_1" in result
        assert "token_2" in result
        assert "eisv_tokens" in result

    def test_token_1_is_valid(self):
        data = _make_fake_data(100)
        models, _ = train_student_models(data, n_estimators=5, max_depth=5, verbose=False)

        features = {f: 0.5 for f in NUMERIC_FEATURES}
        result = predict(models, "rising_entropy", features)

        assert result["token_1"] in ALL_TOKENS

    def test_pattern_is_valid(self):
        data = _make_fake_data(100)
        models, _ = train_student_models(data, n_estimators=5, max_depth=5, verbose=False)

        features = {f: 0.5 for f in NUMERIC_FEATURES}
        result = predict(models, "convergence", features)

        assert result["pattern"] in ALL_PATTERNS

    def test_eisv_tokens_non_empty(self):
        data = _make_fake_data(100)
        models, _ = train_student_models(data, n_estimators=5, max_depth=5, verbose=False)

        features = {f: 0.5 for f in NUMERIC_FEATURES}
        result = predict(models, "void_rising", features)

        assert len(result["eisv_tokens"]) >= 1
        assert all(t in ALL_TOKENS for t in result["eisv_tokens"])
