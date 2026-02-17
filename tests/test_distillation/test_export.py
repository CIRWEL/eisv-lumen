"""Tests for student model JSON export and standalone inference."""

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
    predict as sklearn_predict,
    train_student_models,
)
from eisv_lumen.distillation.export_student import (
    export_student_to_json,
    generate_inference_module,
    verify_export,
)


def _make_fake_data(n=200, seed=42):
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


@pytest.fixture
def trained_models_and_dir():
    """Train models and export to temp directory."""
    data = _make_fake_data(200)
    models, _ = train_student_models(data, n_estimators=10, max_depth=5, verbose=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = os.path.join(tmpdir, "exported")
        export_student_to_json(models, export_dir)
        generate_inference_module(export_dir)
        yield models, export_dir


class TestExportToJson:
    def test_json_files_created(self, trained_models_and_dir):
        _, export_dir = trained_models_and_dir
        expected = [
            "pattern_forest.json", "token1_forest.json", "token2_forest.json",
            "scaler.json", "mappings.json", "student_inference.py",
        ]
        for fname in expected:
            assert os.path.exists(os.path.join(export_dir, fname)), f"Missing {fname}"

    def test_forest_json_is_list_of_trees(self, trained_models_and_dir):
        _, export_dir = trained_models_and_dir
        with open(os.path.join(export_dir, "pattern_forest.json")) as f:
            forest = json.load(f)
        assert isinstance(forest, list)
        assert len(forest) == 10  # n_estimators=10

    def test_tree_has_expected_structure(self, trained_models_and_dir):
        _, export_dir = trained_models_and_dir
        with open(os.path.join(export_dir, "token1_forest.json")) as f:
            forest = json.load(f)
        tree = forest[0]
        # Root should be a split node or leaf
        if tree.get("leaf"):
            assert "probs" in tree
            assert isinstance(tree["probs"], list)
        else:
            assert "feature" in tree
            assert "threshold" in tree
            assert "left" in tree
            assert "right" in tree

    def test_scaler_has_mean_and_scale(self, trained_models_and_dir):
        _, export_dir = trained_models_and_dir
        with open(os.path.join(export_dir, "scaler.json")) as f:
            scaler = json.load(f)
        assert "mean" in scaler
        assert "scale" in scaler
        assert len(scaler["mean"]) == 12  # 12 numeric features

    def test_mappings_complete(self, trained_models_and_dir):
        _, export_dir = trained_models_and_dir
        with open(os.path.join(export_dir, "mappings.json")) as f:
            mappings = json.load(f)
        assert len(mappings["shapes"]) == 9
        assert len(mappings["patterns"]) == 5
        assert len(mappings["tokens"]) == 15
        assert len(mappings["tokens_with_none"]) == 16


class TestStandaloneInference:
    def test_inference_module_loads(self, trained_models_and_dir):
        _, export_dir = trained_models_and_dir
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "student_inference",
            os.path.join(export_dir, "student_inference.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        student = mod.StudentInference(export_dir)
        assert student is not None

    def test_predict_returns_valid_output(self, trained_models_and_dir):
        _, export_dir = trained_models_and_dir
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "student_inference",
            os.path.join(export_dir, "student_inference.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        student = mod.StudentInference(export_dir)

        features = {f: 0.5 for f in NUMERIC_FEATURES}
        result = student.predict("settled_presence", features)

        assert result["pattern"] in ALL_PATTERNS
        assert result["token_1"] in ALL_TOKENS
        assert len(result["eisv_tokens"]) >= 1

    def test_export_matches_sklearn(self, trained_models_and_dir):
        models, export_dir = trained_models_and_dir
        ok = verify_export(models, export_dir, n_tests=50)
        assert ok, "JSON export predictions differ from sklearn"
