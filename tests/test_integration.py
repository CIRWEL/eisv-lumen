"""Integration tests with real Lumen data.

Validates the full EISV pipeline end-to-end against the real anima.db:
extraction, assembly, shape classification, baseline evaluation,
expression generation, and HuggingFace format conversion.

Skipped automatically in CI environments where anima.db is unavailable.
Run with: python3 -m pytest tests/test_integration.py -v -s -m integration
"""

import os
import json
import pytest

ANIMA_DB = "/Users/cirwel/.anima/anima.db"
SKIP_REASON = "anima.db not found at expected path"

pytestmark = [
    pytest.mark.skipif(
        not os.path.exists(ANIMA_DB),
        reason=SKIP_REASON,
    ),
    pytest.mark.integration,
]


class TestRealDataExtraction:
    """Verify extraction from real anima.db works."""

    def test_extract_state_history(self):
        from eisv_lumen.extract.lumen_states import extract_state_history

        records = extract_state_history(ANIMA_DB, compute_eisv=True, limit=100)
        assert len(records) == 100
        assert "eisv" in records[0]
        eisv = records[0]["eisv"]
        assert all(0.0 <= eisv[k] <= 1.0 for k in ("E", "I", "S", "V"))

    def test_extract_primitive_history(self):
        from eisv_lumen.extract.lumen_expressions import extract_primitive_history

        records = extract_primitive_history(ANIMA_DB, limit=50)
        assert len(records) == 50
        assert "tokens" in records[0]
        assert isinstance(records[0]["tokens"], list)


class TestFullPipeline:
    """End-to-end pipeline on real data."""

    def test_assemble_and_classify(self):
        from eisv_lumen.extract.lumen_states import extract_state_history
        from eisv_lumen.extract.lumen_expressions import extract_primitive_history
        from eisv_lumen.extract.assembler import assemble_dataset
        from eisv_lumen.shapes.shape_classes import TrajectoryShape

        states = extract_state_history(ANIMA_DB, compute_eisv=True, limit=1000)
        expressions = extract_primitive_history(ANIMA_DB)

        records = assemble_dataset(states, expressions, window_size=20, stride=10)
        assert len(records) > 0

        # Check shape distribution
        shapes = [r["shape"] for r in records]
        unique_shapes = set(shapes)
        assert len(unique_shapes) >= 1
        # All shapes should be valid
        valid = {s.value for s in TrajectoryShape}
        assert unique_shapes.issubset(valid)

        # Print distribution for inspection
        from collections import Counter

        dist = Counter(shapes)
        print(f"\nShape distribution ({len(records)} windows):")
        for shape, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"  {shape}: {count} ({100 * count / len(records):.1f}%)")

    def test_baseline_evaluation_on_real(self):
        from eisv_lumen.extract.lumen_states import extract_state_history
        from eisv_lumen.extract.assembler import assemble_dataset
        from eisv_lumen.eval.baseline import evaluate_baseline, BaselineCondition

        states = extract_state_history(ANIMA_DB, compute_eisv=True, limit=500)
        records = assemble_dataset(states, window_size=20, stride=10)

        for condition in BaselineCondition:
            result = evaluate_baseline(records, condition, n_tokens=3, seed=42)
            print(f"\n{condition.value}: coherence={result['mean_coherence']:.3f}")
            assert 0.0 <= result["mean_coherence"] <= 1.0
            assert result["n_records"] == len(records)

    def test_expression_generator_on_real(self):
        from eisv_lumen.extract.lumen_states import extract_state_history
        from eisv_lumen.extract.assembler import assemble_dataset
        from eisv_lumen.shapes.expression_generator import ExpressionGenerator
        from eisv_lumen.eval.metrics import expression_trajectory_coherence

        states = extract_state_history(ANIMA_DB, compute_eisv=True, limit=500)
        records = assemble_dataset(states, window_size=20, stride=10)

        gen = ExpressionGenerator(seed=42)
        coherences = []
        for rec in records:
            tokens = gen.generate(rec["shape"])
            coh = expression_trajectory_coherence(rec["shape"], tokens)
            coherences.append(coh)

        mean_coh = sum(coherences) / len(coherences) if coherences else 0.0
        print(f"\nExpression generator mean coherence: {mean_coh:.3f}")
        # Should beat random baseline (~4/15 ~ 0.27 for uniform)
        assert mean_coh > 0.3

    def test_hf_format_conversion(self):
        from eisv_lumen.extract.lumen_states import extract_state_history
        from eisv_lumen.extract.assembler import assemble_dataset
        from eisv_lumen.publish.hf_dataset import (
            trajectories_to_hf_format,
            generate_dataset_card,
        )
        from collections import Counter

        states = extract_state_history(ANIMA_DB, compute_eisv=True, limit=500)
        records = assemble_dataset(states, window_size=20, stride=10)

        hf_data = trajectories_to_hf_format(records)
        assert len(hf_data["shape"]) == len(records)

        # Verify JSON fields parse correctly
        parsed = json.loads(hf_data["eisv_states"][0])
        assert isinstance(parsed, list)

        # Generate card
        shape_counts = dict(Counter(hf_data["shape"]))
        card = generate_dataset_card(
            dataset_name="CIRWEL/unitares-eisv-trajectories",
            n_records=len(records),
            shape_counts=shape_counts,
        )
        assert "EISV" in card
        assert str(len(records)) in card
        print(f"\nDataset card generated ({len(card)} chars)")
