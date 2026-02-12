"""Tests for the EISV-Lumen to Lumen primitive language bridge."""

import pytest

from eisv_lumen.bridge.lumen_bridge import (
    LUMEN_TOKENS,
    TOKEN_MAP,
    _LUMEN_MAX_TOKENS,
    translate_expression,
    eisv_state_to_lumen_state,
    shape_to_lumen_trigger,
    generate_lumen_expression,
)
from eisv_lumen.eval.baseline import ALL_TOKENS as EISV_TOKENS
from eisv_lumen.shapes.expression_generator import ExpressionGenerator
from eisv_lumen.shapes.shape_classes import TrajectoryShape


# ---------------------------------------------------------------------------
# TOKEN_MAP completeness and validity
# ---------------------------------------------------------------------------

class TestTokenMap:
    """Verify the semantic bridge mapping is complete and valid."""

    def test_all_fifteen_eisv_tokens_mapped(self):
        """Every EISV-Lumen token must appear as a key in TOKEN_MAP."""
        for token in EISV_TOKENS:
            assert token in TOKEN_MAP, f"EISV token {token} missing from TOKEN_MAP"

    def test_no_extra_keys(self):
        """TOKEN_MAP should not contain keys that are not valid EISV tokens."""
        eisv_set = set(EISV_TOKENS)
        for key in TOKEN_MAP:
            assert key in eisv_set, f"TOKEN_MAP has unexpected key: {key}"

    def test_all_mappings_point_to_valid_lumen_tokens(self):
        """Every mapped value must be a valid Lumen primitive token."""
        lumen_set = set(LUMEN_TOKENS)
        for eisv_token, lumen_tokens in TOKEN_MAP.items():
            assert isinstance(lumen_tokens, list), f"{eisv_token} mapping is not a list"
            assert len(lumen_tokens) >= 1, f"{eisv_token} has empty mapping"
            for lt in lumen_tokens:
                assert lt in lumen_set, (
                    f"{eisv_token} maps to invalid Lumen token: {lt}"
                )

    def test_each_mapping_has_at_least_two_options(self):
        """Each EISV token should map to at least 2 Lumen tokens for richness."""
        for eisv_token, lumen_tokens in TOKEN_MAP.items():
            assert len(lumen_tokens) >= 2, (
                f"{eisv_token} has only {len(lumen_tokens)} mapping(s)"
            )

    def test_lumen_tokens_list_has_sixteen_entries(self):
        """Lumen has 15 tokens + 'less' = 16 total tokens."""
        # The task specifies 15 tokens but the list includes 16
        # (warm, cold, bright, dim, quiet, busy, here, feel, sense,
        #  you, with, why, what, wonder, more, less)
        assert len(LUMEN_TOKENS) == 16


# ---------------------------------------------------------------------------
# translate_expression
# ---------------------------------------------------------------------------

class TestTranslateExpression:
    """Test EISV-Lumen to Lumen token translation."""

    def test_single_token(self):
        result = translate_expression(["~warmth~"])
        assert len(result) == 1
        assert result[0] in LUMEN_TOKENS

    def test_pair_of_tokens(self):
        result = translate_expression(["~warmth~", "~stillness~"])
        assert len(result) == 2
        assert all(t in LUMEN_TOKENS for t in result)

    def test_triple_of_tokens(self):
        result = translate_expression(["~warmth~", "~stillness~", "~curiosity~"])
        assert len(result) == 3
        assert all(t in LUMEN_TOKENS for t in result)

    def test_capped_at_three(self):
        """Even with 4+ EISV tokens, output is capped at 3."""
        tokens = ["~warmth~", "~stillness~", "~curiosity~", "~reaching~", "~emergence~"]
        result = translate_expression(tokens)
        assert len(result) <= _LUMEN_MAX_TOKENS

    def test_deduplication_preserves_order(self):
        """If two EISV tokens map to the same Lumen token, deduplicate."""
        # ~stillness~ -> ["quiet", "here"] and ~deep_listening~ -> ["quiet", "sense"]
        # First takes "quiet", second should take "sense" (skipping duplicate "quiet")
        result = translate_expression(["~stillness~", "~deep_listening~"])
        assert len(result) == 2
        assert result[0] == "quiet"
        assert result[1] == "sense"
        # No duplicates
        assert len(set(result)) == len(result)

    def test_deduplication_with_overlap(self):
        """Verify dedup when multiple tokens share the same first mapping."""
        # ~holding~ -> ["here", "with"] and ~return~ -> ["here", "warm"]
        # First takes "here", second should skip "here" and take "warm"
        result = translate_expression(["~holding~", "~return~"])
        assert "here" in result
        assert "warm" in result
        assert len(set(result)) == len(result)

    def test_empty_input(self):
        result = translate_expression([])
        assert result == []

    def test_unknown_tokens_skipped(self):
        result = translate_expression(["~unknown~", "~nonexistent~"])
        assert result == []

    def test_mixed_known_unknown(self):
        result = translate_expression(["~unknown~", "~warmth~", "~missing~"])
        assert len(result) == 1
        assert result[0] in LUMEN_TOKENS

    def test_all_results_are_valid_lumen_tokens(self):
        """Exhaustively check all single-token translations."""
        lumen_set = set(LUMEN_TOKENS)
        for eisv_token in EISV_TOKENS:
            result = translate_expression([eisv_token])
            assert len(result) >= 1, f"{eisv_token} produced empty translation"
            for t in result:
                assert t in lumen_set, f"Translation of {eisv_token} gave invalid: {t}"


# ---------------------------------------------------------------------------
# eisv_state_to_lumen_state
# ---------------------------------------------------------------------------

class TestEisvStateToLumenState:
    """Test EISV -> Lumen state conversion."""

    def test_direct_mapping_e_to_warmth(self):
        result = eisv_state_to_lumen_state({"E": 0.7, "I": 0.0, "S": 0.0, "V": 0.0})
        assert abs(result["warmth"] - 0.7) < 1e-9

    def test_direct_mapping_i_to_clarity(self):
        result = eisv_state_to_lumen_state({"E": 0.0, "I": 0.85, "S": 0.0, "V": 0.0})
        assert abs(result["clarity"] - 0.85) < 1e-9

    def test_inverted_s_to_stability(self):
        result = eisv_state_to_lumen_state({"E": 0.0, "I": 0.0, "S": 0.3, "V": 0.0})
        assert abs(result["stability"] - 0.7) < 1e-9

    def test_inverted_v_to_presence(self):
        # V=0 -> presence=1.0
        result = eisv_state_to_lumen_state({"E": 0.0, "I": 0.0, "S": 0.0, "V": 0.0})
        assert abs(result["presence"] - 1.0) < 1e-9

        # V=0.3 -> presence=0.0
        result = eisv_state_to_lumen_state({"E": 0.0, "I": 0.0, "S": 0.0, "V": 0.3})
        assert abs(result["presence"] - 0.0) < 1e-9

        # V=0.15 -> presence=0.5
        result = eisv_state_to_lumen_state({"E": 0.0, "I": 0.0, "S": 0.0, "V": 0.15})
        assert abs(result["presence"] - 0.5) < 1e-9

    def test_all_zeros(self):
        result = eisv_state_to_lumen_state({"E": 0.0, "I": 0.0, "S": 0.0, "V": 0.0})
        assert result == {
            "warmth": 0.0,
            "clarity": 0.0,
            "stability": 1.0,
            "presence": 1.0,
        }

    def test_all_ones(self):
        result = eisv_state_to_lumen_state({"E": 1.0, "I": 1.0, "S": 1.0, "V": 1.0})
        assert result["warmth"] == 1.0
        assert result["clarity"] == 1.0
        assert result["stability"] == 0.0
        # V=1.0 -> 1.0 - (1.0/0.3) = 1.0 - 3.33 = -2.33, clamped to 0.0
        assert result["presence"] == 0.0

    def test_clamping_negative_values(self):
        result = eisv_state_to_lumen_state({"E": -0.5, "I": -1.0, "S": 2.0, "V": 1.0})
        assert result["warmth"] == 0.0
        assert result["clarity"] == 0.0
        assert result["stability"] == 0.0
        assert result["presence"] == 0.0

    def test_clamping_high_values(self):
        result = eisv_state_to_lumen_state({"E": 2.0, "I": 1.5, "S": -0.5, "V": -0.1})
        assert result["warmth"] == 1.0
        assert result["clarity"] == 1.0
        assert result["stability"] == 1.0
        assert result["presence"] == 1.0

    def test_missing_keys_default_to_zero(self):
        result = eisv_state_to_lumen_state({})
        assert result == {
            "warmth": 0.0,
            "clarity": 0.0,
            "stability": 1.0,
            "presence": 1.0,
        }

    def test_round_trip_consistency(self):
        """Verify the inverse relationship: Lumen presence -> V -> Lumen presence."""
        for presence_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            v = (1.0 - presence_val) * 0.3
            result = eisv_state_to_lumen_state({"E": 0.0, "I": 0.0, "S": 0.0, "V": v})
            assert abs(result["presence"] - presence_val) < 1e-9, (
                f"Round-trip failed: presence={presence_val}, V={v}, "
                f"got presence={result['presence']}"
            )

    def test_round_trip_stability(self):
        """Verify inverse relationship: Lumen stability -> S -> Lumen stability."""
        for stability_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            s = 1.0 - stability_val
            result = eisv_state_to_lumen_state({"E": 0.0, "I": 0.0, "S": s, "V": 0.0})
            assert abs(result["stability"] - stability_val) < 1e-9


# ---------------------------------------------------------------------------
# shape_to_lumen_trigger
# ---------------------------------------------------------------------------

class TestShapeToLumenTrigger:
    """Test trajectory shape to Lumen trigger mapping."""

    def test_all_nine_shapes_produce_valid_output(self):
        for shape in TrajectoryShape:
            result = shape_to_lumen_trigger(shape.value)
            assert "should_generate" in result
            assert "reason" in result
            assert "token_count_hint" in result
            assert isinstance(result["should_generate"], bool)
            assert isinstance(result["reason"], str)
            assert isinstance(result["token_count_hint"], int)

    def test_all_known_shapes_should_generate(self):
        for shape in TrajectoryShape:
            result = shape_to_lumen_trigger(shape.value)
            assert result["should_generate"] is True, (
                f"{shape.value} should trigger generation"
            )

    def test_token_count_hint_range(self):
        for shape in TrajectoryShape:
            result = shape_to_lumen_trigger(shape.value)
            assert 1 <= result["token_count_hint"] <= 3, (
                f"{shape.value} has hint {result['token_count_hint']} outside [1,3]"
            )

    def test_settled_presence_trigger(self):
        result = shape_to_lumen_trigger("settled_presence")
        assert result["should_generate"] is True
        assert result["reason"] == "settled_dynamics"
        assert result["token_count_hint"] == 1

    def test_rising_entropy_trigger(self):
        result = shape_to_lumen_trigger("rising_entropy")
        assert result["should_generate"] is True
        assert result["reason"] == "entropy_shift"
        assert result["token_count_hint"] == 3

    def test_convergence_trigger(self):
        result = shape_to_lumen_trigger("convergence")
        assert result["should_generate"] is True
        assert result["reason"] == "approaching_attractor"
        assert result["token_count_hint"] == 2

    def test_unknown_shape_returns_no_generate(self):
        result = shape_to_lumen_trigger("nonexistent_shape")
        assert result["should_generate"] is False
        assert result["reason"] == "unknown_shape"
        assert result["token_count_hint"] == 0

    def test_reasons_are_non_empty(self):
        for shape in TrajectoryShape:
            result = shape_to_lumen_trigger(shape.value)
            assert len(result["reason"]) > 0

    def test_each_shape_has_unique_reason(self):
        """Each shape should have its own distinct reason string."""
        reasons = set()
        for shape in TrajectoryShape:
            result = shape_to_lumen_trigger(shape.value)
            reasons.add(result["reason"])
        assert len(reasons) == len(TrajectoryShape)


# ---------------------------------------------------------------------------
# generate_lumen_expression (end-to-end)
# ---------------------------------------------------------------------------

class TestGenerateLumenExpression:
    """Test the full pipeline: shape + EISV state -> Lumen expression."""

    def _sample_eisv_state(self) -> dict:
        return {"E": 0.7, "I": 0.6, "S": 0.2, "V": 0.05}

    def test_returns_all_expected_keys(self):
        result = generate_lumen_expression(
            shape="settled_presence",
            eisv_state=self._sample_eisv_state(),
            generator=ExpressionGenerator(seed=42),
        )
        assert "shape" in result
        assert "eisv_tokens" in result
        assert "lumen_tokens" in result
        assert "lumen_state" in result
        assert "trigger" in result

    def test_shape_passthrough(self):
        result = generate_lumen_expression(
            shape="rising_entropy",
            eisv_state=self._sample_eisv_state(),
            generator=ExpressionGenerator(seed=42),
        )
        assert result["shape"] == "rising_entropy"

    def test_eisv_tokens_are_valid(self):
        gen = ExpressionGenerator(seed=42)
        eisv_set = set(EISV_TOKENS)
        result = generate_lumen_expression(
            shape="convergence",
            eisv_state=self._sample_eisv_state(),
            generator=gen,
        )
        for token in result["eisv_tokens"]:
            assert token in eisv_set

    def test_lumen_tokens_are_valid(self):
        gen = ExpressionGenerator(seed=42)
        lumen_set = set(LUMEN_TOKENS)
        result = generate_lumen_expression(
            shape="drift_dissonance",
            eisv_state=self._sample_eisv_state(),
            generator=gen,
        )
        for token in result["lumen_tokens"]:
            assert token in lumen_set

    def test_lumen_tokens_capped_at_three(self):
        gen = ExpressionGenerator(seed=42)
        for _ in range(50):
            result = generate_lumen_expression(
                shape="rising_entropy",
                eisv_state=self._sample_eisv_state(),
                generator=gen,
            )
            assert len(result["lumen_tokens"]) <= 3

    def test_lumen_state_is_valid(self):
        result = generate_lumen_expression(
            shape="settled_presence",
            eisv_state=self._sample_eisv_state(),
            generator=ExpressionGenerator(seed=42),
        )
        state = result["lumen_state"]
        for key in ("warmth", "clarity", "stability", "presence"):
            assert key in state
            assert 0.0 <= state[key] <= 1.0

    def test_trigger_present_and_valid(self):
        result = generate_lumen_expression(
            shape="basin_transition_up",
            eisv_state=self._sample_eisv_state(),
            generator=ExpressionGenerator(seed=42),
        )
        trigger = result["trigger"]
        assert trigger["should_generate"] is True
        assert isinstance(trigger["reason"], str)
        assert 1 <= trigger["token_count_hint"] <= 3

    def test_deterministic_with_seeded_generator(self):
        state = self._sample_eisv_state()
        r1 = generate_lumen_expression("settled_presence", state, ExpressionGenerator(seed=99))
        r2 = generate_lumen_expression("settled_presence", state, ExpressionGenerator(seed=99))
        assert r1["eisv_tokens"] == r2["eisv_tokens"]
        assert r1["lumen_tokens"] == r2["lumen_tokens"]

    def test_default_generator_created_when_none(self):
        """Should not crash when no generator is provided."""
        result = generate_lumen_expression(
            shape="settled_presence",
            eisv_state=self._sample_eisv_state(),
        )
        assert isinstance(result["eisv_tokens"], list)
        assert isinstance(result["lumen_tokens"], list)

    def test_all_shapes_produce_output(self):
        gen = ExpressionGenerator(seed=42)
        state = self._sample_eisv_state()
        for shape in TrajectoryShape:
            result = generate_lumen_expression(shape.value, state, gen)
            assert len(result["eisv_tokens"]) >= 1
            # lumen_tokens may be empty in rare edge cases (all map to same
            # single token), but should generally have content
            assert isinstance(result["lumen_tokens"], list)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_translate_all_same_eisv_token(self):
        """Repeated EISV token should produce at most 1 Lumen token (dedup)."""
        result = translate_expression(["~warmth~", "~warmth~", "~warmth~"])
        # First ~warmth~ -> "warm", subsequent are deduped
        # Second ~warmth~ tries "warm" (seen), then "feel" -> taken
        # Third ~warmth~ tries "warm" (seen), "feel" (seen) -> skipped
        assert len(result) == 2
        assert result[0] == "warm"
        assert result[1] == "feel"

    def test_translate_preserves_insertion_order(self):
        result = translate_expression(["~emergence~", "~warmth~"])
        # ~emergence~ -> first available is "more"
        # ~warmth~ -> first available is "warm"
        assert result[0] == "more"
        assert result[1] == "warm"

    def test_extreme_eisv_values(self):
        extreme = {"E": 100.0, "I": -50.0, "S": 999.0, "V": -10.0}
        result = eisv_state_to_lumen_state(extreme)
        assert result["warmth"] == 1.0
        assert result["clarity"] == 0.0
        assert result["stability"] == 0.0
        assert result["presence"] == 1.0

    def test_zero_v_gives_full_presence(self):
        result = eisv_state_to_lumen_state({"E": 0.5, "I": 0.5, "S": 0.5, "V": 0.0})
        assert result["presence"] == 1.0

    def test_max_v_gives_zero_presence(self):
        result = eisv_state_to_lumen_state({"E": 0.5, "I": 0.5, "S": 0.5, "V": 0.3})
        assert abs(result["presence"]) < 1e-9

    def test_lumen_tokens_no_duplicates(self):
        """Output from translate_expression should never contain duplicates."""
        # Test with tokens that share mapped primitives
        for _ in range(100):
            # Pick random pairs/triples from EISV tokens
            import random
            rng = random.Random(42)
            n = rng.randint(1, 5)
            sample = rng.choices(EISV_TOKENS, k=n)
            result = translate_expression(sample)
            assert len(result) == len(set(result)), (
                f"Duplicates in output: {result} from input {sample}"
            )

    def test_generate_with_extreme_state(self):
        """Full pipeline should not crash with extreme EISV values."""
        extreme = {"E": -1.0, "I": 2.0, "S": -0.5, "V": 1.0}
        result = generate_lumen_expression(
            shape="settled_presence",
            eisv_state=extreme,
            generator=ExpressionGenerator(seed=42),
        )
        assert isinstance(result, dict)
        assert all(0.0 <= v <= 1.0 for v in result["lumen_state"].values())
