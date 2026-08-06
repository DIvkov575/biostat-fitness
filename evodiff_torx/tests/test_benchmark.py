"""Tests for the multi-family benchmark driver.

These are deliberately fast: the table and summary formatters are fed synthetic
result dicts, and `run_benchmark` is exercised with `train_and_evaluate` patched
out. Nothing here trains a model, so the suite does NOT cover the numbers in the
README -- the real end-to-end check is `python evodiff_torx/benchmark.py`, which
takes a couple of minutes.

The synthetic dicts carry the full key set `train_and_evaluate` returns, so they
stay honest stand-ins; `test_train.py::test_result_keys_are_stable` is what pins
that contract on the producing side.
"""

from __future__ import annotations

import pytest

from evodiff_torx import benchmark
from evodiff_torx.benchmark import (
    FAMILIES,
    format_comparison_table,
    format_summary,
    run_benchmark,
)


def make_result(
    family: str = "FAM_A",
    seq_len: int = 31,
    n_train: int = 4000,
    model_accuracy: float = 0.7750,
    pssm_baseline_accuracy: float = 0.5292,
) -> dict:
    """A `train_and_evaluate`-shaped result, with `margin` kept consistent.

    `n_holdout` is deliberately distinct from `n_train` so a table that reports
    the wrong one is caught rather than coincidentally passing.
    """
    return {
        "family": family,
        "seq_len": seq_len,
        "n_train": n_train,
        "n_holdout": 400,
        "num_timesteps": 200,
        "num_train_steps": 1000,
        "model_accuracy": model_accuracy,
        "pssm_baseline_accuracy": pssm_baseline_accuracy,
        "margin": model_accuracy - pssm_baseline_accuracy,
        "final_loss": 1.234,
        "train_seconds": 20.0,
    }


@pytest.fixture
def three_results() -> list[dict]:
    """Stands in for the default sweep, with the shape of the real numbers."""
    return [
        make_result("YAP1_HUMAN", seq_len=31, model_accuracy=0.7750,
                    pssm_baseline_accuracy=0.5292),
        make_result("RL401_YEAST", seq_len=71, model_accuracy=0.7239,
                    pssm_baseline_accuracy=0.6298),
        make_result("PABP_YEAST", seq_len=82, model_accuracy=0.6034,
                    pssm_baseline_accuracy=0.3794),
    ]


class TestFormatComparisonTable:

    def test_has_a_header_rule_and_one_row_per_family(self, three_results):
        lines = format_comparison_table(three_results).splitlines()

        assert len(lines) == 2 + len(three_results)
        assert set(lines[1]) <= {"-", " "}

    def test_reports_every_required_column(self, three_results):
        header = format_comparison_table(three_results).splitlines()[0]
        for column in ("family", "n_positions", "n_sequences", "model", "PSSM", "margin"):
            assert column in header

    def test_rows_follow_the_input_order(self, three_results):
        rows = format_comparison_table(three_results).splitlines()[2:]
        assert [row.split()[0] for row in rows] == [
            "YAP1_HUMAN",
            "RL401_YEAST",
            "PABP_YEAST",
        ]

    def test_row_carries_the_right_value_in_each_column(self):
        """Pins the result-key -> column mapping, e.g. n_positions is `seq_len`."""
        result = make_result(
            "FAM_A", seq_len=71, n_train=4000,
            model_accuracy=0.75, pssm_baseline_accuracy=0.50,
        )
        row = format_comparison_table([result]).splitlines()[2]

        assert row.split() == ["FAM_A", "71", "4000", "0.7500", "0.5000", "+0.2500"]

    def test_shows_signed_margins_not_a_pass_fail_flag(self, three_results):
        """The report must expose how big each margin is, per the project's bar."""
        rendered = format_comparison_table(three_results)
        for result in three_results:
            assert f"{result['margin']:+.4f}" in rendered

    def test_negative_margin_renders_with_a_minus_sign(self):
        losing = make_result(model_accuracy=0.40, pssm_baseline_accuracy=0.55)
        assert "-0.1500" in format_comparison_table([losing])

    def test_columns_stay_aligned_under_a_long_family_name(self):
        """Widths follow the widest cell, so a long name must not shift the numbers."""
        results = [
            make_result("A"),
            make_result("A_MUCH_LONGER_FAMILY_NAME", n_train=123456),
        ]
        lines = format_comparison_table(results).splitlines()

        assert len({len(line) for line in lines}) == 1
        # The margin column is last and right-aligned, so equal line lengths mean
        # every row's margin ends at the same offset the header's does.
        assert all(line.endswith(("margin", "-", "+0.2458")) for line in lines)

    def test_empty_results_rejected(self):
        with pytest.raises(ValueError, match="no results"):
            format_comparison_table([])


class TestFormatSummary:

    def test_says_all_when_every_family_clears_the_bar(self, three_results):
        smallest = min(result["margin"] for result in three_results)
        summary = format_summary(three_results)

        assert "all 3 families beat" in summary
        assert f"{smallest:+.4f}" in summary

    def test_counts_the_winners_when_one_family_misses(self, three_results):
        three_results[1] = make_result(
            "RL401_YEAST", model_accuracy=0.40, pssm_baseline_accuracy=0.55
        )
        summary = format_summary(three_results)

        assert "2 of 3 families beat" in summary
        assert "-0.1500" in summary

    def test_reports_the_smallest_margin(self):
        results = [
            make_result("FAM_A", model_accuracy=0.90, pssm_baseline_accuracy=0.50),
            make_result("FAM_B", model_accuracy=0.60, pssm_baseline_accuracy=0.59),
        ]
        assert "+0.0100" in format_summary(results)


class TestRunBenchmark:

    @pytest.fixture
    def recorded_calls(self, monkeypatch) -> list[dict]:
        """Replace training with a recorder, so the loop is testable in milliseconds."""
        calls: list[dict] = []

        def fake_train_and_evaluate(**kwargs) -> dict:
            calls.append(kwargs)
            return make_result(kwargs["family"])

        monkeypatch.setattr(
            benchmark, "train_and_evaluate", fake_train_and_evaluate
        )
        return calls

    def test_runs_each_family_once_in_order(self, recorded_calls):
        results = run_benchmark(["FAM_A", "FAM_B"], progress=False)

        assert [call["family"] for call in recorded_calls] == ["FAM_A", "FAM_B"]
        assert [result["family"] for result in results] == ["FAM_A", "FAM_B"]

    def test_forwards_the_run_configuration(self, recorded_calls):
        run_benchmark(["FAM_A"], num_train_steps=7, seed=3, progress=False)

        assert recorded_calls[0]["num_train_steps"] == 7
        assert recorded_calls[0]["seed"] == 3

    def test_defaults_to_the_module_family_list(self, recorded_calls):
        run_benchmark(progress=False)
        assert [call["family"] for call in recorded_calls] == FAMILIES


def test_default_families_are_the_three_benchmarked_alignments():
    """The README's results table is written for exactly these three."""
    assert FAMILIES == ["YAP1_HUMAN", "RL401_YEAST", "PABP_YEAST"]
