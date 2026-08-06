"""Run the D3PM training pipeline across several protein families and tabulate it.

Calls `evodiff_torx.train.train_and_evaluate` once per family and prints one
side-by-side table of model accuracy vs. PSSM baseline accuracy, with the actual
margins rather than a bare pass/fail. This is the project's headline result: the
bar is a positive margin on every family.

Runs sequentially -- each family is 15-30s on CPU with the training defaults, so
the whole default sweep is under two minutes and parallelism would only
complicate the output ordering. Exits non-zero if any family misses the bar,
which makes the script usable as a gate; the per-family run is deterministic
given `--seed`, so that exit code is reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

# Same bootstrap as `train.py`: running this file by path puts only its own
# directory on sys.path, so the absolute imports below need the repo root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evodiff_torx.train import DEFAULT_NUM_TRAIN_STEPS, train_and_evaluate

# Short, medium and long alignments (L=31, 71, 82) -- enough spread to show the
# margin holding as sequence length grows. Extend the list to sweep more.
FAMILIES = ["YAP1_HUMAN", "RL401_YEAST", "PABP_YEAST"]

# (header, alignment, cell) per column. The header names are the report's, the
# lambdas read `train_and_evaluate`'s result keys -- which differ, e.g. the
# reported "n_positions" is the result's "seq_len".
_COLUMNS: tuple[tuple[str, str, Callable[[dict], str]], ...] = (
    ("family", "<", lambda r: str(r["family"])),
    ("n_positions", ">", lambda r: str(r["seq_len"])),
    ("n_sequences", ">", lambda r: str(r["n_train"])),
    ("model", ">", lambda r: f"{r['model_accuracy']:.4f}"),
    ("PSSM", ">", lambda r: f"{r['pssm_baseline_accuracy']:.4f}"),
    ("margin", ">", lambda r: f"{r['margin']:+.4f}"),
)

_COLUMN_GAP = "  "


def run_benchmark(
    families: list[str] | None = None,
    num_train_steps: int = DEFAULT_NUM_TRAIN_STEPS,
    seed: int = 0,
    progress: bool = True,
) -> list[dict]:
    """Train and evaluate each family in turn, returning the results in that order.

    Each element is a `train_and_evaluate` result dict; `progress` prints a line
    per family before it starts, since a silent two-minute run looks hung.
    """
    families = FAMILIES if families is None else families
    results = []
    for index, family in enumerate(families, start=1):
        if progress:
            print(f"[{index}/{len(families)}] training on {family}", flush=True)
        results.append(
            train_and_evaluate(
                family=family, num_train_steps=num_train_steps, seed=seed
            )
        )
    return results


def format_comparison_table(results: list[dict]) -> str:
    """Render results as one aligned plain-text table, one row per family.

    Column widths fit the widest cell, so long family names do not break the
    alignment. Raises on an empty list rather than printing a bare header, which
    would read as "every family passed".
    """
    if not results:
        raise ValueError("no results to tabulate")

    cells = [[render(result) for _, _, render in _COLUMNS] for result in results]
    widths = [
        max(len(header), *(len(row[column]) for row in cells))
        for column, (header, _, _) in enumerate(_COLUMNS)
    ]

    def line(values: list[str]) -> str:
        return _COLUMN_GAP.join(
            f"{value:{align}{width}}"
            for value, (_, align, _), width in zip(values, _COLUMNS, widths)
        ).rstrip()

    header = line([header for header, _, _ in _COLUMNS])
    rule = _COLUMN_GAP.join("-" * width for width in widths)
    return "\n".join([header, rule, *(line(row) for row in cells)])


def format_summary(results: list[dict]) -> str:
    """One-line verdict: how many families cleared the bar, and by how little."""
    cleared = [result for result in results if result["margin"] > 0]
    smallest = min(result["margin"] for result in results)
    verdict = "all" if len(cleared) == len(results) else f"{len(cleared)} of"
    return (
        f"{verdict} {len(results)} families beat the PSSM baseline"
        f" (smallest margin {smallest:+.4f})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--families",
        nargs="+",
        default=FAMILIES,
        metavar="FAMILY",
        help="family names under data/alignments/ (default: %(default)s)",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=DEFAULT_NUM_TRAIN_STEPS,
        help="lower this for a quick smoke run (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = run_benchmark(
        families=args.families, num_train_steps=args.num_train_steps, seed=args.seed
    )
    print()
    print(format_comparison_table(results))
    print()
    print(format_summary(results))

    raise SystemExit(0 if all(result["margin"] > 0 for result in results) else 1)


if __name__ == "__main__":
    main()
