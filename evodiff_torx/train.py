"""Train the D3PM denoiser on one protein family and score it against a PSSM.

The end-to-end loop: load a family's fixed-width alignment, tokenize it, split
off a held-out set, and fit `evodiff_torx.denoiser.Denoiser` to reconstruct the
original tokens from `evodiff_torx.corruption.ForwardCorruption`'s noisy samples.
The objective is the D3PM "predict x_0 from x_t" reconstruction cross-entropy,
the simplified variant EvoDiff also supports, not the full LVB/KL loss.

`train_and_evaluate` is the whole pipeline as one call and returns a plain dict,
so a multi-family driver can loop over it directly instead of shelling out.

Reading the two accuracies it reports: the model sees the corrupted sequence, so
positions that noise happened to leave alone are free -- echoing the input alone
scores ~0.54 on these families. The PSSM baseline sees no input at all. The
margin therefore measures context modeling on top of per-column marginals, which
is the bar this project is held to; it is not a from-scratch generation score.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

# `python evodiff_torx/train.py` puts this file's directory on sys.path, not the
# repo root, so the absolute imports below would not resolve. Prepend the root so
# the documented CLI invocation works alongside `python -m evodiff_torx.train`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evodiff_torx.corruption import ForwardCorruption
from evodiff_torx.data import load_alignment
from evodiff_torx.denoiser import Denoiser
from evodiff_torx.schedule import NUM_DIFFUSION_STATES
from evodiff_torx.tokenizer import tokenize

# Defaults tuned on YAP1_HUMAN (L=31), RL401_YEAST (L=71) and PABP_YEAST (L=82);
# each clears the PSSM baseline by ~0.22 in well under two minutes on CPU.
# The loss plateaus by a few hundred steps on the short family and ~1000 on the
# long one, so 1000 is where extra steps stop buying accuracy. Subsampling the
# training pool to 4000 of the family's 21k-152k sequences costs nothing
# measurable in accuracy and keeps setup instant.
DEFAULT_NUM_TIMESTEPS = 200
DEFAULT_NUM_TRAIN_STEPS = 1000
DEFAULT_BATCH_SIZE = 128
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_TRAIN_SEQUENCES = 4000
DEFAULT_NUM_HOLDOUT_SEQUENCES = 400


def tokenize_sequences(sequences: list[str]) -> jnp.ndarray:
    """Stack equal-length sequence strings into an `(n, L)` int32 token matrix.

    Tokenizes the concatenation in one call and reshapes -- 200x faster than
    per-sequence `tokenize` at these batch sizes, and exact only because
    `evodiff_torx.data.load_alignment` guarantees a single width.
    """
    if not sequences:
        raise ValueError("no sequences to tokenize")
    seq_len = len(sequences[0])
    return tokenize("".join(sequences)).reshape(len(sequences), seq_len)


def split_holdout(
    sequences: list[str],
    num_train: int,
    num_holdout: int,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shuffle, then carve disjoint `(train_tokens, holdout_tokens)` token matrices.

    `num_train` is capped at whatever remains after the held-out slice, so small
    families and synthetic test fixtures do not need to know the family size.
    """
    if len(sequences) <= num_holdout:
        raise ValueError(
            f"need more than num_holdout={num_holdout} sequences to leave a "
            f"training set, got {len(sequences)}"
        )
    order = np.random.default_rng(seed).permutation(len(sequences))
    holdout = order[:num_holdout]
    train = order[num_holdout : num_holdout + num_train]
    return (
        tokenize_sequences([sequences[i] for i in train]),
        tokenize_sequences([sequences[i] for i in holdout]),
    )


def pssm_frequencies(
    tokens: jnp.ndarray, num_states: int = NUM_DIFFUSION_STATES
) -> jnp.ndarray:
    """Per-column empirical token frequencies of an `(n, L)` batch, shape `(L, num_states)`.

    Rows sum to 1. This is the position-specific scoring matrix the baseline
    below reduces to a consensus.
    """
    counts = jax.nn.one_hot(tokens, num_states, dtype=jnp.float32).sum(axis=0)
    return counts / counts.sum(axis=-1, keepdims=True)


def pssm_consensus(
    tokens: jnp.ndarray, num_states: int = NUM_DIFFUSION_STATES
) -> jnp.ndarray:
    """Most frequent token per column, shape `(L,)` -- the baseline's only prediction."""
    return jnp.argmax(pssm_frequencies(tokens, num_states), axis=-1)


def reconstruction_loss(
    model: Denoiser,
    corrupted: jnp.ndarray,
    timesteps: jnp.ndarray,
    original: jnp.ndarray,
) -> jnp.ndarray:
    """Mean per-position cross-entropy of `model`'s x_0 prediction against `original`.

    All three batch arguments are `(B, L)` (`timesteps` is `(B,)`). The denoiser
    emits `tokenizer.VOCAB_SIZE` logits including a mask class that no target
    ever takes; leaving it in the softmax is intended -- the model simply learns
    to suppress it.
    """
    logits = jax.vmap(model)(corrupted, timesteps)
    return optax.softmax_cross_entropy_with_integer_labels(logits, original).mean()


@eqx.filter_jit
def train_step(
    corruption: ForwardCorruption,
    model: Denoiser,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    key: jax.Array,
    train_tokens: jnp.ndarray,
    batch_size: int,
) -> tuple[Denoiser, optax.OptState, jnp.ndarray]:
    """One Adam step on a freshly drawn, freshly corrupted minibatch.

    `corruption` must be an argument rather than a closure: it is an
    `equinox.Module` with float leaves, so `jax.jit` would refuse to hold it
    static (see `ForwardCorruption`'s docstring). `filter_jit` traces its arrays
    and holds the non-array arguments -- `optimizer` and `batch_size` -- static,
    which is why the sampled batch shape is a compile-time constant.

    `filter_value_and_grad` differentiates only w.r.t. `model`, so the schedule
    in `corruption` is never touched by the optimizer.
    """
    batch_key, corrupt_key = jax.random.split(key)
    indices = jax.random.randint(batch_key, (batch_size,), 0, train_tokens.shape[0])
    batch = train_tokens[indices]
    corrupted, timesteps = corruption.corrupt_batch(corrupt_key, batch)
    loss, grads = eqx.filter_value_and_grad(reconstruction_loss)(
        model, corrupted, timesteps, batch
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    return eqx.apply_updates(model, updates), opt_state, loss


def holdout_accuracy(
    model: Denoiser,
    corruption: ForwardCorruption,
    key: jax.Array,
    holdout_tokens: jnp.ndarray,
) -> float:
    """Fraction of held-out positions the model recovers from a training-distribution corruption."""
    corrupted, timesteps = corruption.corrupt_batch(key, holdout_tokens)
    predictions = jnp.argmax(jax.vmap(model)(corrupted, timesteps), axis=-1)
    return float((predictions == holdout_tokens).mean())


def train_and_evaluate(
    family: str,
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    num_train_steps: int = DEFAULT_NUM_TRAIN_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_train_sequences: int = DEFAULT_NUM_TRAIN_SEQUENCES,
    num_holdout_sequences: int = DEFAULT_NUM_HOLDOUT_SEQUENCES,
    seed: int = 0,
    data_dir: Path | None = None,
    log_every: int = 0,
) -> dict:
    """Train on one family and score the result against its PSSM consensus.

    `family` is a name resolved under `data/alignments/`, or a path to an `.a2m`
    file. `log_every > 0` prints the running loss every that many steps.

    Returns, all under fixed keys so callers can tabulate several families::

        {"family", "seq_len", "n_train", "n_holdout", "num_timesteps",
         "num_train_steps", "model_accuracy", "pssm_baseline_accuracy",
         "margin", "final_loss", "train_seconds"}

    `margin` is `model_accuracy - pssm_baseline_accuracy`; positive is the bar.
    Everything downstream of `seed` is deterministic, so repeat calls with the
    same arguments return identical accuracies.
    """
    sequences = load_alignment(family, data_dir=data_dir)
    train_tokens, holdout_tokens = split_holdout(
        sequences, num_train_sequences, num_holdout_sequences, seed
    )
    seq_len = train_tokens.shape[1]

    corruption = ForwardCorruption(seq_len=seq_len, num_timesteps=num_timesteps)
    init_key, train_key, eval_key = jax.random.split(jax.random.key(seed), 3)
    model = Denoiser(key=init_key)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    started = time.perf_counter()
    loss = jnp.asarray(jnp.nan)
    for step in range(num_train_steps):
        train_key, step_key = jax.random.split(train_key)
        model, opt_state, loss = train_step(
            corruption, model, opt_state, optimizer, step_key, train_tokens, batch_size
        )
        if log_every and (step % log_every == 0 or step == num_train_steps - 1):
            print(f"  step {step:5d}  loss {float(loss):.4f}")
    train_seconds = time.perf_counter() - started

    model_accuracy = holdout_accuracy(model, corruption, eval_key, holdout_tokens)
    consensus = pssm_consensus(train_tokens)
    baseline_accuracy = float((holdout_tokens == consensus[None, :]).mean())

    return {
        "family": str(family),
        "seq_len": seq_len,
        "n_train": int(train_tokens.shape[0]),
        "n_holdout": int(holdout_tokens.shape[0]),
        "num_timesteps": num_timesteps,
        "num_train_steps": num_train_steps,
        "model_accuracy": model_accuracy,
        "pssm_baseline_accuracy": baseline_accuracy,
        "margin": model_accuracy - baseline_accuracy,
        "final_loss": float(loss),
        "train_seconds": train_seconds,
    }


def format_result(result: dict) -> str:
    """Human-readable block for one `train_and_evaluate` result."""
    return "\n".join(
        [
            f"family                 {result['family']}",
            f"sequence length        {result['seq_len']}",
            f"training sequences     {result['n_train']}",
            f"held-out sequences     {result['n_holdout']}",
            f"training steps         {result['num_train_steps']}"
            f" ({result['train_seconds']:.1f}s, final loss {result['final_loss']:.4f})",
            f"model accuracy         {result['model_accuracy']:.4f}",
            f"PSSM baseline accuracy {result['pssm_baseline_accuracy']:.4f}",
            f"margin                 {result['margin']:+.4f}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--family",
        required=True,
        help="family name under data/alignments/ (e.g. YAP1_HUMAN), or an .a2m path",
    )
    parser.add_argument("--num-timesteps", type=int, default=DEFAULT_NUM_TIMESTEPS)
    parser.add_argument("--num-train-steps", type=int, default=DEFAULT_NUM_TRAIN_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--num-train-sequences", type=int, default=DEFAULT_NUM_TRAIN_SEQUENCES
    )
    parser.add_argument(
        "--num-holdout-sequences", type=int, default=DEFAULT_NUM_HOLDOUT_SEQUENCES
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-every", type=int, default=100, help="0 to silence per-step logging"
    )
    args = parser.parse_args()

    print(f"training on {args.family}")
    result = train_and_evaluate(
        family=args.family,
        num_timesteps=args.num_timesteps,
        num_train_steps=args.num_train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_sequences=args.num_train_sequences,
        num_holdout_sequences=args.num_holdout_sequences,
        seed=args.seed,
        log_every=args.log_every,
    )
    print(format_result(result))
    verdict = "beats" if result["margin"] > 0 else "does NOT beat"
    print(f"\nmodel {verdict} the PSSM baseline")


if __name__ == "__main__":
    main()
