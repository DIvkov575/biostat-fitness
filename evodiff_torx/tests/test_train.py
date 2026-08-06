"""Tests for the D3PM training loop and its PSSM baseline.

The end-to-end test runs against a synthetic `.a2m` fixture rather than a real
family: same code path, but a few hundred short sequences and a handful of steps
keep it fast. It asserts wiring and shapes, NOT that the model beats the
baseline -- at this step count that would be a flaky coin flip. The real
comparison is `train.py --family YAP1_HUMAN`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from evodiff_torx.corruption import ForwardCorruption
from evodiff_torx.data import load_alignment
from evodiff_torx.denoiser import Denoiser
from evodiff_torx.schedule import NUM_DIFFUSION_STATES
from evodiff_torx.tokenizer import AMINO_ACIDS, GAP, VOCAB_SIZE, tokenize
from evodiff_torx.train import (
    format_result,
    holdout_accuracy,
    pssm_consensus,
    pssm_frequencies,
    reconstruction_loss,
    split_holdout,
    tokenize_sequences,
    train_and_evaluate,
    train_step,
)

SEQ_LEN = 16
NUM_SEQUENCES = 300
NUM_TIMESTEPS = 20


class _ConstantLogits(eqx.Module):
    """Stand-in denoiser returning fixed logits, so the loss can be checked by hand."""

    logits: jnp.ndarray

    def __call__(self, tokens, t) -> jnp.ndarray:
        del tokens, t
        return self.logits


@pytest.fixture(scope="module")
def synthetic_family(tmp_path_factory) -> str:
    """A tiny `.a2m` file with a strong per-column consensus, as a resolvable path.

    Each column has one dominant residue that 70% of records carry, so the PSSM
    baseline is meaningfully above chance and the columns are learnable.
    """
    rng = np.random.default_rng(0)
    alphabet = np.array(list(AMINO_ACIDS + GAP))
    consensus = rng.choice(alphabet, size=SEQ_LEN)

    lines = []
    for index in range(NUM_SEQUENCES):
        keep = rng.random(SEQ_LEN) < 0.7
        residues = np.where(keep, consensus, rng.choice(alphabet, size=SEQ_LEN))
        lines.append(f">synthetic_{index}\n{''.join(residues)}\n")

    path = tmp_path_factory.mktemp("alignments") / "SYNTHETIC_FAMILY.a2m"
    path.write_text("".join(lines))
    return str(path)


@pytest.fixture(scope="module")
def sequences(synthetic_family) -> list[str]:
    return load_alignment(synthetic_family)


class TestTokenizeSequences:

    def test_matches_per_sequence_tokenize(self, sequences):
        """The concatenate-then-reshape shortcut must equal the obvious loop."""
        batched = tokenize_sequences(sequences[:32])
        expected = jnp.stack([tokenize(sequence) for sequence in sequences[:32]])

        assert batched.shape == (32, SEQ_LEN)
        assert batched.dtype == jnp.int32
        assert jnp.array_equal(batched, expected)

    def test_tokens_stay_inside_the_diffusion_alphabet(self, sequences):
        """Never emits the mask index -- feeding it to the corruption gate is silent garbage."""
        tokens = tokenize_sequences(sequences)
        assert int(tokens.min()) >= 0
        assert int(tokens.max()) < NUM_DIFFUSION_STATES

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match="no sequences"):
            tokenize_sequences([])


class TestSplitHoldout:

    def test_shapes_and_disjointness(self, sequences):
        """No held-out sequence may also be trained on -- that would inflate accuracy."""
        train, holdout = split_holdout(sequences, num_train=100, num_holdout=40, seed=0)

        assert train.shape == (100, SEQ_LEN)
        assert holdout.shape == (40, SEQ_LEN)

        # Random 16-mers over 21 states are distinct in practice, so comparing by
        # value is a faithful stand-in for comparing record identity.
        train_rows = {tuple(row) for row in np.asarray(train).tolist()}
        holdout_rows = {tuple(row) for row in np.asarray(holdout).tolist()}
        assert len(train_rows) == 100
        assert len(holdout_rows) == 40
        assert train_rows.isdisjoint(holdout_rows)

    def test_train_size_capped_by_available_sequences(self, sequences):
        train, holdout = split_holdout(
            sequences, num_train=10_000, num_holdout=50, seed=0
        )
        assert train.shape[0] == len(sequences) - 50
        assert holdout.shape[0] == 50

    def test_seed_controls_the_split(self, sequences):
        first, _ = split_holdout(sequences, 100, 40, seed=0)
        same, _ = split_holdout(sequences, 100, 40, seed=0)
        other, _ = split_holdout(sequences, 100, 40, seed=1)

        assert jnp.array_equal(first, same)
        assert not jnp.array_equal(first, other)

    def test_holdout_larger_than_family_rejected(self, sequences):
        with pytest.raises(ValueError, match="num_holdout"):
            split_holdout(sequences, num_train=10, num_holdout=len(sequences), seed=0)


class TestPSSM:

    def test_frequencies_are_per_column_distributions(self, sequences):
        frequencies = pssm_frequencies(tokenize_sequences(sequences))

        assert frequencies.shape == (SEQ_LEN, NUM_DIFFUSION_STATES)
        assert jnp.allclose(frequencies.sum(axis=-1), 1.0, atol=1e-5)
        assert jnp.all(frequencies >= 0.0)

    def test_consensus_is_the_column_mode(self):
        # Column 0 is token 3 twice; column 1 is token 7 twice.
        tokens = jnp.array([[3, 7], [3, 1], [5, 7]], dtype=jnp.int32)
        assert jnp.array_equal(pssm_consensus(tokens), jnp.array([3, 7]))

    def test_consensus_recovers_a_planted_column(self, sequences):
        """With 70% of records carrying the planted residue, the mode must find it."""
        tokens = tokenize_sequences(sequences)
        consensus = pssm_consensus(tokens)
        accuracy = float((tokens == consensus[None, :]).mean())
        assert accuracy > 0.5


class TestReconstructionLoss:

    def test_untrained_loss_is_near_the_uniform_softmax_cost(self):
        model = Denoiser(key=jax.random.key(0))
        tokens = jnp.zeros((2, SEQ_LEN), dtype=jnp.int32)
        timesteps = jnp.ones(2, dtype=jnp.int32)

        untrained = float(reconstruction_loss(model, tokens, timesteps, tokens))
        # A uniform 22-way softmax costs log(22) ~ 3.09; an untrained net is near it.
        assert untrained == pytest.approx(float(jnp.log(VOCAB_SIZE)), abs=1.0)

    def test_matches_hand_computed_cross_entropy(self):
        """Pin the loss to the textbook formula using a model with known outputs."""
        logits = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        targets = jnp.array([[0, 2]])
        expected = 0.5 * (
            -jax.nn.log_softmax(logits[0])[0] - jax.nn.log_softmax(logits[1])[2]
        )

        actual = reconstruction_loss(
            _ConstantLogits(logits),
            corrupted=jnp.zeros((1, 2), dtype=jnp.int32),
            timesteps=jnp.ones(1, dtype=jnp.int32),
            original=targets,
        )
        assert float(actual) == pytest.approx(float(expected), abs=1e-6)

    def test_gradients_reach_only_the_model(self):
        """The fixed schedule must never receive an update."""
        model = Denoiser(key=jax.random.key(0))
        corruption = ForwardCorruption(seq_len=SEQ_LEN, num_timesteps=NUM_TIMESTEPS)
        tokens = jnp.zeros((4, SEQ_LEN), dtype=jnp.int32)
        corrupted, timesteps = corruption.corrupt_batch(jax.random.key(1), tokens)

        grads = eqx.filter_grad(reconstruction_loss)(
            model, corrupted, timesteps, tokens
        )
        leaves = [leaf for leaf in jax.tree.leaves(grads) if eqx.is_inexact_array(leaf)]
        assert leaves
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
        assert any(jnp.any(leaf != 0.0) for leaf in leaves)


class TestTrainStep:

    def test_step_updates_parameters_and_leaves_schedule_untouched(self, sequences):
        train_tokens = tokenize_sequences(sequences[:64])
        corruption = ForwardCorruption(seq_len=SEQ_LEN, num_timesteps=NUM_TIMESTEPS)
        model = Denoiser(key=jax.random.key(0))
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        updated, _, loss = train_step(
            corruption, model, opt_state, optimizer, jax.random.key(1), train_tokens, 8
        )

        assert jnp.isfinite(loss)
        assert not jnp.array_equal(updated.out_proj.weight, model.out_proj.weight)
        assert jnp.array_equal(corruption.q_bar, corruption.gate.base.matrices)

    def test_loss_decreases_over_a_short_run(self, sequences):
        """Sanity that the optimizer is actually pointed downhill."""
        train_tokens = tokenize_sequences(sequences)
        corruption = ForwardCorruption(seq_len=SEQ_LEN, num_timesteps=NUM_TIMESTEPS)
        model = Denoiser(key=jax.random.key(0))
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        key = jax.random.key(1)
        losses = []
        for _ in range(60):
            key, step_key = jax.random.split(key)
            model, opt_state, loss = train_step(
                corruption, model, opt_state, optimizer, step_key, train_tokens, 32
            )
            losses.append(float(loss))

        assert np.mean(losses[-10:]) < np.mean(losses[:10])


class TestHoldoutAccuracy:

    def test_accuracy_is_a_fraction(self, sequences):
        holdout = tokenize_sequences(sequences[:32])
        corruption = ForwardCorruption(seq_len=SEQ_LEN, num_timesteps=NUM_TIMESTEPS)
        model = Denoiser(key=jax.random.key(0))

        accuracy = holdout_accuracy(model, corruption, jax.random.key(2), holdout)
        assert 0.0 <= accuracy <= 1.0


def _short_run(family: str) -> dict:
    return train_and_evaluate(
        family=family,
        num_timesteps=NUM_TIMESTEPS,
        num_train_steps=40,
        batch_size=32,
        num_train_sequences=200,
        num_holdout_sequences=50,
        seed=0,
    )


@pytest.fixture(scope="module")
def result(synthetic_family) -> dict:
    return _short_run(synthetic_family)


class TestTrainAndEvaluate:

    def test_reports_the_family_stats_it_was_asked_for(self, result, synthetic_family):
        assert result["family"] == synthetic_family
        assert result["seq_len"] == SEQ_LEN
        assert result["n_train"] == 200
        assert result["n_holdout"] == 50
        assert result["num_timesteps"] == NUM_TIMESTEPS
        assert result["num_train_steps"] == 40

    def test_accuracies_are_fractions_and_margin_is_their_difference(self, result):
        assert 0.0 <= result["model_accuracy"] <= 1.0
        assert 0.0 <= result["pssm_baseline_accuracy"] <= 1.0
        assert result["margin"] == pytest.approx(
            result["model_accuracy"] - result["pssm_baseline_accuracy"]
        )

    def test_loss_and_timing_are_finite(self, result):
        assert np.isfinite(result["final_loss"])
        assert result["train_seconds"] > 0.0

    def test_same_seed_reproduces_the_run(self, synthetic_family, result):
        repeat = _short_run(synthetic_family)
        assert repeat["model_accuracy"] == result["model_accuracy"]
        assert repeat["pssm_baseline_accuracy"] == result["pssm_baseline_accuracy"]

    def test_result_keys_are_stable(self, result):
        """`benchmark.py` tabulates these keys; renaming one breaks it silently."""
        assert set(result) == {
            "family",
            "seq_len",
            "n_train",
            "n_holdout",
            "num_timesteps",
            "num_train_steps",
            "model_accuracy",
            "pssm_baseline_accuracy",
            "margin",
            "final_loss",
            "train_seconds",
        }

    def test_format_result_reports_both_accuracies(self, result):
        rendered = format_result(result)
        assert f"{result['model_accuracy']:.4f}" in rendered
        assert f"{result['pssm_baseline_accuracy']:.4f}" in rendered
        assert f"{result['margin']:+.4f}" in rendered
