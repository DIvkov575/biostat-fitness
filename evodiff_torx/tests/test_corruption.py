"""Tests for the D3PM forward-corruption sampler."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evodiff_torx.corruption import (
    corruption_gate,
    CorruptedBatch,
    ForwardCorruption,
    MIN_TIMESTEP,
)
from evodiff_torx.schedule import NUM_DIFFUSION_STATES, uniform_transition_schedule
from evodiff_torx.tokenizer import MASK_INDEX

NUM_TIMESTEPS = 50
# Real family widths (YAP1_HUMAN, RL401_YEAST); nothing may be hardcoded to one.
FAMILY_WIDTHS = (31, 71)


def _tokens(seq_len: int, seed: int = 0) -> jnp.ndarray:
    return jax.random.randint(
        jax.random.key(seed), (seq_len,), 0, NUM_DIFFUSION_STATES, dtype=jnp.int32
    )


@pytest.fixture(scope="module", params=FAMILY_WIDTHS, ids=lambda w: f"L{w}")
def corruption(request) -> ForwardCorruption:
    return ForwardCorruption(seq_len=request.param, num_timesteps=NUM_TIMESTEPS)


class TestConstruction:

    @pytest.mark.parametrize("seq_len", FAMILY_WIDTHS)
    def test_gate_port_shapes_follow_seq_len(self, seq_len):
        _, q_bar = uniform_transition_schedule(NUM_TIMESTEPS)
        gate = corruption_gate(q_bar, seq_len)

        assert gate.input_ports["in"].shape == (seq_len, 1)
        assert gate.output_spec.shape == (seq_len, 1)
        # weight_tied: one scalar timestep shared by every position.
        assert gate.init_params(jax.random.key(0)).shape == ()

    def test_metadata_matches_schedule(self, corruption):
        assert corruption.seq_len == corruption.gate.n_tiles
        assert corruption.num_timesteps == NUM_TIMESTEPS
        assert corruption.num_states == NUM_DIFFUSION_STATES
        assert corruption.q.shape == corruption.q_bar.shape
        assert corruption.q_bar.shape == (
            NUM_TIMESTEPS,
            NUM_DIFFUSION_STATES,
            NUM_DIFFUSION_STATES,
        )

    def test_gate_holds_the_same_q_bar(self, corruption):
        assert jnp.array_equal(corruption.gate.base.matrices, corruption.q_bar)

    def test_empty_timestep_range_rejected(self):
        with pytest.raises(ValueError, match="num_timesteps must be >"):
            ForwardCorruption(seq_len=8, num_timesteps=MIN_TIMESTEP)

    def test_zero_length_sequence_rejected(self):
        _, q_bar = uniform_transition_schedule(NUM_TIMESTEPS)
        with pytest.raises(ValueError, match="seq_len must be >= 1"):
            corruption_gate(q_bar, 0)

    def test_mismatched_sequence_length_rejected(self, corruption):
        wrong = jnp.zeros(corruption.seq_len + 1, dtype=jnp.int32)
        with pytest.raises(ValueError, match="to match the gate's"):
            corruption.corrupt_sequence(jax.random.key(0), wrong, 1)

    def test_mask_token_corrupts_silently(self, corruption):
        """The hazard `corrupt_sequence` documents: ``MASK_INDEX == num_states``.

        Out-of-range tokens are neither clamped nor rejected -- the state lookup
        NaNs and the position degrades silently, staying inside the valid output
        range so nothing downstream notices. Contrasted against the highest valid
        state, which at ``t=1`` is preserved almost always.
        """
        assert MASK_INDEX == corruption.num_states

        rows = 256
        shape = (rows, corruption.seq_len)
        timesteps = jnp.full((rows,), MIN_TIMESTEP, dtype=jnp.int32)
        masked = jnp.full(shape, MASK_INDEX, dtype=jnp.int32)
        highest_valid = jnp.full(shape, corruption.num_states - 1, dtype=jnp.int32)

        from_mask = corruption.corrupt_batch_at(jax.random.key(0), masked, timesteps)
        from_valid = corruption.corrupt_batch_at(
            jax.random.key(1), highest_valid, timesteps
        )

        # No error, and the garbage is indistinguishable from a real token.
        assert jnp.all(from_mask < corruption.num_states)
        assert not jnp.any(from_mask == MASK_INDEX)
        # The valid state is preserved ~96% of the time at t=1; mask, never.
        assert float(jnp.mean(from_valid == corruption.num_states - 1)) > 0.9
        assert float(jnp.mean(from_mask == corruption.num_states - 1)) < 0.01


class TestCorruptBatch:

    batch_size = 6

    def test_shapes_and_dtypes(self, corruption):
        tokens = jnp.stack([_tokens(corruption.seq_len, s) for s in range(6)])
        result = corruption.corrupt_batch(jax.random.key(0), tokens)

        assert isinstance(result, CorruptedBatch)
        assert result.corrupted.shape == tokens.shape
        assert result.timesteps.shape == (tokens.shape[0],)
        assert result.corrupted.dtype == jnp.int32
        assert result.timesteps.dtype == jnp.int32

    def test_outputs_are_valid_tokens(self, corruption):
        tokens = jnp.stack([_tokens(corruption.seq_len, s) for s in range(6)])
        corrupted, _ = corruption.corrupt_batch(jax.random.key(1), tokens)

        assert int(corrupted.min()) >= 0
        assert int(corrupted.max()) < corruption.num_states

    def test_timesteps_are_in_range(self, corruption):
        timesteps = corruption.sample_timesteps(jax.random.key(2), 4096)

        assert int(timesteps.min()) >= MIN_TIMESTEP
        assert int(timesteps.max()) < corruption.num_timesteps

    def test_timesteps_vary_across_the_batch(self, corruption):
        """One `t` per SEQUENCE, independently drawn -- not one `t` per batch."""
        timesteps = np.asarray(corruption.sample_timesteps(jax.random.key(3), 64))

        assert len(set(timesteps.tolist())) > 1
        # 64 draws over 49 values: a broadcast bug would collapse to a single value.
        assert len(set(timesteps.tolist())) > 10

    def test_batch_rows_are_independent(self, corruption):
        """Identical inputs must not produce identical rows (per-row keys)."""
        tokens = jnp.tile(_tokens(corruption.seq_len), (8, 1))
        corrupted = corruption.corrupt_batch_at(
            jax.random.key(4), tokens, jnp.full((8,), NUM_TIMESTEPS - 1)
        )

        rows = {tuple(row.tolist()) for row in np.asarray(corrupted)}
        assert len(rows) == 8

    def test_filter_jit_compatible(self, corruption):
        """The pattern a training step should copy: module as a traced argument.

        Plain `jax.jit` on the bound method raises ``unhashable type`` -- it
        makes `self` static, and this module carries array leaves.
        """

        @eqx.filter_jit
        def step(corruption, key, tokens):
            return corruption.corrupt_batch(key, tokens)

        tokens = jnp.stack([_tokens(corruption.seq_len, s) for s in range(6)])
        corrupted, timesteps = step(corruption, jax.random.key(5), tokens)

        assert corrupted.shape == tokens.shape
        assert timesteps.shape == (tokens.shape[0],)
        assert int(corrupted.max()) < corruption.num_states

    def test_plain_jit_on_bound_method_is_rejected(self):
        """Pins the gotcha above, so the docstring can't silently go stale."""
        corruption = ForwardCorruption(seq_len=8, num_timesteps=NUM_TIMESTEPS)
        tokens = jnp.zeros((2, 8), dtype=jnp.int32)

        with pytest.raises(TypeError, match="unhashable"):
            jax.jit(corruption.corrupt_batch)(jax.random.key(0), tokens)


class TestMarginals:
    """Empirical per-position distributions against ``q_bar[t]``'s columns."""

    n_samples = 8000
    timestep = 30
    # At t=30 the closest pair of q_bar columns differs by 0.39, while sampling
    # noise at 8000 draws peaks near 0.017 across seeds -- so this tolerance is
    # ~10x the noise floor and still ~13x below the deviation a wrong column
    # would produce. Tighten it and the suite goes flaky, not more sensitive.
    atol = 0.03

    @pytest.mark.parametrize("seq_len", FAMILY_WIDTHS)
    def test_per_position_marginal_matches_q_bar_column(self, seq_len):
        corruption = ForwardCorruption(seq_len=seq_len, num_timesteps=NUM_TIMESTEPS)
        tokens = _tokens(seq_len, seed=7)
        batch = jnp.tile(tokens, (self.n_samples, 1))
        timesteps = jnp.full((self.n_samples,), self.timestep, dtype=jnp.int32)

        corrupted = np.asarray(
            corruption.corrupt_batch_at(jax.random.key(8), batch, timesteps)
        )
        q_bar_t = np.asarray(corruption.q_bar[self.timestep])

        # Every position, not a sample: a per-position wiring bug must not hide.
        for position in range(seq_len):
            empirical = np.bincount(
                corrupted[:, position], minlength=corruption.num_states
            ) / self.n_samples
            expected = q_bar_t[:, int(tokens[position])]
            deviation = np.abs(empirical - expected).max()
            assert deviation < self.atol, (
                f"position {position} (token {int(tokens[position])}) deviates by "
                f"{deviation:.4f} from q_bar[{self.timestep}]'s column"
            )

    def test_positions_are_independent(self, corruption):
        """i.i.d. positions: a pair's joint must factor into its marginals."""
        seq_len = corruption.seq_len
        constant = jnp.zeros(seq_len, dtype=jnp.int32)
        batch = jnp.tile(constant, (self.n_samples, 1))
        timesteps = jnp.full((self.n_samples,), self.timestep, dtype=jnp.int32)
        corrupted = np.asarray(
            corruption.corrupt_batch_at(jax.random.key(9), batch, timesteps)
        )

        for a, b in [(0, 1), (0, seq_len - 1), (seq_len // 2, seq_len - 1)]:
            joint = (
                np.histogram2d(
                    corrupted[:, a],
                    corrupted[:, b],
                    bins=corruption.num_states,
                    range=[[0, corruption.num_states]] * 2,
                )[0]
                / self.n_samples
            )
            product = np.outer(joint.sum(axis=1), joint.sum(axis=0))
            assert np.allclose(joint, product, atol=0.02), (
                f"positions {a} and {b} are correlated"
            )


def _shift_schedule(num_states: int, num_timesteps: int) -> jnp.ndarray:
    """Deterministic ``(T, K, K)`` stack where timestep ``t`` maps ``x -> x + t mod K``.

    A permutation schedule turns sampling into an exact assertion, which the real
    (symmetric, diffuse) ``q_bar`` cannot support: at ``t=30`` a transposed lookup
    is numerically identical and an off-by-one ``t`` sits inside sampling noise,
    so `TestMarginals` cannot see either. Here both are visible exactly.
    """
    shifts = (jnp.arange(num_states)[None, :] + jnp.arange(num_timesteps)[:, None]) % (
        num_states
    )
    # matrices[t][out, in] = 1 iff out == (in + t) % K.
    return jax.nn.one_hot(shifts, num_states, dtype=jnp.float32).transpose(0, 2, 1)


class TestTimestepIndexing:
    """Exact tests of the ``theta -> matrix`` lookup, via a permutation schedule."""

    num_timesteps = 7

    def _gate(self, seq_len, num_states=NUM_DIFFUSION_STATES):
        return corruption_gate(
            _shift_schedule(num_states, self.num_timesteps), seq_len
        )

    def test_shift_schedule_is_column_stochastic(self):
        """Guard the fixture: a broken schedule would make the rest vacuous."""
        stack = _shift_schedule(NUM_DIFFUSION_STATES, self.num_timesteps)

        assert stack.shape == (
            self.num_timesteps,
            NUM_DIFFUSION_STATES,
            NUM_DIFFUSION_STATES,
        )
        assert jnp.allclose(stack.sum(axis=1), 1.0)
        # Asymmetric for t != 0, which is what makes orientation testable.
        assert not jnp.allclose(stack[1], stack[1].T)

    @pytest.mark.parametrize("seq_len", FAMILY_WIDTHS)
    def test_theta_selects_exactly_its_timestep(self, seq_len):
        """Pins indexing AND ``q_bar[out, in]`` orientation with no tolerance.

        Off-by-one, a reversed stack, and a transposed lookup each produce a
        different shift, so all three fail here.
        """
        gate = self._gate(seq_len)
        tokens = _tokens(seq_len, seed=20)

        for t in range(self.num_timesteps):
            output = gate.sample(
                jax.random.key(21), {"in": tokens.reshape(-1, 1)}, jnp.array(t)
            ).reshape(-1)
            expected = (tokens + t) % NUM_DIFFUSION_STATES
            assert jnp.array_equal(output, expected), f"timestep {t} shifted wrongly"

    def test_all_positions_share_one_timestep(self, corruption):
        """``weight_tied=True``: one `t` per sequence, applied at every position."""
        seq_len = corruption.seq_len
        gate = self._gate(seq_len)
        tokens = _tokens(seq_len, seed=22)

        output = gate.sample(
            jax.random.key(23), {"in": tokens.reshape(-1, 1)}, jnp.array(3)
        ).reshape(-1)
        realized_shifts = (output - tokens) % NUM_DIFFUSION_STATES

        assert jnp.array_equal(realized_shifts, jnp.full(seq_len, 3))

    def test_batch_vmap_applies_each_rows_own_timestep(self):
        """Exact per-row check that `t` does not leak or reorder across the batch.

        Substitutes the permutation stack for the whole schedule -- ``q``,
        ``q_bar`` and the gate together -- so the instance stays self-consistent.
        """
        seq_len = FAMILY_WIDTHS[0]
        shift = _shift_schedule(NUM_DIFFUSION_STATES, self.num_timesteps)
        corruption = eqx.tree_at(
            lambda c: (c.q, c.q_bar, c.gate),
            ForwardCorruption(seq_len=seq_len, num_timesteps=self.num_timesteps),
            (shift, shift, self._gate(seq_len)),
        )
        tokens = jnp.stack([_tokens(seq_len, s) for s in range(self.num_timesteps)])
        timesteps = jnp.arange(self.num_timesteps, dtype=jnp.int32)

        corrupted = corruption.corrupt_batch_at(jax.random.key(24), tokens, timesteps)
        expected = (tokens + timesteps[:, None]) % NUM_DIFFUSION_STATES

        assert jnp.array_equal(corrupted, expected)


class TestTimestepEffect:

    n_samples = 2000

    def _fraction_unchanged(self, corruption, tokens, t, seed):
        batch = jnp.tile(tokens, (self.n_samples, 1))
        timesteps = jnp.full((self.n_samples,), t, dtype=jnp.int32)
        corrupted = np.asarray(
            corruption.corrupt_batch_at(jax.random.key(seed), batch, timesteps)
        )
        return float((corrupted == np.asarray(tokens)).mean())

    def test_corruption_grows_with_timestep(self, corruption):
        """Low `t` barely perturbs the sequence; high `t` approaches uniform."""
        tokens = _tokens(corruption.seq_len, seed=10)

        low = self._fraction_unchanged(corruption, tokens, MIN_TIMESTEP, seed=11)
        high = self._fraction_unchanged(
            corruption, tokens, corruption.num_timesteps - 1, seed=12
        )

        assert low > 0.9, f"t={MIN_TIMESTEP} corrupts too much: {low:.3f} unchanged"
        assert high < 0.2, f"t=T-1 corrupts too little: {high:.3f} unchanged"
        assert low > high

    def test_high_timestep_approaches_uniform(self, corruption):
        """The uniform-transition stationary distribution is ``1 / K`` per state."""
        tokens = _tokens(corruption.seq_len, seed=13)
        unchanged = self._fraction_unchanged(
            corruption, tokens, corruption.num_timesteps - 1, seed=14
        )

        assert unchanged == pytest.approx(1.0 / corruption.num_states, abs=0.05)

    def test_each_row_follows_its_own_timestep(self, corruption):
        """Per-row `t` must not leak across the batch vmap.

        Interleaves low and high timesteps so a bug that broadcasts row 0's `t`
        (or reverses the order) shows up as rows disagreeing with their own `t`.
        """
        tokens = _tokens(corruption.seq_len, seed=15)
        high = corruption.num_timesteps - 1
        pattern = jnp.asarray([MIN_TIMESTEP, high] * 64, dtype=jnp.int32)
        batch = jnp.tile(tokens, (pattern.shape[0], 1))

        corrupted = np.asarray(
            corruption.corrupt_batch_at(jax.random.key(16), batch, pattern)
        )
        unchanged = (corrupted == np.asarray(tokens)).mean(axis=1)
        low_rows = unchanged[::2].mean()
        high_rows = unchanged[1::2].mean()

        assert low_rows > 0.9, f"low-t rows corrupted too much: {low_rows:.3f}"
        assert high_rows < 0.2, f"high-t rows corrupted too little: {high_rows:.3f}"

    def test_identity_timestep_preserves_the_sequence(self, corruption):
        """A hand-injected identity ``q_bar`` must round-trip every position.

        Pins the ``q_bar[out, in]`` orientation end-to-end: a transposed lookup
        would still be a valid matrix here, but a permuted one would not.
        """
        identity = jnp.tile(
            jnp.eye(corruption.num_states), (corruption.num_timesteps, 1, 1)
        )
        gate = corruption_gate(identity, corruption.seq_len)
        tokens = _tokens(corruption.seq_len, seed=17)

        output = gate.sample(
            jax.random.key(18), {"in": tokens.reshape(-1, 1)}, jnp.array(3)
        )

        assert jnp.array_equal(output.reshape(-1), tokens)
