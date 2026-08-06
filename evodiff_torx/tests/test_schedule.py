import jax.numpy as jnp
import pytest

from evodiff_torx import tokenizer as tk
from evodiff_torx.schedule import (
    NUM_DIFFUSION_STATES,
    beta_schedule,
    cumulative_transitions,
    uniform_transition_schedule,
)

# float32 tolerance: q_bar accumulates one matmul per timestep.
ATOL = 1e-5

TIMESTEPS = [1, 2, 5, 50, 200]
STATE_COUNTS = [2, 4, 21, 22]


def test_beta_schedule_matches_sohl_dickstein_formula_by_hand():
    # beta_t = 1 / (T - t + 1); for T=5 that is 1/6, 1/5, 1/4, 1/3, 1/2.
    assert beta_schedule(5) == pytest.approx([1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2], abs=ATOL)
    assert beta_schedule(1) == pytest.approx([1 / 2], abs=ATOL)
    assert beta_schedule(2) == pytest.approx([1 / 3, 1 / 2], abs=ATOL)


@pytest.mark.parametrize("num_timesteps", TIMESTEPS)
def test_beta_schedule_increases_within_unit_interval(num_timesteps):
    betas = beta_schedule(num_timesteps)
    assert betas.shape == (num_timesteps,)
    assert jnp.all(betas > 0.0) and jnp.all(betas <= 0.5)
    assert jnp.all(jnp.diff(betas) > 0.0)
    # Endpoints: least corruption first, most last.
    assert betas[0] == pytest.approx(1 / (num_timesteps + 1), abs=ATOL)
    assert betas[-1] == pytest.approx(0.5, abs=ATOL)


def test_beta_schedule_rejects_empty_schedule():
    with pytest.raises(ValueError, match="num_timesteps"):
        beta_schedule(0)


def test_schedule_rejects_degenerate_state_count():
    with pytest.raises(ValueError, match="num_states"):
        uniform_transition_schedule(4, num_states=1)


@pytest.mark.parametrize("num_timesteps", TIMESTEPS)
@pytest.mark.parametrize("num_states", STATE_COUNTS)
def test_both_matrix_stacks_are_column_stochastic(num_timesteps, num_states):
    q, q_bar = uniform_transition_schedule(num_timesteps, num_states)
    for name, matrices in (("q", q), ("q_bar", q_bar)):
        assert matrices.shape == (num_timesteps, num_states, num_states), name
        assert jnp.all(matrices >= 0.0), name
        assert matrices.sum(axis=1) == pytest.approx(1.0, abs=ATOL), name


@pytest.mark.parametrize("num_timesteps", TIMESTEPS)
def test_single_step_matrix_matches_closed_form(num_timesteps):
    # q[t] = (1 - beta_t) I + beta_t J / K
    num_states = 7
    q, _ = uniform_transition_schedule(num_timesteps, num_states)
    betas = beta_schedule(num_timesteps)[:, None, None]
    identity = jnp.eye(num_states)
    uniform = jnp.ones((num_states, num_states)) / num_states
    expected = (1.0 - betas) * identity + betas * uniform
    assert q == pytest.approx(expected, abs=ATOL)


@pytest.mark.parametrize("num_timesteps", [1, 2, 5, 50])
def test_cumulative_matrix_equals_sequential_single_steps_on_one_hot(num_timesteps):
    """q_bar[t] applied once == q[0], then q[1], ... then q[t] applied in turn."""
    num_states = 6
    q, q_bar = uniform_transition_schedule(num_timesteps, num_states)
    initial = jnp.eye(num_states)[:, 2]  # one-hot column vector, state 2

    state = initial
    for t in range(num_timesteps):
        state = q[t] @ state
        assert q_bar[t] @ initial == pytest.approx(state, abs=ATOL), f"t={t}"


def test_cumulative_transitions_applies_earliest_step_first():
    """Pin composition order with NON-commuting matrices.

    The real schedule's q[t] = aI + bJ/K all commute with each other, so a
    reversed cumulative product would pass every test above. This is the only
    test that actually catches a backwards `cumulative_transitions`.
    """
    first = jnp.array([[1.0, 1.0], [0.0, 0.0]])  # both states -> state 0
    second = jnp.array([[0.0, 1.0], [1.0, 0.0]])  # swap
    q_bar = cumulative_transitions(jnp.stack([first, second]))

    assert q_bar[0] == pytest.approx(first, abs=ATOL)
    assert q_bar[1] == pytest.approx(second @ first, abs=ATOL)
    assert q_bar[1] != pytest.approx(first @ second, abs=ATOL)

    # Read it off semantically: collapse to state 0, then swap -> state 1.
    assert q_bar[1] @ jnp.array([1.0, 0.0]) == pytest.approx([0.0, 1.0], abs=ATOL)


@pytest.mark.parametrize("num_states", [4, 21])
def test_corruption_increases_monotonically_toward_uniform(num_states):
    num_timesteps = 100
    _, q_bar = uniform_transition_schedule(num_timesteps, num_states)
    uniform = jnp.ones((num_states, num_states)) / num_states

    distance = jnp.abs(q_bar - uniform).max(axis=(1, 2))
    assert jnp.all(jnp.diff(distance) < 0.0)
    # First step barely corrupts; the last is close to the uniform prior.
    assert distance[0] > distance[-1]
    assert distance[-1] < 1.0 / num_states


@pytest.mark.parametrize("num_timesteps", TIMESTEPS)
def test_cumulative_retention_matches_analytic_alpha_bar(num_timesteps):
    """prod_{s<=t}(1 - beta_s) telescopes to (T - t) / (T + 1)."""
    num_states = 5
    _, q_bar = uniform_transition_schedule(num_timesteps, num_states)
    t = jnp.arange(num_timesteps)
    alpha_bar = (num_timesteps - t) / (num_timesteps + 1.0)

    identity = jnp.eye(num_states)
    uniform = jnp.ones((num_states, num_states)) / num_states
    expected = alpha_bar[:, None, None] * identity + (1.0 - alpha_bar)[:, None, None] * uniform
    assert q_bar == pytest.approx(expected, abs=ATOL)


def test_diffusion_states_exclude_mask_and_prefix_the_tokenizer_alphabet():
    """schedule.py indexes tokenized data directly -- only valid if mask is last."""
    assert NUM_DIFFUSION_STATES == tk.VOCAB_SIZE - 1 == 21
    assert tk.MASK_INDEX == tk.VOCAB_SIZE - 1
    assert tk.ALPHABET[:NUM_DIFFUSION_STATES] == tk.AMINO_ACIDS + tk.GAP
    assert tk.GAP_INDEX < NUM_DIFFUSION_STATES


def test_default_state_count_is_used_when_omitted():
    q, _ = uniform_transition_schedule(3)
    assert q.shape == (3, NUM_DIFFUSION_STATES, NUM_DIFFUSION_STATES)


def test_returns_named_fields_and_unpacks_positionally():
    schedule = uniform_transition_schedule(4, 5)
    q, q_bar = schedule
    assert schedule.q is q
    assert schedule.q_bar is q_bar
