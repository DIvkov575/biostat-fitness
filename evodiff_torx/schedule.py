"""D3PM uniform-transition noise schedule (no BLOSUM, no absorbing/mask state).

Mirrors the algorithm of EvoDiff's ``_beta_schedule('sohl-dickstein')`` +
``q_random_schedule`` + ``cumprod_matrix``, reimplemented in ``jax.numpy``.

Conventions used throughout this module:

- Matrices are **column-stochastic**: ``Q[out, in]``, so every *column* sums to
  1 and a state is a column vector. One diffusion step is ``x_t = Q_t @ x_t1``.
- ``Q[t]`` is the single-step transition applied at step ``t`` (0-indexed).
- ``Q_bar[t] = Q[t] @ Q[t - 1] @ ... @ Q[0]`` — the cumulative transition taking
  the *original* sequence straight to its distribution at noise level ``t``.
  The rightmost factor is applied first, so ``Q[0]`` acts on the state first.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from evodiff_torx.tokenizer import AMINO_ACIDS, GAP

# The uniform-transition (multinomial) D3PM corrupts among *real* states only:
# its stationary distribution is uniform over all K states, so every state in
# the schedule's alphabet is something the model may legitimately emit. The mask
# token `#` is therefore NOT part of K -- including it would make `#` a valid
# corrupted residue and put 1/K of the t=T prior on it. (Mask belongs to the
# *absorbing* D3PM variant, which this project does not implement.) The gap `-`
# IS included: it is a real alignment state present in the MSA data.
#
# Pass this as `num_states`, not `tokenizer.VOCAB_SIZE`.
#
# Because MASK is the last token in ALPHABET, indices 0..NUM_DIFFUSION_STATES-1
# are exactly `AMINO_ACIDS + GAP`, so a (K, K) matrix indexes tokenized data
# directly with no remapping. `test_schedule.py` pins that alphabet invariant.
NUM_DIFFUSION_STATES = len(AMINO_ACIDS + GAP)


class UniformTransitionSchedule(NamedTuple):
    """Per-step and cumulative transition matrices, both ``(T, K, K)``.

    Unpacks as ``(q, q_bar)``; prefer the named fields at call sites, since
    mixing the two up is silent.
    """

    q: jnp.ndarray
    q_bar: jnp.ndarray


def beta_schedule(num_timesteps: int) -> jnp.ndarray:
    """Sohl-Dickstein corruption rates, shape ``(num_timesteps,)``.

    ``beta_t = 1 / (T - t + 1)``, increasing from ``1 / (T + 1)`` at ``t = 0``
    to ``1 / 2`` at ``t = T - 1``.
    """
    if num_timesteps < 1:
        raise ValueError(f"num_timesteps must be >= 1, got {num_timesteps}")
    t = jnp.arange(num_timesteps, dtype=jnp.float32)
    return 1.0 / (num_timesteps - t + 1.0)


def cumulative_transitions(q: jnp.ndarray) -> jnp.ndarray:
    """Running composition of per-step matrices: ``out[t] = q[t] @ ... @ q[0]``.

    ``q`` is ``(T, K, K)`` column-stochastic, earliest step first; the result has
    the same shape.
    """

    def compose(acc, q_t):
        acc = q_t @ acc
        return acc, acc

    identity = jnp.eye(q.shape[-1], dtype=q.dtype)
    _, q_bar = jax.lax.scan(compose, identity, q)
    return q_bar


def uniform_transition_schedule(
    num_timesteps: int, num_states: int = NUM_DIFFUSION_STATES
) -> UniformTransitionSchedule:
    """Build the D3PM uniform-transition schedule over ``num_states`` states.

    Each ``q[t]`` mixes ``beta_t`` of every column's mass uniformly across all
    states and keeps the rest in place, giving the closed form
    ``q[t] = (1 - beta_t) I + beta_t J / K``.

    Returns ``(q, q_bar)``, each ``(num_timesteps, num_states, num_states)`` and
    column-stochastic. See the module docstring for the ``q_bar`` composition
    order, and ``NUM_DIFFUSION_STATES`` for why ``num_states`` excludes the mask
    token. ``q_bar`` accumulates ``num_timesteps`` float32 matmuls, so its
    columns sum to 1 only to ~1e-5 for schedules of ~1000 steps.
    """
    if num_states < 2:
        raise ValueError(f"num_states must be >= 2, got {num_states}")

    betas = beta_schedule(num_timesteps)
    uniform = jnp.ones((num_states, num_states), dtype=jnp.float32) / num_states
    off_diagonal = betas[:, None, None] * uniform
    retained = 1.0 - off_diagonal.sum(axis=1)  # per-column mass left on the diagonal
    q = off_diagonal + retained[:, None, :] * jnp.eye(num_states, dtype=jnp.float32)
    return UniformTransitionSchedule(q=q, q_bar=cumulative_transitions(q))
