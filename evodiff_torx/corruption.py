"""D3PM forward corruption: apply ``Q_bar[t]`` independently at every position.

Wires `evodiff_torx.schedule`'s cumulative transition matrices into a torx
factor: one `torx.psc.PMarkov` gate (a lookup table selecting a timestep's
matrix) replicated across a sequence's positions by `torx.TiledFactor`.

Two nested ``vmap`` levels are in play, and which is which matters:

- `TiledFactor` maps the gate over sequence POSITIONS. ``weight_tied=True``
  makes every position share one ``theta``, i.e. one timestep -- exactly D3PM's
  behavior: one ``t`` is drawn per training SEQUENCE, then the same
  ``Q_bar[t]`` is applied independently at each of that sequence's positions.
- `ForwardCorruption.corrupt_batch` maps that over the training BATCH, so every
  sequence in a batch gets its own ``t``.

Tokens are integer indices in ``[0, num_states)``, which line up with
`evodiff_torx.tokenizer` because the mask token sorts last and is not a
diffusion state -- see `evodiff_torx.schedule.NUM_DIFFUSION_STATES`.
"""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import torx
from torx.psc import PMarkov

from evodiff_torx.schedule import NUM_DIFFUSION_STATES, uniform_transition_schedule

# EvoDiff's `D3PMCollater.__call__` draws `t = np.random.randint(1, num_timesteps)`,
# i.e. from `[1, T)`; mirrored here. Timestep 0 is skipped because the D3PM loss
# compares the reverse posterior against `Q_bar[t - 1]`, and `Q_bar[0]` is already
# one noise step rather than the identity (see `schedule.py`), so `t = 0` has no
# valid predecessor. Sampling from `[1, T)` keeps `t - 1` in range for every draw.
MIN_TIMESTEP = 1


def corruption_gate(q_bar: jnp.ndarray, seq_len: int) -> torx.TiledFactor:
    """Tile one `PMarkov` lookup gate across ``seq_len`` independent positions.

    ``q_bar`` is the ``(T, K, K)`` column-stochastic cumulative stack from
    `evodiff_torx.schedule.uniform_transition_schedule`. The result's ``"in"``
    port and output both have shape ``(seq_len, 1)`` -- the trailing ``1`` is
    `PMarkov`'s single-site port, which `TiledFactor` prefixes with the tile
    axis. `ForwardCorruption` hides that reshape from callers.
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    return torx.TiledFactor(
        base=PMarkov(sites=0, matrices=q_bar), n_tiles=seq_len, weight_tied=True
    )


class CorruptedBatch(NamedTuple):
    """One forward-corruption draw. Unpacks as ``(corrupted, timesteps)``.

    ``corrupted`` is ``(B, L)`` int32 and ``timesteps`` is ``(B,)`` int32, with
    ``timesteps[i]`` the noise level that produced ``corrupted[i]``. The loss
    needs both, so they travel together.
    """

    corrupted: jnp.ndarray
    timesteps: jnp.ndarray


class ForwardCorruption(eqx.Module):
    """Noise schedule plus the tiled torx gate that samples from it.

    Built for a fixed ``seq_len``, so one instance serves one family width;
    construct another for a different width. `corrupt_batch` is the entry point
    a training loop wants -- it draws the timesteps itself and returns them
    alongside the corrupted tokens.

    ``q`` and ``q_bar`` are float leaves of this pytree (as is the gate's copy of
    ``q_bar``), so filter them out of any `equinox.filter_grad` over a tree that
    contains this object; the schedule is fixed, never learned.

    Those array leaves also make plain ``jax.jit(instance.corrupt_batch)`` fail
    with ``unhashable type`` -- ``jit`` would hold ``self`` static. Take the
    instance as a traced argument of an `equinox.filter_jit` step instead::

        @eqx.filter_jit
        def train_step(corruption, model, opt_state, key, tokens):
            corrupted, timesteps = corruption.corrupt_batch(key, tokens)
            ...
    """

    q: jnp.ndarray
    q_bar: jnp.ndarray
    gate: torx.TiledFactor

    def __init__(
        self,
        seq_len: int,
        num_timesteps: int,
        num_states: int = NUM_DIFFUSION_STATES,
    ):
        if num_timesteps <= MIN_TIMESTEP:
            raise ValueError(
                f"num_timesteps must be > {MIN_TIMESTEP} for the timestep range "
                f"[{MIN_TIMESTEP}, num_timesteps) to be non-empty, got {num_timesteps}"
            )
        self.q, self.q_bar = uniform_transition_schedule(num_timesteps, num_states)
        self.gate = corruption_gate(self.q_bar, seq_len)

    @property
    def seq_len(self) -> int:
        return self.gate.n_tiles

    @property
    def num_timesteps(self) -> int:
        return self.q_bar.shape[0]

    @property
    def num_states(self) -> int:
        return self.q_bar.shape[-1]

    def sample_timesteps(self, key: jax.Array, batch_size: int) -> jnp.ndarray:
        """``(batch_size,)`` int32 timesteps, i.i.d. uniform on ``[MIN_TIMESTEP, T)``."""
        return jax.random.randint(
            key, (batch_size,), MIN_TIMESTEP, self.num_timesteps, dtype=jnp.int32
        )

    def corrupt_sequence(self, key: jax.Array, tokens, t) -> jnp.ndarray:
        """Corrupt ONE sequence ``(L,)`` at scalar timestep ``t``, returning ``(L,)``.

        Every position is drawn independently from ``q_bar[t][:, tokens[i]]``.

        Tokens MUST be in ``[0, num_states)``. A larger value is not clamped and
        not rejected: the gate's state lookup yields NaN, which casts to index 0,
        so the position silently corrupts as if it were state 0. The trap is
        `evodiff_torx.tokenizer.MASK_INDEX`, which equals ``num_states`` exactly
        -- feeding mask tokens in here produces plausible-looking garbage. Only
        pass tokens drawn from the diffusion alphabet.
        """
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        if tokens.shape != (self.seq_len,):
            raise ValueError(
                f"expected tokens of shape ({self.seq_len},) to match the gate's "
                f"tile count, got {tokens.shape}"
            )
        sampled = self.gate.sample(
            key, {"in": tokens.reshape(-1, 1)}, jnp.asarray(t, dtype=jnp.int32)
        )
        return sampled.reshape(-1)

    def corrupt_batch_at(self, key: jax.Array, tokens, timesteps) -> jnp.ndarray:
        """Corrupt ``(B, L)`` tokens at the given per-sequence ``(B,)`` timesteps."""
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        keys = jax.random.split(key, tokens.shape[0])
        return jax.vmap(self.corrupt_sequence)(
            keys, tokens, jnp.asarray(timesteps, dtype=jnp.int32)
        )

    def corrupt_batch(self, key: jax.Array, tokens) -> CorruptedBatch:
        """Draw one timestep per sequence and corrupt ``(B, L)`` original tokens.

        Returns `CorruptedBatch`; pass `CorruptedBatch.timesteps` to the loss,
        which needs the noise level each row was corrupted at.
        """
        tokens = jnp.asarray(tokens, dtype=jnp.int32)
        timestep_key, corrupt_key = jax.random.split(key)
        timesteps = self.sample_timesteps(timestep_key, tokens.shape[0])
        return CorruptedBatch(
            corrupted=self.corrupt_batch_at(corrupt_key, tokens, timesteps),
            timesteps=timesteps,
        )
