"""Reverse-denoising network: per-position logits over the clean token.

Scaled far down from EvoDiff's ByteNet backbone -- these families are tens of
residues wide, so a handful of dilated convolutions already spans a whole
sequence. Two properties the rest of the pipeline relies on:

- The network is written for a SINGLE sequence `(L,)` plus a scalar timestep.
  Batch it with `jax.vmap`, the standard Equinox pattern; batching is not baked
  into the module.
- `'SAME'` padding leaves `L` free, so one instance serves every family width
  (31, 71, 82, ...) with no reconstruction.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .tokenizer import VOCAB_SIZE

DEFAULT_HIDDEN_DIM = 64
DEFAULT_DILATIONS = (1, 2, 4, 8)
DEFAULT_KERNEL_SIZE = 3

_TIME_MAX_PERIOD = 10_000.0


def sinusoidal_timestep_features(t, dim: int) -> jax.Array:
    """Transformer-style sinusoidal encoding of a scalar `t`, shape `(dim,)`. `dim` must be even."""
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    half = dim // 2
    frequencies = jnp.exp(
        -jnp.log(_TIME_MAX_PERIOD) * jnp.arange(half, dtype=jnp.float32) / half
    )
    angles = jnp.asarray(t, dtype=jnp.float32) * frequencies
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])


def _normalize_each_position(norm: eqx.nn.LayerNorm, h: jax.Array) -> jax.Array:
    """Apply a channel-wise `LayerNorm` at every position of a `(C, L)` activation."""
    return jax.vmap(norm, in_axes=1, out_axes=1)(h)


class _DilatedBlock(eqx.Module):
    """Pre-norm residual block: dilated conv -> gelu -> pointwise conv."""

    norm: eqx.nn.LayerNorm
    dilated: eqx.nn.Conv1d
    pointwise: eqx.nn.Conv1d

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        *,
        key: jax.Array,
    ):
        dilated_key, pointwise_key = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(channels)
        self.dilated = eqx.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding="SAME",
            dilation=dilation,
            key=dilated_key,
        )
        self.pointwise = eqx.nn.Conv1d(channels, channels, 1, key=pointwise_key)

    def __call__(self, h: jax.Array) -> jax.Array:
        residual = _normalize_each_position(self.norm, h)
        residual = self.dilated(residual)
        residual = jax.nn.gelu(residual)
        return h + self.pointwise(residual)


class Denoiser(eqx.Module):
    """Maps a corrupted sequence `(L,)` and a scalar timestep to logits `(L, K)`.

    Constructed with a keyword-only `key`, matching `equinox.nn.Conv1d`:
    `Denoiser(key=key)` for the defaults, or override `hidden_dim` / `dilations`.
    """

    embedding: eqx.nn.Embedding
    time_proj: eqx.nn.Linear
    blocks: tuple[_DilatedBlock, ...]
    out_norm: eqx.nn.LayerNorm
    out_proj: eqx.nn.Linear

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        *,
        key: jax.Array,
    ):
        if hidden_dim < 2:
            raise ValueError(f"hidden_dim must be at least 2, got {hidden_dim}")
        if not dilations:
            raise ValueError("dilations must contain at least one entry")

        embed_key, time_key, out_key, *block_keys = jax.random.split(
            key, 3 + len(dilations)
        )
        self.embedding = eqx.nn.Embedding(vocab_size, hidden_dim, key=embed_key)
        # Sinusoidal features come in sin/cos pairs, so round the count down to even.
        self.time_proj = eqx.nn.Linear(2 * (hidden_dim // 2), hidden_dim, key=time_key)
        self.blocks = tuple(
            _DilatedBlock(hidden_dim, kernel_size, dilation, key=block_key)
            for dilation, block_key in zip(dilations, block_keys, strict=True)
        )
        self.out_norm = eqx.nn.LayerNorm(hidden_dim)
        self.out_proj = eqx.nn.Linear(hidden_dim, vocab_size, key=out_key)

    @property
    def vocab_size(self) -> int:
        return self.out_proj.out_features

    @property
    def receptive_radius(self) -> int:
        """Positions on either side that can influence one output position."""
        return sum(
            block.dilated.dilation[0] * (block.dilated.kernel_size[0] - 1) // 2
            for block in self.blocks
        )

    def __call__(self, tokens: jax.Array, t) -> jax.Array:
        """`tokens`: int indices `(L,)`. `t`: scalar timestep. Returns logits `(L, K)`."""
        # Conv1d is channels-first, so activations are carried as (hidden, L).
        h = jax.vmap(self.embedding)(tokens).T
        time = self.time_proj(
            sinusoidal_timestep_features(t, self.time_proj.in_features)
        )
        h = h + time[:, None]
        for block in self.blocks:
            h = block(h)
        h = _normalize_each_position(self.out_norm, h)
        return jax.vmap(self.out_proj)(h.T)


def as_deterministic_factor(model: Denoiser, seq_len: int):
    """Wrap `model` as a `torx.DeterministicFactor`: `(tokens, t) -> logits`.

    The network itself is plain Equinox and runs outside any circuit; torx is
    imported lazily so this is the only place the two meet.
    """
    import torx

    def fn(inputs, site_info):
        del site_info
        return model(inputs["tokens"], inputs["t"])

    return torx.DeterministicFactor(
        fn=fn,
        input_ports={
            "tokens": jax.ShapeDtypeStruct((seq_len,), jnp.int32),
            "t": jax.ShapeDtypeStruct((), jnp.int32),
        },
        output_spec=jax.ShapeDtypeStruct((seq_len, model.vocab_size), jnp.float32),
    )
