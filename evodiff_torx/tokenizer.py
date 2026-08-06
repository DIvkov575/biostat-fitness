"""Minimal 22-token protein alphabet: 20 canonical amino acids + gap + mask."""

from __future__ import annotations

import jax.numpy as jnp

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"
MASK = "#"

ALPHABET = AMINO_ACIDS + GAP + MASK
VOCAB_SIZE = len(ALPHABET)
GAP_INDEX = ALPHABET.index(GAP)
MASK_INDEX = ALPHABET.index(MASK)

_CHAR_TO_INDEX = {char: index for index, char in enumerate(ALPHABET)}


def tokenize(sequence: str) -> jnp.ndarray:
    """Map a sequence string to an int32 array of token indices, shape (len,)."""
    try:
        return jnp.array([_CHAR_TO_INDEX[char] for char in sequence], dtype=jnp.int32)
    except KeyError as err:
        raise ValueError(
            f"character {err.args[0]!r} is not in the alphabet {ALPHABET!r}"
        ) from None


def untokenize(indices) -> str:
    """Map token indices back to a sequence string."""
    return "".join(ALPHABET[int(index)] for index in jnp.ravel(jnp.asarray(indices)))


def one_hot(indices) -> jnp.ndarray:
    """One-hot encode token indices; trailing axis of size VOCAB_SIZE is appended."""
    return jnp.eye(VOCAB_SIZE, dtype=jnp.float32)[jnp.asarray(indices)]
