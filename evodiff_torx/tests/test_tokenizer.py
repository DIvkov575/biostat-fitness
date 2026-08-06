import jax.numpy as jnp
import pytest

from evodiff_torx import tokenizer as tk


def test_alphabet_is_22_unique_tokens():
    assert tk.VOCAB_SIZE == 22
    assert len(set(tk.ALPHABET)) == 22
    assert tk.ALPHABET[:20] == "ACDEFGHIKLMNPQRSTVWY"
    assert tk.ALPHABET[tk.GAP_INDEX] == tk.GAP
    assert tk.ALPHABET[tk.MASK_INDEX] == tk.MASK


def test_tokenize_untokenize_round_trips_every_token():
    indices = tk.tokenize(tk.ALPHABET)
    assert indices.shape == (22,)
    assert indices.tolist() == list(range(22))
    assert tk.untokenize(indices) == tk.ALPHABET


def test_tokenize_real_sequence():
    sequence = "MQIFVKTL--GK"
    assert tk.untokenize(tk.tokenize(sequence)) == sequence


def test_tokenize_rejects_unknown_character():
    with pytest.raises(ValueError, match="'X'"):
        tk.tokenize("ACDX")


def test_one_hot_shape_and_values():
    encoded = tk.one_hot(tk.tokenize("AC-"))
    assert encoded.shape == (3, 22)
    assert jnp.all(encoded.sum(axis=-1) == 1.0)
    assert encoded[0, 0] == 1.0
    assert encoded[1, 1] == 1.0
    assert encoded[2, tk.GAP_INDEX] == 1.0


def test_one_hot_batched():
    batch = jnp.stack([tk.tokenize("AC"), tk.tokenize("-#")])
    encoded = tk.one_hot(batch)
    assert encoded.shape == (2, 2, 22)
    assert jnp.all(encoded.sum(axis=-1) == 1.0)
    assert encoded[1, 1, tk.MASK_INDEX] == 1.0


def test_one_hot_recovers_indices_via_argmax():
    indices = tk.tokenize(tk.ALPHABET)
    assert jnp.array_equal(tk.one_hot(indices).argmax(axis=-1), indices)
