import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from evodiff_torx import tokenizer as tk
from evodiff_torx.denoiser import (
    Denoiser,
    as_deterministic_factor,
    sinusoidal_timestep_features,
)

# Real match-column widths from data/alignments/: YAP1_HUMAN and RL401_YEAST.
YAP1_LEN = 31
RL401_LEN = 71


@pytest.fixture
def model():
    return Denoiser(key=jax.random.key(0))


def random_tokens(key, length, vocab_size=tk.VOCAB_SIZE):
    return jax.random.randint(key, (length,), 0, vocab_size, dtype=jnp.int32)


def test_construction_from_key_gives_finite_params(model):
    leaves = [leaf for leaf in jax.tree.leaves(model) if jnp.issubdtype(leaf.dtype, jnp.floating)]
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert model.vocab_size == tk.VOCAB_SIZE


def test_forward_pass_shape_and_finiteness(model):
    tokens = random_tokens(jax.random.key(1), YAP1_LEN)
    logits = model(tokens, jnp.int32(3))
    assert logits.shape == (YAP1_LEN, tk.VOCAB_SIZE)
    assert logits.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(logits))


@pytest.mark.parametrize("length", [YAP1_LEN, RL401_LEN, 82, 1])
def test_same_instance_handles_every_family_width(model, length):
    """One model serves all family widths -- no reconstruction, no hardcoded L."""
    tokens = random_tokens(jax.random.key(2), length)
    assert model(tokens, jnp.int32(0)).shape == (length, tk.VOCAB_SIZE)


def test_vmap_over_batch_matches_per_sequence_calls(model):
    batch = jax.vmap(lambda k: random_tokens(k, YAP1_LEN))(jax.random.split(jax.random.key(3), 5))
    timesteps = jnp.arange(5, dtype=jnp.int32)

    batched = jax.vmap(model)(batch, timesteps)
    assert batched.shape == (5, YAP1_LEN, tk.VOCAB_SIZE)

    for index in range(5):
        assert jnp.allclose(batched[index], model(batch[index], timesteps[index]), atol=1e-5)


def test_vmap_broadcasts_a_shared_timestep(model):
    batch = jax.vmap(lambda k: random_tokens(k, RL401_LEN))(jax.random.split(jax.random.key(4), 3))
    logits = jax.vmap(model, in_axes=(0, None))(batch, jnp.int32(7))
    assert logits.shape == (3, RL401_LEN, tk.VOCAB_SIZE)


def test_jit_compiles_and_agrees_with_eager(model):
    tokens = random_tokens(jax.random.key(5), YAP1_LEN)
    eager = model(tokens, jnp.int32(2))
    compiled = jax.jit(lambda m, x, t: m(x, t))(model, tokens, jnp.int32(2))
    assert jnp.allclose(eager, compiled, atol=1e-5)


def test_timestep_changes_the_prediction(model):
    tokens = random_tokens(jax.random.key(6), YAP1_LEN)
    assert not jnp.allclose(model(tokens, jnp.int32(0)), model(tokens, jnp.int32(50)))


def test_tokens_change_the_prediction(model):
    tokens = random_tokens(jax.random.key(7), YAP1_LEN)
    perturbed = tokens.at[0].set((tokens[0] + 1) % tk.VOCAB_SIZE)
    assert not jnp.allclose(model(tokens, jnp.int32(1)), model(perturbed, jnp.int32(1)))


def test_gradients_flow_to_every_parameter(model):
    tokens = random_tokens(jax.random.key(8), YAP1_LEN)
    targets = random_tokens(jax.random.key(9), YAP1_LEN)

    def loss(m):
        logits = m(tokens, jnp.int32(4))
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), targets[:, None], axis=-1))

    grads = eqx.filter_grad(loss)(model)
    leaves = [leaf for leaf in jax.tree.leaves(grads) if jnp.issubdtype(leaf.dtype, jnp.floating)]
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(leaf != 0) for leaf in leaves)


def test_position_outside_receptive_field_is_not_influenced(model):
    """Dilated stack has a finite receptive field, so far-away edits must not leak."""
    length = 2 * model.receptive_radius + 21
    tokens = random_tokens(jax.random.key(10), length)
    perturbed = tokens.at[-1].set((tokens[-1] + 1) % tk.VOCAB_SIZE)

    baseline = model(tokens, jnp.int32(1))
    edited = model(perturbed, jnp.int32(1))
    assert jnp.allclose(baseline[0], edited[0], atol=1e-6)
    assert not jnp.allclose(baseline[-1], edited[-1])


def test_sinusoidal_features_are_bounded_and_distinct():
    features = sinusoidal_timestep_features(jnp.int32(5), 64)
    assert features.shape == (64,)
    assert jnp.all(jnp.abs(features) <= 1.0)
    other = sinusoidal_timestep_features(jnp.int32(6), 64)
    assert not jnp.allclose(features, other)


def test_sinusoidal_features_reject_odd_dim():
    with pytest.raises(ValueError, match="even"):
        sinusoidal_timestep_features(jnp.int32(0), 33)


def test_odd_hidden_dim_is_supported():
    model = Denoiser(hidden_dim=33, key=jax.random.key(11))
    tokens = random_tokens(jax.random.key(12), YAP1_LEN)
    assert model(tokens, jnp.int32(0)).shape == (YAP1_LEN, tk.VOCAB_SIZE)


def test_rejects_empty_dilations():
    with pytest.raises(ValueError, match="at least one"):
        Denoiser(dilations=(), key=jax.random.key(13))


def test_deterministic_factor_sample_matches_direct_call(model):
    import torx

    factor = as_deterministic_factor(model, YAP1_LEN)
    assert isinstance(factor, torx.DeterministicFactor)
    assert factor.output_spec.shape == (YAP1_LEN, tk.VOCAB_SIZE)
    assert factor.input_ports["tokens"].shape == (YAP1_LEN,)
    assert factor.input_ports["t"].shape == ()

    tokens = random_tokens(jax.random.key(14), YAP1_LEN)
    t = jnp.int32(9)
    params = factor.init_params(jax.random.key(15))
    sampled = factor.sample(jax.random.key(16), {"tokens": tokens, "t": t}, params)

    assert sampled.shape == factor.output_spec.shape
    assert jnp.array_equal(sampled, model(tokens, t))


def test_deterministic_factor_ports_follow_the_family_width(model):
    factor = as_deterministic_factor(model, RL401_LEN)
    tokens = random_tokens(jax.random.key(17), RL401_LEN)
    sampled = factor.sample(
        jax.random.key(18),
        {"tokens": tokens, "t": jnp.int32(0)},
        factor.init_params(jax.random.key(19)),
    )
    assert sampled.shape == (RL401_LEN, tk.VOCAB_SIZE)
