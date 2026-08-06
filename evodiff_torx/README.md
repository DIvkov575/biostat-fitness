# evodiff_torx

A standalone, small-scale D3PM-style discrete diffusion model over protein
sequences (algorithmically inspired by Microsoft's EvoDiff), built on Extropic's
[`torx`](https://github.com/DIvkov575/torx) JAX library for probabilistic
circuits. This is a from-scratch reimplementation of the *algorithm*, not a port
of released EvoDiff checkpoints, and it does not depend on the `evodiff` or
`sequence_models` packages.

This directory is **unrelated to the rest of this repo**, which does protein
fitness-landscape GP regression over ESM2 embeddings. The only thing shared is
the alignment data under `data/alignments/`. Dependencies are pinned separately
in `evodiff_torx/requirements.txt`; the repo-root `requirements.txt`
(torch/transformers/gpytorch) is not used here.

## Current contents

- `tokenizer.py` — 22-token alphabet (20 amino acids + gap `-` + mask `#`),
  `tokenize` / `untokenize` / `one_hot`.
- `data.py` — `.a2m` alignment parser returning fixed-width match-column
  sequences as a list of strings.
- `schedule.py` — D3PM uniform-transition noise schedule:
  `uniform_transition_schedule(T)` returns `(q, q_bar)`, each a `(T, K, K)`
  stack of column-stochastic matrices. Corrupts over the 21 real states
  (amino acids + gap), *excluding* the mask token — see `NUM_DIFFUSION_STATES`.

- `corruption.py` — the Torx forward-corruption sampler. `ForwardCorruption(seq_len, num_timesteps)`
  bundles the schedule with a `TiledFactor(PMarkov(...))` gate; `corrupt_batch(key, tokens)`
  returns `(corrupted, timesteps)` for a `(B, L)` batch of original sequences,
  drawing one timestep per sequence as EvoDiff's `D3PMCollater` does.
- `denoiser.py` — reverse-denoising network: dilated 1-D convolutions mapping a
  corrupted sequence plus a timestep to per-position logits.
- `train.py` — the end-to-end training loop. `train_and_evaluate(family, ...)`
  loads a family, trains the denoiser on the reconstruction cross-entropy, and
  returns a dict of accuracies; `--family` runs it from the CLI.
- `benchmark.py` — the multi-family driver. Runs `train_and_evaluate` over
  `FAMILIES` and prints the comparison table below; exits non-zero if any family
  fails to beat its PSSM baseline.

## Training

One family:

```bash
evodiff_torx/.venv/bin/python evodiff_torx/train.py --family YAP1_HUMAN
```

All three families side by side:

```bash
evodiff_torx/.venv/bin/python evodiff_torx/benchmark.py
```

Defaults: 200 diffusion timesteps, 1000 Adam steps at `lr=1e-3`, batch 128, a
4000-sequence training subsample and 400 held-out sequences. Each family trains
in 15–30s on CPU, so the full benchmark is under two minutes.

Two accuracies are reported, both "fraction of held-out positions where the
predicted residue is correct":

- **model** — corrupt the held-out sequences exactly as in training
  (`corrupt_batch`, one random timestep per sequence), then argmax the denoiser's
  logits.
- **PSSM baseline** — each column's most frequent training residue, always, with
  no view of the input.

Measured with the defaults above, as printed by `benchmark.py`:

| family | L | model | PSSM | margin |
|---|---|---|---|---|
| YAP1_HUMAN | 31 | 0.7750 | 0.5292 | +0.2458 |
| RL401_YEAST | 71 | 0.7239 | 0.6298 | +0.0942 |
| PABP_YEAST | 82 | 0.6034 | 0.3794 | +0.2241 |

The model sees the corrupted sequence, so positions noise left untouched are
free — echoing the input scores ~0.54. The margin over the PSSM is what measures
context modeling beyond per-column marginals.

## Setup

```bash
python3 -m venv evodiff_torx/.venv
evodiff_torx/.venv/bin/pip install -r evodiff_torx/requirements.txt
evodiff_torx/.venv/bin/python -m pytest evodiff_torx/tests -q
```

## Alignment data

`data.py` resolves a family name to `data/alignments/{family}.a2m` relative to
the repo root, independent of the working directory. In `.a2m`, uppercase letters
and `-` are match columns; lowercase letters and `.` are insert columns and are
stripped. Sequences may wrap across multiple lines and are joined until the next
`>` header.

All 17 families were verified to have a single match-column width each, e.g.
`YAP1_HUMAN` 31, `RL401_YEAST` 71, `PABP_YEAST` 82. Roughly 0.03%–6% of records
per family contain IUPAC ambiguity codes (`X`, `B`, `Z`) outside the 22-token
alphabet; `load_alignment` drops those records by default, and
`keep_ambiguous=True` returns them for callers that want to handle them.
