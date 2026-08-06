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

Later work (Torx forward-corruption sampler, training loop) builds on these.

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
