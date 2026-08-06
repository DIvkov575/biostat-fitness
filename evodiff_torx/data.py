"""Parser for `.a2m` alignment files.

In `.a2m`, each record is a `>` header line followed by one or more sequence
lines that continue until the next `>`. Uppercase letters and `-` are match
columns (the fixed-width alignment); lowercase letters and `.` are insert
columns and are discarded. Every record in a family yields the same match-column
width -- verified across all 17 files in `data/alignments/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .tokenizer import AMINO_ACIDS, GAP

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "alignments"

# The mask token is model-side only and never appears in alignment files.
_ALLOWED = frozenset(AMINO_ACIDS + GAP)


def _iter_records(path: Path) -> Iterator[str]:
    """Yield one joined sequence string per record, inserts still included."""
    chunks: list[str] = []
    started = False
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if started:
                    yield "".join(chunks)
                chunks = []
                started = True
            else:
                chunks.append(line)
    if started:
        yield "".join(chunks)


def match_columns(sequence: str) -> str:
    """Strip insert columns (lowercase letters and `.`), keeping uppercase and `-`."""
    return "".join(char for char in sequence if char.isupper() or char == "-")


def resolve_path(family_or_path: str | Path, data_dir: Path | None = None) -> Path:
    """Resolve a family name to `{data_dir}/{family}.a2m`, or pass a path through."""
    candidate = Path(family_or_path)
    if candidate.suffix == ".a2m":
        return candidate
    return (data_dir or DEFAULT_DATA_DIR) / f"{candidate.name}.a2m"


def load_alignment(
    family_or_path: str | Path,
    data_dir: Path | None = None,
    keep_ambiguous: bool = False,
) -> list[str]:
    """Load a family's match-column sequences as a list of equal-length strings.

    Real files contain a small fraction of IUPAC ambiguity codes (`X`, `B`, `Z`)
    that the 22-token alphabet cannot represent; those records are dropped unless
    `keep_ambiguous` is set, in which case the caller must handle them before
    tokenizing.
    """
    path = resolve_path(family_or_path, data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no alignment file at {path}")

    sequences = []
    for record in _iter_records(path):
        sequence = match_columns(record)
        if keep_ambiguous or _ALLOWED.issuperset(sequence):
            sequences.append(sequence)

    if not sequences:
        raise ValueError(f"{path} produced no sequences")

    widths = {len(sequence) for sequence in sequences}
    if len(widths) != 1:
        raise ValueError(
            f"{path}: match columns are not fixed width, found widths {sorted(widths)}"
        )
    return sequences


def available_families(data_dir: Path | None = None) -> list[str]:
    """List family names with a `.a2m` file in `data_dir`."""
    return sorted(path.stem for path in (data_dir or DEFAULT_DATA_DIR).glob("*.a2m"))
