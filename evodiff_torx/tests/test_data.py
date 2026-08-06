import pytest

from evodiff_torx import data
from evodiff_torx import tokenizer as tk

# Widths empirically confirmed by scanning every record of each file.
EXPECTED_WIDTHS = {"YAP1_HUMAN": 31, "RL401_YEAST": 71, "PABP_YEAST": 82}


def test_match_columns_strips_inserts():
    assert data.match_columns("dvpLPAGWEMAK") == "LPAGWEMAK"
    assert data.match_columns("...LPEGW-..") == "LPEGW-"
    assert data.match_columns("mQIFVKlrgg") == "QIFVK"


def test_wrapped_record_joins_to_fixed_width(tmp_path):
    path = tmp_path / "TOY.a2m"
    path.write_text(
        ">first/1-6\naACDEF-g\n"
        ">second/1-6\n..ACD\nEF-\n"
        ">third/1-6\nACDEF-\n"
    )
    assert data.load_alignment("TOY", data_dir=tmp_path) == ["ACDEF-"] * 3


def test_ragged_widths_raise(tmp_path):
    path = tmp_path / "BAD.a2m"
    path.write_text(">a\nACDEF\n>b\nACD\n")
    with pytest.raises(ValueError, match="not fixed width"):
        data.load_alignment("BAD", data_dir=tmp_path)


def test_ambiguous_records_dropped_by_default(tmp_path):
    path = tmp_path / "AMB.a2m"
    path.write_text(">a\nACDEF-\n>b\nACDEX-\n>c\nACDEB-\n")
    assert data.load_alignment("AMB", data_dir=tmp_path) == ["ACDEF-"]
    assert len(data.load_alignment("AMB", data_dir=tmp_path, keep_ambiguous=True)) == 3


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_alignment("NOPE", data_dir=tmp_path)


def test_available_families_includes_known_families():
    families = data.available_families()
    assert set(EXPECTED_WIDTHS).issubset(families)


@pytest.mark.parametrize("family,width", sorted(EXPECTED_WIDTHS.items()))
def test_real_family_parses_to_expected_width(family, width):
    sequences = data.load_alignment(family)
    assert len(sequences) > 1000
    assert {len(sequence) for sequence in sequences} == {width}


def test_family_name_and_path_resolve_identically():
    by_name = data.resolve_path("YAP1_HUMAN")
    by_path = data.resolve_path(data.DEFAULT_DATA_DIR / "YAP1_HUMAN.a2m")
    assert by_name == by_path
    assert by_name.is_file()


def test_parsed_sequences_tokenize():
    sequences = data.load_alignment("YAP1_HUMAN")
    indices = tk.tokenize(sequences[0])
    assert indices.shape == (EXPECTED_WIDTHS["YAP1_HUMAN"],)
    assert tk.untokenize(indices) == sequences[0]
    assert tk.one_hot(indices).shape == (EXPECTED_WIDTHS["YAP1_HUMAN"], tk.VOCAB_SIZE)
