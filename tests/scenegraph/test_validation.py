import pytest

from src.scenegraph.vocabulary import vocab_sha1
from src.scenegraph.validation import validate_provenance

MAP_SHA1 = "0123456789abcdef0123456789abcdef01234567"   # stands in for a dataset map fingerprint


@pytest.fixture
def processed(toy):
    """The toy graph as a dataset builder would produce it: with its provenance stamped."""
    toy.raw_metadata["canonical_vocab_sha1"] = vocab_sha1()
    toy.raw_metadata["dataset_map_sha1"] = MAP_SHA1
    return toy


def test_graph_with_the_expected_provenance_passes(processed):
    assert validate_provenance(processed, vocab_sha1(), MAP_SHA1) == []


@pytest.mark.parametrize("key", ["canonical_vocab_sha1", "dataset_map_sha1"])
def test_missing_fingerprint_fails(processed, key):
    del processed.raw_metadata[key]
    problems = validate_provenance(processed, vocab_sha1(), MAP_SHA1)
    assert problems == [f"toy: raw_metadata has no {key}"]


def test_graph_from_another_vocabulary_fails(processed):
    problems = validate_provenance(processed, "f" * 40, MAP_SHA1)
    assert len(problems) == 1 and "vocabulary" in problems[0]


def test_graph_from_another_dataset_map_fails(processed):
    problems = validate_provenance(processed, vocab_sha1(), "f" * 40)
    assert len(problems) == 1 and "dataset map" in problems[0]


def test_validation_does_not_rewrite_provenance(processed):
    validate_provenance(processed, "f" * 40, "f" * 40)
    assert processed.raw_metadata["canonical_vocab_sha1"] == vocab_sha1()
    assert processed.raw_metadata["dataset_map_sha1"] == MAP_SHA1
