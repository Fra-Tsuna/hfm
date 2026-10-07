"""Checks on processed scene graphs.

Used by the data tests and by the scripts that build graphs. Every check returns a list of
problems; an empty list means the graph passes.
"""

from __future__ import annotations

from src.scenegraph.graph import GRAPH_METADATA_KEYS, SceneGraph


def validate_provenance(graph: SceneGraph, vocab_sha1: str, dataset_map_sha1: str) -> list[str]:
    """Check that a processed dataset graph records the vocabulary and mapping we expect.

    The expected fingerprints are passed in by the pipeline that processed the graph, rather
    than read from the current vocabulary: a graph built with an older vocabulary then fails
    as an old graph, instead of having its provenance silently rewritten.
    """
    problems = []
    for key in GRAPH_METADATA_KEYS:
        if key not in graph.raw_metadata:
            problems.append(f"{graph.scene_id}: raw_metadata has no {key}")
    stored_vocab = graph.raw_metadata.get("canonical_vocab_sha1")
    if stored_vocab is not None and stored_vocab != vocab_sha1:
        problems.append(f"{graph.scene_id}: built with vocabulary {stored_vocab}, expected {vocab_sha1}")
    stored_map = graph.raw_metadata.get("dataset_map_sha1")
    if stored_map is not None and stored_map != dataset_map_sha1:
        problems.append(f"{graph.scene_id}: built with dataset map {stored_map}, expected {dataset_map_sha1}")
    return problems
