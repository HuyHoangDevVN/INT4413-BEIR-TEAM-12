import json
import pytest
from lab.runner import read_ranking


def test_rank_override_preserves_candidate_tie_and_raw_scores(tmp_path, monkeypatch):
    pytest.importorskip('beir', reason='Native bridge fixture uses the isolated paper environment')
    pytest.importorskip('elasticsearch')
    import run_paper_baseline as bridge
    # Equal model scores must retain the upstream candidate order, not reverse doc ID.
    monkeypatch.setattr(bridge, 'verify_run', lambda folder: None)
    rankings = {'q': [dict(document_id='a', score=.5, rank=1), dict(document_id='z', score=.5, rank=2)]}
    results = {'q': {'a': .5, 'z': .5}}
    metadata = {}
    bridge.persist(tmp_path, {'retriever':'qwen3'}, metadata, results,
                   {'q': {'a':2, 'z':0}}, {}, rankings_override=rankings)
    assert read_ranking(tmp_path/'ranking.tsv') == rankings
    assert json.loads((tmp_path/'metrics.json').read_text())['nDCG@10'] == 1.0
    assert json.loads((tmp_path/'upstream_results.json').read_text()) == results
    assert metadata['native_evaluation_uses_rank_preserving_synthetic_scores'] is True


def test_paper_persist_removes_query_document_before_serialization(tmp_path, monkeypatch):
    pytest.importorskip('beir')
    pytest.importorskip('elasticsearch')
    import run_paper_baseline as bridge
    monkeypatch.setattr(bridge, 'verify_run', lambda folder: None)
    metadata = {}
    bridge.persist(tmp_path, {'retriever':'paper-anserini-bm25'}, metadata,
                   {'q': {'q':9., 'relevant':1.}}, {'q': {'relevant':1}}, {})
    assert json.loads((tmp_path/'upstream_results.json').read_text()) == {'q': {'relevant':1.}}
    assert read_ranking(tmp_path/'ranking.tsv')['q'][0]['document_id'] == 'relevant'
    assert metadata['identical_ids_removed'] == 1
