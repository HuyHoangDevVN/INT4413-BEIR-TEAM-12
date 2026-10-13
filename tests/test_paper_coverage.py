import json
import pytest
from lab.runner import read_ranking, write_ranking


def test_empty_query_coverage_roundtrip_and_corrupt_count(tmp_path):
    rows = {'matched': [dict(document_id='doc', score=1.0, rank=1)], 'no_match': []}
    path = tmp_path / 'ranking.tsv'
    write_ranking(path, rows)
    coverage = tmp_path / 'query_coverage.json'
    coverage.write_text(json.dumps({'matched': 1, 'no_match': 0}), encoding='utf-8')
    assert read_ranking(path) == rows
    coverage.write_text(json.dumps({'matched': 1, 'no_match': 1}), encoding='utf-8')
    with pytest.raises(ValueError, match='row count'):
        read_ranking(path)


def test_paper_reference_only_attaches_to_anserini(tmp_path):
    import yaml
    from generate_report import collect
    for system in ['bm25', 'paper-anserini-bm25', 'paper-tas-b']:
        folder = tmp_path / system
        folder.mkdir()
        (folder / 'config.yaml').write_text(yaml.safe_dump(dict(dataset='scifact', retriever=system, reranker=None)))
        (folder / 'status.json').write_text(json.dumps(dict(status='VERIFIED')))
        (folder / 'metrics.json').write_text(json.dumps({'nDCG@10': .665}))
    rows = {row['system']: row for row in collect(tmp_path)}
    assert rows['bm25']['Paper result'] is None
    assert rows['paper-anserini-bm25']['Paper result'] == .665
    assert rows['paper-tas-b']['Paper result'] == .643
