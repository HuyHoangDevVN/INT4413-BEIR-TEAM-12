import json

import pytest
import run_remaining_cpu_baselines as worker


def test_candidate_source_must_match_completed_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    run = tmp_path / 'runs/source'
    run.mkdir(parents=True)
    session = tmp_path / 'reports/session'
    session.mkdir(parents=True)
    (session / 'session.json').write_text(json.dumps([
        {'dataset': 'fiqa', 'system': 'bm25', 'status': 'VERIFIED', 'run': str(run)}]))
    (run / 'status.json').write_text(json.dumps({'status': 'VERIFIED'}))
    (run / 'metadata.json').write_text(json.dumps({'dataset': 'fiqa'}))
    assert worker.candidate_runs(session, ['fiqa']) == {'fiqa': str(run)}
    (run / 'status.json').write_text(json.dumps({'status': 'RUNNING'}))
    with pytest.raises(ValueError, match='Invalid candidate artifact'):
        worker.candidate_runs(session, ['fiqa'])


def test_missing_candidate_source_fails_before_model_start(tmp_path):
    (tmp_path / 'session.json').write_text('[]')
    with pytest.raises(ValueError, match='Expected one VERIFIED BM25'):
        worker.candidate_runs(tmp_path, ['fiqa'])
