import json
from lab.cache import finalized_artifact_hashes,file_digest


def test_concurrent_jobs_are_not_immutable_history(tmp_path):
    results=tmp_path/'results';results.mkdir();(results/'legacy.json').write_text('{}')
    done=tmp_path/'runs/done';done.mkdir(parents=True)
    (done/'status.json').write_text(json.dumps({'status':'VERIFIED'}))
    (done/'metrics.json').write_text('{}')
    active=tmp_path/'runs/active';active.mkdir()
    (active/'status.json').write_text(json.dumps({'status':'RUNNING'}))
    before=finalized_artifact_hashes(tmp_path)
    assert 'results/legacy.json' in {p.replace('\\','/') for p in before}
    assert not any('active' in p for p in before)
    (active/'status.json').write_text(json.dumps({'status':'VERIFIED'}))
    assert all(file_digest(tmp_path/p)==v for p,v in before.items())
    (done/'metrics.json').write_text('{"changed":true}')
    assert any(file_digest(tmp_path/p)!=v for p,v in before.items())
