import pytest
from lab.data import load_dataset, ARGUANA_MISSING
from evaluation.metrics import evaluate


def fixture_data(tmp_path, monkeypatch, missing):
    import lab.data as data
    folder=tmp_path/'arguana'/'qrels'; folder.mkdir(parents=True)
    (folder/'test.tsv').write_text('query-id\tcorpus-id\tscore\nq\t'+missing+'\t1\n',encoding='utf-8')
    monkeypatch.setattr(data,'read_jsonl',lambda folder,name: {'doc': {'title':'','text':'body'}}
        if name=='corpus' else {'q': {'text':'query'}})


def test_known_missing_arguana_keeps_query_and_qrel(tmp_path, monkeypatch):
    missing=sorted(ARGUANA_MISSING)[0]
    fixture_data(tmp_path,monkeypatch,missing)
    corpus,queries,qrels,meta=load_dataset('arguana',data_dir=tmp_path)
    assert list(queries)==['q'] and qrels=={'q':{missing:1}}
    assert meta['known_missing_qrel_documents']==[missing]
    assert evaluate({'q':[]},qrels)['nDCG@10']==0


def test_unlisted_missing_document_still_rejected(tmp_path, monkeypatch):
    fixture_data(tmp_path,monkeypatch,'unreviewed-missing-document')
    with pytest.raises(ValueError,match='Unknown qrel ID'):
        load_dataset('arguana',data_dir=tmp_path)


def test_trec_unjudged_label_is_retained_and_negative_two_rejected(tmp_path, monkeypatch):
    fixture_data(tmp_path,monkeypatch,'doc')
    path=tmp_path/'arguana/qrels/test.tsv'
    path.write_text('query-id\tcorpus-id\tscore\nq\tdoc\t-1\n')
    _,_,qrels,meta=load_dataset('arguana',data_dir=tmp_path)
    assert qrels['q']['doc']==-1 and meta['unjudged_qrels_count']==1
    assert evaluate({'q':['doc']},qrels)['nDCG@10']==0
    path.write_text('query-id\tcorpus-id\tscore\nq\tdoc\t-2\n')
    with pytest.raises(ValueError):load_dataset('arguana',data_dir=tmp_path)
