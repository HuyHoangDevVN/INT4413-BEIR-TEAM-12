import json
import math
import numpy as np
import pytest
from evaluation.metrics import evaluate
from lab.cache import EmbeddingCache
from lab.data import load_dataset
from lab.retrieval import BM25, VectorIndex, rrf, tokenize, topk
from lab.runner import read_ranking, write_ranking


def test_graded_metrics_missing_and_zero():
    gold = {'q': {'a': 2, 'b': 1, 'z': 0}, 'missing': {'c': 1}}
    result = evaluate({'q': ['b', 'z', 'a']}, gold)
    assert result['nDCG@10'] == pytest.approx((1 + 2 / math.log2(4)) / (2 + 1 / math.log2(3)) / 2)
    assert result['Recall@10'] == .5
    assert result['MRR@10'] == .5
    assert result['MAP@100'] == pytest.approx((1 + 2/3)/2/2)
    assert result['legacy_exponential_ndcg@10'] != result['nDCG@10']
    with pytest.raises(ValueError):
        evaluate({'q': ['a', 'a']}, gold)
    with pytest.raises(ValueError):
        evaluate({'q': [{'document_id': 'a', 'score': float('nan')}]}, gold)


def test_pytrec_graded():
    trec = pytest.importorskip('pytrec_eval')
    gold = {'q': {'a': 2, 'b': 1, 'z': 0}}
    ours = evaluate({'q': ['b', 'z', 'a']}, gold)
    reference = trec.RelevanceEvaluator(gold, {'ndcg_cut.10', 'recall.100', 'map_cut.100'}).evaluate({'q': {'b': 3., 'z': 2., 'a': 1.}})['q']
    assert ours['nDCG@10'] == pytest.approx(reference['ndcg_cut_10'])
    assert ours['Recall@100'] == reference['recall_100']
    assert ours['MAP@100'] == pytest.approx(reference['map_cut_100'])


@pytest.mark.parametrize('query', ['common', 'rare rare', 'unseen', 'common rare'])
def test_bm25_matches_reference(query):
    from rank_bm25 import BM25Okapi
    corpus = {'a': ('', 'common rare'), 'b': ('', 'common'), 'c': ('', 'common other')}
    reference = BM25Okapi([tokenize(t + ' ' + x) for t, x in corpus.values()])
    actual = BM25(corpus)
    expected = reference.get_scores(tokenize(query))
    assert actual.scores(query) == pytest.approx(expected, abs=1e-12)
    assert actual.search(query, 3) == topk(expected, list(corpus), 3)


def test_rrf_known():
    rows = lambda ids: [{'document_id': d} for d in ids]
    result = rrf([rows(['a', 'b']), rows(['b', 'c'])], k=0)
    assert [r['document_id'] for r in result] == ['b', 'a', 'c']
    assert [r['score'] for r in result] == [1.5, 1., .5]


def test_vector_chunked_and_ties():
    matrix = np.array([[1., 0], [1., 0], [0, 1]], dtype=np.float32)
    index = VectorIndex(matrix, ['a', 'b', 'c'], chunk_size=1)
    assert index.search(np.array([1., 0], dtype=np.float32), 2) == topk(matrix @ np.array([1., 0]), ['a', 'b', 'c'], 2)
    assert len(index.search(np.array([1., 0]), 100)) == 3


def test_faiss_equivalence():
    pytest.importorskip('faiss')
    matrix = np.array([[1., 0], [1., 0], [0, 1]], dtype=np.float32)
    vector = np.array([1., 0], dtype=np.float32)
    assert VectorIndex(matrix, ['a','b','c'], 'faiss').search(vector, 2) == VectorIndex(matrix, ['a','b','c']).search(vector, 2)


def test_cache_validation(tmp_path):
    cache = EmbeddingCache(tmp_path)
    manifest = {'model': 'm', 'revision': 'sha', 'corpus_checksum': 'a', 'dimension': 2}
    cache.save(manifest, ['a', 'b'], np.eye(2, dtype=np.float32))
    assert cache.load(manifest, ['a','b']) is not None
    assert cache.load({**manifest, 'corpus_checksum': 'b'}, ['a','b']) is None
    assert cache.load(manifest, ['b','a']) is None
    assert cache.load({**manifest, 'revision':'other'}, ['a','b']) is None


def test_serialization_and_rerank_recall(tmp_path):
    rows = [{'document_id': str(i), 'score': float(100-i), 'rank':i+1} for i in range(100)]
    write_ranking(tmp_path/'ranking.tsv', {'q':rows})
    assert read_ranking(tmp_path/'ranking.tsv') == {'q': rows}
    gold = {'q': {'2': 1, '3':1, 'missing':1}}
    reversed_rows = [dict(row, rank=i+1) for i,row in enumerate(reversed(rows))]
    assert evaluate({'q':rows}, gold)['Recall@100'] == evaluate({'q':reversed_rows}, gold)['Recall@100']


def test_loader_offline():
    for name, docs in [('scifact',5183),('nfcorpus',3633)]:
        corpus, queries, qrels, metadata = load_dataset(name)
        assert len(corpus) == docs
        assert set(queries) == set(qrels)
        assert len(metadata['corpus_checksum']) == 64


def test_loader_rejects_duplicate(tmp_path):
    folder = tmp_path/'scifact'
    folder.mkdir()
    (folder/'corpus.jsonl').write_text('{"_id":"a","text":"x"}\n{"_id":"a","text":"y"}\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_dataset('scifact', data_dir=tmp_path)


def test_candidate_cache_checksums(tmp_path):
    from lab.cache import CandidateCache
    cache = CandidateCache(tmp_path/'candidates.json')
    ranking = {'q':[{'document_id':'a','score':1.,'rank':1}]}
    cache.save({'revision':'sha'},ranking)
    assert cache.load({'revision':'sha'}, {'q':'text'}, {'a':('','doc')}) == ranking
    assert cache.load({'revision':'new'}, {'q':'text'}, {'a':('','doc')}) is None
    value=json.loads((tmp_path/'candidates.json').read_text())
    value['ranking']['q'][0]['score']=2
    (tmp_path/'candidates.json').write_text(json.dumps(value))
    assert cache.load({'revision':'sha'}, {'q':'text'}, {'a':('','doc')}) is None


def test_reranker_adapter_candidate_mapping():
    from lab.models import Reranker
    adapter=Reranker.__new__(Reranker)
    adapter.cooling=None
    adapter.config={'reranker_batch_size':2}
    adapter.device='cpu'
    calls=[]
    def fake_predict(pairs):
        calls.extend(pairs)
        return np.array([float(d.split()[-1]) for _,d in pairs])
    adapter.predict=fake_predict
    corpus={str(i):('',f'doc {i}') for i in range(101)}
    candidates=[{'document_id':str(i),'rank':i+1,'score':0.} for i in range(100)]
    result=adapter.rerank('query', candidates, corpus)
    assert len(calls)==100
    assert {r['document_id'] for r in result}=={r['document_id'] for r in candidates}
    assert result[0]['document_id']=='99'
    assert '100' not in {r['document_id'] for r in result}
    gold={'q':{'2':1,'3':1,'100':1}}
    assert evaluate({'q':result},gold)['Recall@100']==evaluate({'q':candidates},gold)['Recall@100']


def test_oom_retries_current_batch():
    import torch
    from lab.models import batched
    attempts=[]
    def encode(items):
        attempts.append(list(items))
        if len(items)>1:
            raise torch.OutOfMemoryError('fixture')
        return [[items[0]]]
    assert batched([0,1,2],3,encode,'cpu').tolist()==[[0],[1],[2]]
    assert attempts[0]==[0,1,2]


def test_numpy_faiss_random():
    pytest.importorskip('faiss')
    rng=np.random.default_rng(42)
    matrix=rng.normal(size=(37,8)).astype(np.float32)
    ids=[str(i) for i in range(37)]
    vector=rng.normal(size=8).astype(np.float32)
    expected=topk(matrix@vector,ids,17)
    for backend in ['numpy','faiss']:
        rows=VectorIndex(matrix,ids,backend,chunk_size=5).search(vector,17)
        assert [r['document_id'] for r in rows]==[r['document_id'] for r in expected]
        assert [r['score'] for r in rows]==pytest.approx([r['score'] for r in expected],abs=1e-5)


def test_cli_config_precedence():
    # Inspect config resolution without loading a model or writing run files.
    import beir_lab
    from unittest.mock import patch
    captured={}
    class FakeFolder:
        def __truediv__(self, name):
            return self
        def read_text(self):
            return '{"status":"VERIFIED"}'
    def fake_run(config):
        captured.update(config)
        return FakeFolder()
    with patch('sys.argv',['beir_lab.py','run','--config','configs/hybrid.yaml','--candidate-k','20']), patch.object(beir_lab,'run',fake_run):
        beir_lab.main()
    assert captured['candidate_k']==20
    assert captured['retriever']=='hybrid'
    assert captured['dense_model']=='BAAI/bge-m3'


def test_checkpoint_bin_and_windows_serial_download(monkeypatch):
    from types import SimpleNamespace
    import huggingface_hub
    from lab.models import checkpoint
    captured={}
    class API:
        def model_info(self, name, revision=None):
            return SimpleNamespace(sha='immutable-sha',siblings=[SimpleNamespace(rfilename='pytorch_model.bin')])
    def download(name, **kwargs):
        captured.update(kwargs)
        return 'snapshots/immutable-sha'
    monkeypatch.setattr(huggingface_hub,'HfApi',API)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    _, revision, _=checkpoint('BAAI/bge-m3')
    assert revision=='immutable-sha'
    assert '*.bin' in captured['allow_patterns']
    assert captured['max_workers']==1
    assert captured['revision']=='immutable-sha'


def test_dense_query_document_prompt_separation():
    from lab.models import Dense
    adapter=Dense.__new__(Dense)
    adapter.cooling=None
    adapter.prompt='Instruct: fixture\nQuery:'
    adapter.device='cpu'
    adapter.config={'batch_size':2}
    prompts=[]
    class Model:
        def encode(self, texts, **kwargs):
            prompts.append(kwargs['prompt'])
            return np.tile(np.array([[1.,0.]],dtype=np.float32),(len(texts),1))
    adapter.model=Model()
    adapter.encode(['query'],query=True)
    adapter.encode(['document'])
    assert prompts==['Instruct: fixture\nQuery:','']


def test_loader_gzip_and_zero_qrel(tmp_path):
    import gzip
    folder=tmp_path/'scifact'
    (folder/'qrels').mkdir(parents=True)
    for name,row in [('corpus',{'_id':'d','title':'','text':'document'}),('queries',{'_id':'q','text':'query'})]:
        with gzip.open(folder/(name+'.jsonl.gz'),'wt',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
    (folder/'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\nq\td\t0\n')
    corpus,queries,qrels,_=load_dataset('scifact',data_dir=tmp_path)
    assert corpus['d']==('','document') and queries=={'q':'query'}
    assert evaluate({'q':['d']},qrels)['Recall@100']==0


def test_qwen_yes_no_template_scoring():
    import torch
    from lab.models import Reranker
    adapter=Reranker.__new__(Reranker)
    adapter.cooling=None
    adapter.name='Qwen/Qwen3-Reranker-0.6B'
    adapter.device='cpu'
    adapter.config={'reranker_max_length':20}
    adapter.prefix=[8,9]
    adapter.suffix=[10]
    adapter.no,adapter.yes=0,1
    captured={}
    class Inputs(dict):
        def to(self,device):
            return self
    class Tokenizer:
        def __call__(self,texts,**kwargs):
            captured['texts']=texts
            captured['limit']=kwargs['max_length']
            return {'input_ids':[[2,3] for t in texts]}
        def pad(self,inputs,**kwargs):
            captured['ids']=inputs['input_ids']
            return Inputs(input_ids=torch.tensor(inputs['input_ids']))
    class Model:
        def __call__(self,**inputs):
            from types import SimpleNamespace
            assert inputs['logits_to_keep']==1 and inputs['use_cache'] is False
            logits=torch.zeros((len(inputs['input_ids']),5,2))
            logits[:,-1,1]=math.log(3)
            return SimpleNamespace(logits=logits)
    adapter.tokenizer,adapter.model=Tokenizer(),Model()
    scores=adapter.predict([('query','document')])
    assert scores.tolist()==pytest.approx([.75])
    assert captured['ids']==[[8,9,2,3,10]]
    assert captured['limit']==17
    assert '<Query>: query' in captured['texts'][0]
    assert '<Document>: document' in captured['texts'][0]


def test_legacy_import_does_not_probe_network(monkeypatch):
    import requests
    import importlib
    import utils
    def forbidden(*a,**k):
        raise AssertionError('Import must not access network')
    monkeypatch.setattr(requests,'head',forbidden)
    monkeypatch.setattr(requests,'get',forbidden)
    importlib.reload(utils)
    assert utils.SSL_VERIFY is not False


def test_historical_results_preserved():
    from lab.data import ROOT
    for original in (ROOT/'legacy/v1/results').glob('*.json'):
        assert original.read_bytes()==(ROOT/'results'/original.name).read_bytes()


def test_precomputed_reranking_is_sequential_and_scope_labelled(tmp_path, monkeypatch):
    import hashlib
    import yaml
    import lab.runner as runner
    import lab.models as models
    from lab.cache import atomic_json
    folder=tmp_path/'data/scifact'
    (folder/'qrels').mkdir(parents=True)
    (folder/'corpus.jsonl').write_text('{"_id":"d1","text":"alpha"}\n{"_id":"d2","text":"beta"}\n')
    (folder/'queries.jsonl').write_text('{"_id":"q1","text":"alpha"}\n')
    (folder/'qrels/test.tsv').write_text('query-id\tcorpus-id\tscore\nq1\td1\t2\n')
    source=tmp_path/'source'
    source.mkdir()
    config={**runner.DEFAULTS,'dataset':'scifact','data_dir':str(tmp_path/'data'),'top_k':2}
    (source/'config.yaml').write_text(yaml.safe_dump(config))
    _,_,qrels,metadata=load_dataset('scifact',data_dir=tmp_path/'data')
    ranking={'q1':[{'document_id':'d1','score':2.,'rank':1},{'document_id':'d2','score':1.,'rank':2}]}
    write_ranking(source/'ranking.tsv',ranking)
    metadata['ranking_checksum']=hashlib.sha256((source/'ranking.tsv').read_bytes()).hexdigest()
    atomic_json(source/'metadata.json',metadata)
    atomic_json(source/'metrics.json',evaluate(ranking,qrels))
    atomic_json(source/'status.json',{'status':'RUN_SUCCESS'})
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'environment',lambda:{})
    monkeypatch.setattr(runner,'hardware_inventory',lambda:{'device':'cpu'})
    def forbidden(*args,**kwargs):
        raise AssertionError('Retrieval models must not load during precomputed reranking')
    monkeypatch.setattr(runner,'BM25',forbidden)
    monkeypatch.setattr(models,'Dense',forbidden)
    class Reranker:
        name='fixture'; revision='fixture-revision'; load_time=0.; download_time=0.
        def __init__(self,*args):
            pass
        def rerank(self,query,candidates,corpus):
            return [dict(r,rank=i,score=float(len(candidates)-i)) for i,r in enumerate(reversed(candidates),1)]
    monkeypatch.setattr(models,'Reranker',Reranker)
    result=runner.run({**config,'candidate_run':str(source),'reranker':'minilm','candidate_k':2,
                       'device':'cpu','warmup':1,'repeats':1})
    assert json.loads((result/'status.json').read_text())['status']=='VERIFIED'
    assert json.loads((result/'timing.json').read_text())['protocol']['latency_scope']=='reranker_only'
    assert {r['document_id'] for r in read_ranking(result/'ranking.tsv')['q1']}=={'d1','d2'}
