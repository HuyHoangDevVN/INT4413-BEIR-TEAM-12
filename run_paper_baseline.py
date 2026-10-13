"""Paper-era upstream BEIR entrypoints, with local evidence and thermal controls."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

import requests
import yaml
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from evaluation.metrics import evaluate
from lab.cache import atomic_json
from lab.data import ROOT, DATASETS, load_dataset
from lab.runner import DEFAULTS, environment, read_ranking, verify_run, write_ranking

SHA = '52de18ae6f1ccc7426eb70895c0e43e1552ad493'
IMAGE = 'beir/pyserini-fastapi@sha256:c18cb7f60db4390978d3260bb9e80c797bf11e07e102a407323c7c5c804c83bf'
API = 'http://127.0.0.1:18021'


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def persist(folder, config, metadata, results, qrels, timing, rankings_override=None):
    assert set(results) == set(qrels)
    if config.get('ignore_identical_ids', True):
        removed = sum(q in rows for q, rows in results.items())
        results = {q: {d:s for d,s in rows.items() if d != q} for q, rows in results.items()}
        if removed:
            metadata['identical_ids_removed'] = removed
        if rankings_override is not None:
            rankings_override = {q: [dict(r, rank=i) for i,r in enumerate(
                (r for r in rows if r['document_id'] != q), 1)] for q,rows in rankings_override.items()}
    atomic_json(folder / 'upstream_results.json', results)
    atomic_json(folder / 'query_coverage.json', {qid: len(rows) for qid, rows in results.items()})
    native_input = copy.deepcopy(results)
    if rankings_override is not None:
        native_input = {q: {r['document_id']: float(len(rows)-i) for i, r in enumerate(rows)}
                        for q, rows in rankings_override.items()}
        metadata['native_evaluation_uses_rank_preserving_synthetic_scores'] = True
    native = EvaluateRetrieval.evaluate(qrels, native_input, [10, 100])
    atomic_json(folder / 'upstream_metrics.json', dict(ndcg=native[0], map=native[1], recall=native[2], precision=native[3]))
    # trec_eval resolves equal scores by reverse document ID, retain this explicitly.
    rankings = rankings_override if rankings_override is not None else {q: [dict(document_id=d, score=float(s), rank=i + 1)
                    for i, (d, s) in enumerate(sorted(rows.items(), key=lambda x: (x[1], x[0]), reverse=True))]
                for q, rows in results.items()}
    write_ranking(folder / 'ranking.tsv', rankings)
    metrics = evaluate(rankings, qrels)
    for key, value in [('nDCG@10', native[0]['NDCG@10']), ('Recall@100', native[2]['Recall@100']), ('MAP@100', native[1]['MAP@100'])]:
        assert abs(metrics[key] - value) <= 0.0000051, (key, metrics[key], value)
    atomic_json(folder / 'metrics.json', metrics)
    metadata['ranking_checksum'] = hashlib.sha256((folder / 'ranking.tsv').read_bytes()).hexdigest()
    metadata['query_coverage_checksum'] = hashlib.sha256((folder / 'query_coverage.json').read_bytes()).hexdigest()
    metadata['upstream_results_checksum'] = hashlib.sha256((folder / 'upstream_results.json').read_bytes()).hexdigest()
    atomic_json(folder / 'metadata.json', metadata)
    atomic_json(folder / 'timing.json', timing)
    atomic_json(folder / 'status.json', dict(status='RUN_SUCCESS', system=config['retriever']))
    verify_run(folder)
    print(folder, 'VERIFIED', metrics, flush=True)


def run(args):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'upstream/beir', text=True).strip()
    assert commit == SHA
    assert str(ROOT / 'upstream/beir') in str(sys.modules['beir'].__file__)
    (ROOT / 'runs').mkdir(parents=True, exist_ok=True)
    if getattr(args, 'output', None):
        folder = Path(args.output).resolve()
        assert folder.parent == ROOT / 'runs' and folder.is_dir()
    else:
        folder = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
        folder.mkdir()
    config = {**DEFAULTS, 'dataset': args.dataset, 'retriever': 'paper-anserini-bm25', 'k1': 0.9, 'b': 0.4,
              'device': 'cpu', 'warmup': 0, 'repeats': 0, 'upstream_commit': SHA, 'image': IMAGE,
              'fields': {'contents': 1.0, 'title': 1.0}, 'search_query_batch_size': 20,
              'search_threads': 2, 'java_heap_limit': '1536m'}
    config['container_memory_limit_bytes'] = int(subprocess.check_output(
        ['docker','inspect','--format','{{.HostConfig.Memory}}','beir-paper-20211021'],text=True).strip())
    (folder / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    atomic_json(folder / 'environment.json', environment())
    atomic_json(folder / 'command.json', [sys.executable, *sys.argv])
    atomic_json(folder / 'upstream_source_checksums.json', {
        str(p.relative_to(ROOT / 'upstream/beir')): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / 'upstream/beir/beir').rglob('*.py')})
    atomic_json(folder / 'status.json', dict(status='RUNNING', system=config['retriever']))
    started = time.perf_counter()
    log = (folder / 'run.log').open('w', encoding='utf-8')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    try:
        corpus, queries, qrels = GenericDataLoader(str(ROOT / 'data' / args.dataset)).load(split='test')
        local_c, local_q, local_r, metadata = load_dataset(args.dataset)
        assert {key: (doc['title'], doc['text']) for key, doc in corpus.items()} == local_c
        assert queries == local_q and qrels == local_r
        metadata.update(upstream_commit=SHA, image=IMAGE, tie_policy='trec_eval_reverse_document_id', paper='2104.08663v4')
        metadata.update(search_query_batch_size=20, search_threads=2, container_cpu_limit=2,
                        container_memory_limit_bytes=config['container_memory_limit_bytes'], java_heap_limit='1536m',
                        pyserini_version='0.12.0', gpu_inference=False)
        corpus_file = folder / 'pyserini.jsonl'
        with corpus_file.open('w', encoding='utf-8') as f:
            for doc_id, doc in corpus.items():
                f.write(json.dumps(dict(id=doc_id, title=doc['title'], contents=doc['text']), ensure_ascii=False) + '\n')
        with corpus_file.open('rb') as f:
            response = requests.post(API + '/upload/', files={'file': ('pyserini.jsonl', f)}, timeout=120)
            response.raise_for_status()
        response = requests.get(API + '/index/', params={'index_name': 'beir/' + args.dataset, 'threads': 2}, timeout=3600)
        response.raise_for_status()
        first_start = time.perf_counter()
        results = {}
        query_ids = list(queries)
        for offset in range(0, len(query_ids), 20):
            batch = query_ids[offset:offset + 20]
            payload = dict(queries=[queries[q] for q in batch], qids=batch, bm25={'k1': .9, 'b': .4}, fields={'contents': 1.0, 'title': 1.0})
            response = requests.post(API + '/lexical/batch_search/', params={'k': 1000, 'threads': 2}, json=payload, timeout=120)
            response.raise_for_status()
            results.update(response.json()['results'])
            print('Retrieved', len(results), '/', len(queries), flush=True)
        index_log = subprocess.check_output(['docker', 'logs', 'beir-paper-20211021'], stderr=subprocess.STDOUT)
        (folder / 'container.log').write_bytes(index_log)
        timing = dict(protocol=dict(latency_scope='batch_cpu_docker', warmup=0, repeats=0),
                      cold=dict(first_pass_queries_seconds=time.perf_counter() - first_start, total_seconds=time.perf_counter() - started),
                      warm={}, RAM_peak_bytes=None, VRAM_peak_bytes=None)
        persist(folder, config, metadata, results, qrels, timing)
    except BaseException as error:
        traceback.print_exc(file=log)
        atomic_json(folder / 'status.json', dict(status='EXECUTION_ERROR', system=config['retriever'], reason=f'{type(error).__name__}: {error}'))
        raise
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--output')
    run(parser.parse_args())
