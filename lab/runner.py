import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
import numpy as np
import psutil
import yaml
from evaluation.metrics import evaluate
from .cache import atomic_json, directory_size, CandidateCache, file_digest
from .data import ROOT, digest, load_dataset
from .retrieval import BM25, VectorIndex, rrf

DEFAULTS = dict(dataset='scifact', split='test', retriever='bm25', dense_model='bge-m3', reranker=None,
                device='auto', batch_size=16, reranker_batch_size=8, max_length=None, reranker_max_length=512,
                top_k=1000, candidate_k=100, rrf_k=60, fusion='rrf', k1=1.5, b=.75, epsilon=.25,
                backend='numpy', search_batch_size=4096, warmup=5, repeats=3, download=False,
                model_revision=None, reranker_revision=None, data_dir=None, seed=42, ignore_identical_ids=True,
                candidate_run=None, cpu_threads=2, batch_rest_seconds=0, cooling_pause_c=73, cooling_resume_c=60)


def system_id(config):
    system = config['retriever']
    if system == 'hybrid':
        system += '+' + config['dense_model'] + '+rrf'
    if config['reranker']:
        system += '+' + config['reranker'] + f'+k{config["candidate_k"]}'
    if config.get('candidate_run'):
        system += '+precomputed'
    return system


class PeakMemory:
    def __init__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        process = psutil.Process()
        while not self.stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self.stop.wait(.01)


def environment():
    packages = {}
    for package in ['numpy', 'torch', 'transformers', 'sentence-transformers', 'rank-bm25',
                    'pytrec-eval-terrier', 'faiss-cpu', 'psutil', 'PyYAML']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in [ROOT, ROOT / 'lab', ROOT / 'evaluation'] for p in folder.glob('*.py')}
    return {'python': sys.version, 'executable': sys.executable, 'packages': packages,
            'git_commit': commit.stdout.strip() if commit.returncode == 0 else None, 'source_checksums': sources}


def hardware_inventory():
    value = {'cpu': platform.processor(), 'RAM_total_bytes': psutil.virtual_memory().total,
             'logical_cpu_count': psutil.cpu_count(), 'detected_gpus': [],
             'gpu': None, 'cuda_version': None, 'device': 'cpu'}
    if os.name == 'nt':
        for kind, key in [('Win32_Processor','cpu_model'), ('Win32_VideoController','detected_gpus')]:
            try:
                result = subprocess.run(['powershell.exe','-NoProfile','-Command',
                                         f'Get-CimInstance {kind} | Select-Object -ExpandProperty Name'],
                                        capture_output=True, text=True, timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                continue
            names = [x.strip() for x in result.stdout.splitlines() if x.strip()]
            if result.returncode == 0 and names:
                value[key] = names[0] if key == 'cpu_model' else names
    return value


def redact_error(text):
    # Avoid persisting signed CDN query strings or accidental auth headers.
    text = re.sub(r'(https?://[^\s?]+)\?[^\s]+', r'\1?[redacted]', text)
    return re.sub(r'(?i)(authorization|token|password|cookie)\s*[:=]\s*[^\s,]+', r'\1=[redacted]', text)


def write_ranking(path, rankings):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['query_id', 'document_id', 'score', 'rank'])
        for q, rows in rankings.items():
            for row in rows:
                writer.writerow([q, row['document_id'], repr(row['score']), row['rank']])


def read_ranking(path):
    rankings = {}
    with Path(path).open(encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            q = row['query_id']
            values = rankings.setdefault(q, [])
            rank = int(row['rank'])
            if rank != len(values) + 1:
                raise ValueError('Non-contiguous rank')
            values.append({'document_id': row['document_id'], 'score': float(row['score']), 'rank': rank})
    coverage_path = Path(path).parent / 'query_coverage.json'
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text(encoding='utf-8'))
        if not set(rankings).issubset(coverage):
            raise ValueError('Ranking contains query outside coverage manifest')
        for query_id, count in coverage.items():
            if type(count) is not int or count < 0 or len(rankings.get(query_id, [])) != count:
                raise ValueError('Ranking row count differs from coverage manifest')
            rankings.setdefault(query_id, [])
    return rankings


def verify_run(folder):
    folder = Path(folder)
    status = json.loads((folder / 'status.json').read_text())['status']
    if status not in ['RUN_SUCCESS', 'VERIFIED']:
        raise ValueError('Only completed successful runs can be verified')
    config = yaml.safe_load((folder / 'config.yaml').read_text())
    corpus, queries, qrels, metadata = load_dataset(config['dataset'], config['split'], config.get('data_dir'))
    stored = json.loads((folder / 'metadata.json').read_text())
    if any(stored[k] != v for k, v in metadata.items()):
        raise ValueError('Dataset checksum changed')
    ranking = read_ranking(folder / 'ranking.tsv')
    if set(ranking) != set(queries):
        raise ValueError('Ranking query coverage mismatch')
    if any(row['document_id'] not in corpus for rows in ranking.values() for row in rows):
        raise ValueError('Unknown ranking document')
    checksum = hashlib.sha256((folder / 'ranking.tsv').read_bytes()).hexdigest()
    if checksum != stored['ranking_checksum']:
        raise ValueError('Ranking checksum mismatch')
    if stored.get('query_coverage_checksum'):
        coverage_checksum = hashlib.sha256((folder / 'query_coverage.json').read_bytes()).hexdigest()
        if coverage_checksum != stored['query_coverage_checksum']:
            raise ValueError('Query coverage checksum mismatch')
    measured = evaluate(ranking, qrels)
    saved = json.loads((folder / 'metrics.json').read_text())
    if measured != saved:
        raise ValueError('Recomputed metric mismatch')
    validation = {'serialization_recomputed': True, 'pytrec_eval': 'unavailable'}
    try:
        import pytrec_eval
        # Strict decreasing synthetic scores preserve explicit rank and tie policy.
        trec = {q: {r['document_id']: float(len(rows) - i) for i, r in enumerate(rows)} for q, rows in ranking.items()}
        output = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.10,100', 'map_cut.100', 'recip_rank'}).evaluate(trec)
        mapping = {'nDCG@10': 'ndcg_cut_10', 'Recall@10': 'recall_10', 'Recall@100': 'recall_100', 'MAP@100': 'map_cut_100'}
        for key, trec_key in mapping.items():
            expected = sum(output[q][trec_key] for q in qrels) / len(qrels)
            if abs(expected - measured[key]) > 1e-9:
                raise ValueError(f'pytrec_eval mismatch: {key}: {expected} vs {measured[key]}')
        # recip_rank on a top-10 truncated run is MRR@10.
        truncated = {q: dict(list(rows.items())[:10]) for q, rows in trec.items()}
        mrr = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'}).evaluate(truncated)
        if abs(sum(v['recip_rank'] for v in mrr.values()) / len(qrels) - measured['MRR@10']) > 1e-9:
            raise ValueError('pytrec_eval MRR mismatch')
        validation['pytrec_eval'] = 'matched_all_five_metrics'
    except ImportError:
        pass
    atomic_json(folder / 'validation.json', validation)
    atomic_json(folder / 'status.json', {'status': 'VERIFIED', 'system': system_id(config), 'reason': None})
    return measured


def run(config):
    config = {**DEFAULTS, **config}
    for key in ['top_k', 'candidate_k', 'batch_size', 'reranker_batch_size', 'search_batch_size', 'repeats', 'cpu_threads']:
        if config[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if config['warmup'] < 0 or config['rrf_k'] < 0 or not 0 <= config['b'] <= 1 or config['k1'] <= 0:
        raise ValueError('Invalid warmup/RRF/BM25 parameter')
    if config['fusion'] != 'rrf':
        raise ValueError('Only rank-based RRF is supported')
    if config['batch_rest_seconds'] < 0 or not 0 < config['cooling_resume_c'] < config['cooling_pause_c']:
        raise ValueError('Invalid cooling/rest parameters')
    folder = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    (folder / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
    atomic_json(folder / 'environment.json', environment())
    hardware = hardware_inventory()
    atomic_json(folder / 'hardware.json', hardware)
    atomic_json(folder / 'status.json', {'status': 'NOT_RUN', 'system': system_id(config)})
    timing = {'protocol': {'warmup': config['warmup'], 'repeats': config['repeats'], 'query_batch_size': 1,
                           'candidate_cache': 'disabled_during_latency_measurement'}, 'cold': {}, 'warm': {},
              'RAM_peak_bytes': None, 'VRAM_peak_bytes': None, 'VRAM_reserved_peak_bytes': None}
    memory = PeakMemory()
    if config['batch_rest_seconds']:
        timing['protocol']['cooling'] = {k: config[k] for k in ['batch_rest_seconds','cooling_pause_c','cooling_resume_c']}
        timing['protocol']['latency_includes_batch_rest_and_thermal_wait'] = True
    memory.thread.start()
    started = time.perf_counter()
    neural = config['retriever'] != 'bm25' or config['reranker'] is not None
    device = 'cpu'
    phase = 'dataset'
    def sync():
        if neural and device == 'cuda':
            import torch
            torch.cuda.synchronize()
    try:
        with (folder / 'run.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            np.random.seed(config['seed'])
            t = time.perf_counter()
            corpus, queries, qrels, metadata = load_dataset(config['dataset'], config['split'], config['data_dir'], config['download'])
            timing['cold']['dataset_load_download_seconds'] = time.perf_counter() - t
            precomputed = None
            if config['candidate_run']:
                if not config['reranker']:
                    raise ValueError('candidate_run requires a reranker')
                source = Path(config['candidate_run']).resolve()
                verify_run(source)
                source_config = yaml.safe_load((source / 'config.yaml').read_text())
                source_metadata = json.loads((source / 'metadata.json').read_text())
                if source_config['reranker'] or any(source_metadata[k] != v for k,v in metadata.items()):
                    raise ValueError('Candidate source must be an unreranked matching dataset/split')
                if source_config['retriever'] != config['retriever']:
                    raise ValueError('Candidate source retriever mismatch')
                precomputed = read_ranking(source / 'ranking.tsv')
                metadata.update(candidate_source_run=str(source), candidate_source_checksum=source_metadata['ranking_checksum'])
                timing['protocol'].update(candidate_cache='precomputed_upstream_run', latency_scope='reranker_only')
            if neural:
                phase = 'runtime'
                import torch
                from .models import resolve_device, Dense, Reranker
                torch.set_num_threads(config['cpu_threads'])
                hardware['torch_cpu_threads'] = torch.get_num_threads()
                device = resolve_device(config['device'])
                hardware['requested_device'] = config['device']
                config['device'] = device
                (folder / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
                torch.manual_seed(config['seed'])
                if device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                    hardware['gpu'] = torch.cuda.get_device_name(0)
                    hardware['VRAM_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
                hardware.update(device=device, cuda_version=torch.version.cuda)
                print(f'Device: {device}; CPU neural inference may be slow.')
            bm25 = None
            dense = None
            reranker = None
            cache_folder = None
            if precomputed is None and config['retriever'] in ['bm25', 'hybrid']:
                t = time.perf_counter()
                index_dir = ROOT / 'cache' / 'indexes' / 'bm25' / config['dataset']
                manifest = {**metadata, 'k1': config['k1'], 'b': config['b'], 'epsilon': config['epsilon'], 'tokenizer': '[a-z0-9]+'}
                bm25 = None
                try:
                    cached = json.loads((index_dir / 'index.json').read_text())
                    if cached['manifest'] == manifest and cached['ids'] == list(corpus) and cached.get('checksum') == digest(cached['payload']):
                        cached.update(cached['payload'])
                        from collections import defaultdict
                        bm25 = BM25.__new__(BM25)
                        bm25.ids, bm25.k1, bm25.b = cached['ids'], config['k1'], config['b']
                        bm25.postings = defaultdict(list, cached['postings'])
                        bm25.lengths = np.asarray(cached['lengths'])
                        bm25.idf, bm25.avgdl = cached['idf'], cached['avgdl']
                except (OSError, ValueError, KeyError):
                    pass
                timing['cold']['bm25_index_cache_hit'] = bm25 is not None
                if bm25 is None:
                    bm25 = BM25(corpus, config['k1'], config['b'], config['epsilon'])
                    payload = {'postings': dict(bm25.postings), 'lengths': bm25.lengths.tolist(), 'idf': bm25.idf, 'avgdl': bm25.avgdl}
                    atomic_json(index_dir / 'index.json', {'manifest': manifest, 'ids': bm25.ids, 'payload': payload, 'checksum': digest(payload)})
                timing['cold']['bm25_index_seconds'] = time.perf_counter() - t
                timing['index_size_bytes'] = directory_size(index_dir)
            if precomputed is None and config['retriever'] != 'bm25':
                phase = 'model_load'
                dense_config = dict(config)
                if config['retriever'] != 'hybrid':
                    dense_config['dense_model'] = config['retriever']
                dense = Dense(dense_config, device)
                config['max_length'] = dense.config['max_length']
                config['model_revision'] = dense.revision
                (folder / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
                phase = 'embedding'
                matrix, ids, cache_folder, elapsed, hit, manifest = dense.corpus(corpus, metadata)
                metadata.update(dense_model=dense.name, model_revision=dense.revision, query_prompt=dense.prompt,
                                effective_max_length=dense.config['max_length'], precision=str(dense.model[0].auto_model.dtype))
                timing['cold'].update(model_load_seconds=dense.load_time, model_download_cache_resolution_seconds=dense.download_time, embedding_cache_seconds=elapsed,
                                      embedding_time_seconds=0.0 if hit else elapsed, embedding_cache_hit=hit)
                t = time.perf_counter()
                index_dir = ROOT / 'cache' / 'indexes' / dense.name.replace('/', '--') / config['dataset']
                index = None
                index_manifest = {**manifest, 'backend': 'faiss_flat_exact', 'search_type': 'exact'}
                if config['backend'] == 'faiss':
                    try:
                        import faiss
                        stored = json.loads((index_dir / 'manifest.json').read_text())
                        if all(stored.get(k) == v for k,v in index_manifest.items()) and stored['file_checksum'] == file_digest(index_dir / 'index.faiss'):
                            loaded = faiss.read_index(str(index_dir / 'index.faiss'))
                            if loaded.ntotal == len(ids) and loaded.d == matrix.shape[1]:
                                index = VectorIndex(matrix, ids, 'numpy', config['search_batch_size'])
                                index.index, index.backend = loaded, 'faiss_flat_exact'
                    except (ImportError, OSError, ValueError, KeyError, RuntimeError):
                        pass
                timing['cold']['vector_index_cache_hit'] = index is not None
                if index is None:
                    index = VectorIndex(matrix, ids, config['backend'], config['search_batch_size'])
                timing['cold']['vector_index_seconds'] = time.perf_counter() - t
                timing['vector_backend'] = index.backend
                index_dir.mkdir(parents=True, exist_ok=True)
                if index.index is not None:
                    import faiss
                    if not timing['cold']['vector_index_cache_hit']:
                        temp_index = index_dir / ('index.' + uuid.uuid4().hex + '.tmp')
                        faiss.write_index(index.index, str(temp_index))
                        os.replace(temp_index, index_dir / 'index.faiss')
                    index_manifest['file_checksum'] = file_digest(index_dir / 'index.faiss')
                else:
                    index_manifest['backend'] = index.backend
                atomic_json(index_dir / 'manifest.json', index_manifest)
                timing['index_size_bytes'] = timing.get('index_size_bytes', 0) + (directory_size(index_dir) if index.index is not None else matrix.nbytes)
                timing['embedding_cache_size_bytes'] = directory_size(cache_folder)
            if config['reranker']:
                phase = 'model_load'
                reranker = Reranker(config, device)
                config['reranker_revision'] = reranker.revision
                (folder / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
                metadata.update(reranker_model=reranker.name, reranker_revision=reranker.revision)
                if 'Qwen3' in reranker.name:
                    metadata['reranker_scoring'] = {'logits_to_keep': 1, 'use_cache': False, 'probability': 'yes_over_yes_plus_no'}
                timing['cold']['reranker_load_seconds'] = reranker.load_time
                timing['cold']['reranker_download_cache_resolution_seconds'] = reranker.download_time
            def retrieve(qid, text):
                if precomputed is not None:
                    return precomputed[qid]
                depth = config['top_k'] + int(config['ignore_identical_ids'])
                def filter_self(rows):
                    rows = [r for r in rows if not config['ignore_identical_ids'] or r['document_id'] != qid][:config['top_k']]
                    return [dict(r, rank=i) for i,r in enumerate(rows,1)]
                lexical = filter_self(bm25.search(text, depth)) if bm25 else None
                vector = filter_self(index.search(dense.encode([text], query=True)[0], depth)) if dense else None
                rows = rrf([lexical, vector], config['rrf_k'], config['top_k']) if config['retriever'] == 'hybrid' else (lexical if lexical is not None else vector)
                return rows
            def search(qid, text):
                candidates = retrieve(qid, text)
                return reranker.rerank(text, candidates[:config['candidate_k']], corpus) if reranker else candidates
            ranking = {}
            phase = 'inference'
            candidate_manifest = {**{k:v for k,v in metadata.items() if not k.startswith('reranker')},
                                  'retrieval_config': {k:config[k] for k in ['retriever','dense_model','top_k','rrf_k','k1','b','epsilon','max_length','backend','ignore_identical_ids']}}
            candidate_cache = CandidateCache(ROOT / 'cache' / 'candidates' / config['retriever'] / config['dataset'] / (digest(candidate_manifest) + '.json'))
            cached_candidates = precomputed if precomputed is not None else candidate_cache.load(candidate_manifest, queries, corpus)
            timing['cold']['candidate_cache_hit'] = cached_candidates is not None
            generated_candidates = {}
            t = time.perf_counter()
            for query_number, (qid, text) in enumerate(queries.items(), 1):
                candidates = cached_candidates[qid] if cached_candidates is not None else retrieve(qid, text)
                generated_candidates[qid] = candidates
                ranking[qid] = reranker.rerank(text, candidates[:config['candidate_k']], corpus) if reranker else candidates
                if query_number % 50 == 0:
                    print(f'First pass: {query_number}/{len(queries)} queries', flush=True)
            if cached_candidates is None:
                candidate_cache.save(candidate_manifest, generated_candidates)
            sync()
            timing['cold']['first_pass_queries_seconds'] = time.perf_counter() - t
            timing['cold']['total_seconds'] = time.perf_counter() - started
            for qid, text in list(queries.items())[:config['warmup']]:
                search(qid, text)
            latencies = []
            for repeat in range(config['repeats']):
                for qid, text in queries.items():
                    sync()
                    t = time.perf_counter()
                    search(qid, text)
                    sync()
                    latencies.append(time.perf_counter() - t)
                print(f'Warm pass {repeat + 1}/{config["repeats"]} complete', flush=True)
            timing['warm'] = {'query_latency_mean': float(np.mean(latencies)), 'query_latency_p50': float(np.percentile(latencies, 50)),
                              'query_latency_p95': float(np.percentile(latencies, 95)), 'queries_per_second': len(latencies) / sum(latencies),
                              'sample_count': len(latencies)}
            timing['warm']['per_query_latency_seconds'] = latencies
            write_ranking(folder / 'ranking.tsv', ranking)
            metadata['ranking_checksum'] = hashlib.sha256((folder / 'ranking.tsv').read_bytes()).hexdigest()
            atomic_json(folder / 'metadata.json', metadata)
            atomic_json(folder / 'metrics.json', evaluate(ranking, qrels))
            atomic_json(folder / 'status.json', {'status': 'RUN_SUCCESS', 'system': system_id(config)})
            verify_run(folder)
    except Exception as exc:
        status = 'NOT_RUN_RESOURCE_LIMIT' if isinstance(exc, MemoryError) else 'RUN_FAILED'
        if phase == 'model_load' and isinstance(exc, (OSError, ConnectionError)):
            status = 'NOT_RUN_MODEL_DOWNLOAD_FAILED'
        atomic_json(folder / 'status.json', {'status': status, 'system': system_id(config), 'phase': phase,
                                            'reason': redact_error(f'{type(exc).__name__}: {exc}'),
                                            'retry_command': f'{sys.executable} beir_lab.py run --config "{folder / "config.yaml"}"'})
        with (folder / 'run.log').open('a', encoding='utf-8') as f:
            import traceback
            f.write('\n' + redact_error(traceback.format_exc()))
    finally:
        memory.stop.set()
        memory.thread.join()
        timing['RAM_peak_bytes'] = memory.peak
        if neural and device == 'cuda':
            import torch
            timing['VRAM_peak_bytes'] = torch.cuda.max_memory_allocated()
            timing['VRAM_reserved_peak_bytes'] = torch.cuda.max_memory_reserved()
        atomic_json(folder / 'hardware.json', hardware)
        atomic_json(folder / 'timing.json', timing)
    print(folder, json.loads((folder / 'status.json').read_text())['status'], flush=True)
    return folder
