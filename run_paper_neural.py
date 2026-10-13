"""Quality-only upstream TAS-B / Anserini-to-MiniLM top-100 experiment."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
import yaml
from run_paper_baseline import SHA, Tee, persist
from beir.datasets.data_loader import GenericDataLoader
from lab.cache import atomic_json, file_digest
from lab.cooling import Cooling
from lab.data import ROOT, DATASETS, load_dataset
from lab.models import checkpoint
from lab.paper_adapters import cooled_encode, cooled_predict
from lab.runner import DEFAULTS, PeakMemory, environment, hardware_inventory


def run(args):
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT / 'upstream/beir', text=True).strip() == SHA
    folder = Path(args.output).resolve()
    assert folder.parent == ROOT / 'runs' and folder.is_dir()
    system = 'paper-tas-b' if args.system == 'tasb' else 'paper-anserini-bm25'
    config = {**DEFAULTS, 'dataset': args.dataset, 'retriever': system,
              'reranker': None if args.system == 'tasb' else 'paper-minilm-ce',
              'candidate_k': 100, 'device': args.device, 'batch_size': 1, 'reranker_batch_size': 1,
              'cpu_threads': args.cpu_threads, 'max_length': 512, 'reranker_max_length': 512,
              'warmup': 0, 'repeats': 0, 'batch_rest_seconds': 0, 'thermal_control': args.device=='cuda',
              'cooling_pause_c': 73, 'cooling_resume_c': 60, 'precision': 'float32',
              'score_function': 'dot', 'upstream_commit': SHA}
    if args.system == 'ce':
        config.update(k1=.9, b=.4, candidate_run=args.candidate_run)
    (folder / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    atomic_json(folder / 'command.json', [sys.executable, *sys.argv])
    atomic_json(folder / 'environment.json', environment())
    atomic_json(folder / 'status.json', dict(status='RUNNING', system=system))
    atomic_json(folder / 'upstream_source_checksums.json', {
        str(p.relative_to(ROOT / 'upstream/beir')): file_digest(p)
        for p in (ROOT / 'upstream/beir/beir').rglob('*.py')})
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log = (folder / 'run.log').open('w', encoding='utf-8')
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    memory = PeakMemory()
    memory.thread.start()
    started = time.perf_counter()
    phase = 'setup'
    try:
        import torch
        assert args.device == 'cpu' or torch.cuda.is_available(), 'CUDA unavailable; refusing silent CPU fallback'
        torch.set_num_threads(args.cpu_threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
        hardware = hardware_inventory()
        hardware.update(device=args.device, gpu=torch.cuda.get_device_name() if args.device=='cuda' else None, cuda_version=torch.version.cuda)
        atomic_json(folder / 'hardware.json', hardware)
        corpus, queries, qrels = GenericDataLoader(str(ROOT / 'data' / args.dataset)).load(split='test')
        local_c, local_q, local_r, metadata = load_dataset(args.dataset)
        assert {k: (d['title'], d['text']) for k, d in corpus.items()} == local_c
        assert queries == local_q and qrels == local_r
        metadata.update(upstream_commit=SHA, precision='torch.float32', effective_max_length=512,
                        paper='2104.08663v4', tie_policy='trec_eval_reverse_document_id')
        phase = 'model_download'
        name = ('sentence-transformers/msmarco-distilbert-base-tas-b' if args.system == 'tasb'
                else 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        pinned_revision = ('b12d9352e776979147078a8975a4885042984fd1' if args.system == 'tasb'
                           else '233902d25c440f23af6f7d6e94d2946bac0bee0a')
        path, revision, download_seconds = checkpoint(name, pinned_revision)
        phase = 'model_load'
        loaded = time.perf_counter()
        gate = Cooling(config, args.device)
        if args.system == 'tasb':
            # Package __init__ imports unrelated TensorFlow models; load the unchanged SBERT file directly.
            source_file = ROOT / 'upstream/beir/beir/retrieval/models/sentence_bert.py'
            spec = importlib.util.spec_from_file_location('paper_upstream_sentence_bert', source_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            SentenceBERT = module.SentenceBERT
            from beir.retrieval.search.dense import DenseRetrievalExactSearch
            from beir.retrieval.evaluation import EvaluateRetrieval
            model = SentenceBERT(path)
            model.q_model.to(args.device).float()
            model.q_model.max_seq_length = 512
            encode = model.q_model.encode
            model.q_model.encode = lambda items, **kwargs: cooled_encode(encode, items, gate, **kwargs)
            metadata.update(dense_model=name, model_revision=revision, query_prompt='', normalization=False)
            load_seconds = time.perf_counter() - loaded
            phase = 'inference'
            retriever = EvaluateRetrieval(DenseRetrievalExactSearch(model, batch_size=1, corpus_chunk_size=50000), score_function='dot')
            results = retriever.retrieve(corpus, queries)
        else:
            from beir.reranking.models.cross_encoder import CrossEncoder
            from beir.reranking import Rerank
            source = Path(args.candidate_run)
            source_meta = json.loads((source / 'metadata.json').read_text())
            assert json.loads((source / 'status.json').read_text())['status'] == 'VERIFIED'
            assert all(source_meta[k] == metadata[k] for k in ['corpus_checksum', 'query_checksum', 'qrels_checksum'])
            assert file_digest(source / 'upstream_results.json') == source_meta['upstream_results_checksum']
            candidates = json.loads((source / 'upstream_results.json').read_text())
            assert set(candidates) == set(queries)
            model = CrossEncoder(path, device=args.device)
            model.model.model.float()
            model.model.max_length = 512
            predict = model.model.predict
            # The 2021 wrapper uses sentences=; modern SentenceTransformers accepts positional inputs.
            model.model.predict = lambda sentences, **kwargs: cooled_predict(predict, sentences, gate)
            metadata.update(reranker_model=name, reranker_revision=revision,
                            candidate_source_run=source.name, candidate_source_checksum=source_meta['upstream_results_checksum'],
                            compatibility_adapter='sentences keyword to positional inputs; batch-one rest')
            load_seconds = time.perf_counter() - loaded
            phase = 'inference'
            reranker = Rerank(model, batch_size=1)
            results = {}
            for i, qid in enumerate(queries):
                scored = reranker.rerank(corpus, {qid: queries[qid]}, {qid: candidates[qid]}, top_k=100)
                expected = dict(sorted(candidates[qid].items(), key=lambda x: x[1], reverse=True)[:100])
                assert set(scored[qid]) == set(expected)
                results[qid] = scored[qid]
                if (i + 1) % 25 == 0:
                    print(f'Reranked {i + 1}/{len(queries)} queries', flush=True)
            atomic_json(folder / 'candidate_validation.json', dict(candidate_sets_preserved=True,
                        query_count=len(queries), candidate_k=100, pair_count=sum(len(r) for r in results.values())))
        if args.device == 'cuda': torch.cuda.synchronize()
        timing = dict(protocol=dict(warmup=0, repeats=0, quality_only=True,
                    latency_scope='cold_full_retrieval' if args.system == 'tasb' else 'cold_reranker_only',
                    latency_includes_batch_rest_and_thermal_wait=args.device=='cuda',
                    cooling={k: config[k] for k in ['batch_rest_seconds', 'cooling_pause_c', 'cooling_resume_c']}),
                    cold=dict(model_download_cache_resolution_seconds=download_seconds, model_load_seconds=load_seconds,
                              total_seconds=time.perf_counter() - started), warm={},
                    RAM_peak_bytes=memory.peak, VRAM_peak_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
                    VRAM_reserved_peak_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else None)
        persist(folder, config, metadata, results, qrels, timing)
    except BaseException as error:
        traceback.print_exc(file=log)
        resource = isinstance(error, MemoryError) or 'out of memory' in str(error).lower()
        status = 'NOT_RUN_RESOURCE_LIMIT' if resource else ('MODEL_DOWNLOAD_FAILURE' if phase == 'model_download' else 'EXECUTION_ERROR')
        atomic_json(folder / 'status.json', dict(status=status, system=system, reason=f'{type(error).__name__}: {error}', phase=phase))
        raise
    finally:
        memory.stop.set()
        memory.thread.join()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    p.add_argument('--system', choices=['tasb', 'ce'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--candidate-run')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--cpu-threads',type=int,default=1)
    run(p.parse_args())
