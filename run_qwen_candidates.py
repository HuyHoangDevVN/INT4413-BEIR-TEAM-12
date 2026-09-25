"""Qwen reranking at a new candidate depth; frozen upstream dense candidates."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
import yaml
from run_paper_baseline import Tee, persist
from lab.cache import atomic_json, file_digest
from lab.data import ROOT, DATASETS, load_dataset
from lab.models import Reranker, TASK
from lab.runner import DEFAULTS, PeakMemory, environment, hardware_inventory, read_ranking, redact_error

REVISION = 'e61197ed45024b0ed8a2d74b80b4d909f1255473'


def run(args):
    import torch
    folder = Path(args.output).resolve()
    assert folder.parent == ROOT / 'runs' and folder.is_dir()
    config = {**DEFAULTS, 'dataset': args.dataset, 'retriever': 'qwen3', 'reranker': 'qwen3',
              'candidate_run': args.candidate_run, 'candidate_k': args.candidate_k,
              'reranker_revision': REVISION, 'device': 'cuda', 'batch_size': 1,
              'reranker_batch_size': 1, 'cpu_threads': 1, 'max_length': 512, 'reranker_max_length': 512,
              'batch_rest_seconds': 0, 'thermal_control': True, 'cooling_pause_c': 73,
              'cooling_resume_c': 60, 'warmup': 0, 'repeats': 0}
    (folder / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    atomic_json(folder / 'command.json', [sys.executable, *sys.argv])
    atomic_json(folder / 'environment.json', environment())
    atomic_json(folder / 'status.json', dict(status='RUNNING', system='qwen3+qwen3'))
    log = (folder / 'run.log').open('w', encoding='utf-8')
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    memory = PeakMemory(); memory.thread.start()
    started = time.perf_counter(); phase = 'setup'
    try:
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.set_num_threads(1)
        torch.cuda.reset_peak_memory_stats()
        hardware = hardware_inventory()
        hardware.update(device='cuda', gpu=torch.cuda.get_device_name(), cuda_version=torch.version.cuda)
        atomic_json(folder / 'hardware.json', hardware)
        corpus, queries, qrels, metadata = load_dataset(args.dataset)
        source = Path(args.candidate_run)
        source_meta = json.loads((source / 'metadata.json').read_text())
        source_config = yaml.safe_load((source / 'config.yaml').read_text())
        assert source_config['retriever'] == 'qwen3' and source_config['reranker'] is None
        assert json.loads((source / 'status.json').read_text())['status'] == 'VERIFIED'
        assert all(source_meta[k] == metadata[k] for k in ['corpus_checksum','query_checksum','qrels_checksum'])
        assert file_digest(source / 'ranking.tsv') == source_meta['ranking_checksum']
        candidates = read_ranking(source / 'ranking.tsv')
        assert set(candidates) == set(queries)
        phase = 'model_download_or_load'
        model = Reranker(config, 'cuda')
        assert model.revision == REVISION
        metadata.update(reranker_model=model.name, reranker_revision=model.revision,
                        precision=str(next(model.model.parameters()).dtype), effective_max_length=512,
                        candidate_source_run=source.name, candidate_source_checksum=source_meta['ranking_checksum'],
                        model_revision=source_meta['model_revision'], query_prompt=source_meta['query_prompt'],
                        reranker_instruction=TASK, tie_policy='stable_upstream_candidate_rank',
                        prompt_prefix=model.tokenizer.decode(model.prefix), prompt_suffix=model.tokenizer.decode(model.suffix),
                        reranker_scoring=dict(logits_to_keep=1,use_cache=False,probability='yes_over_yes_plus_no'))
        phase = 'inference'; ranking = {}; results = {}; first = time.perf_counter()
        for i, (qid, query) in enumerate(queries.items()):
            prefix = candidates[qid][:args.candidate_k]
            rows = model.rerank(query, prefix, corpus)
            assert {r['document_id'] for r in rows} == {r['document_id'] for r in prefix}
            ranking[qid] = rows
            results[qid] = {r['document_id']: r['score'] for r in rows}
            if (i+1) % 25 == 0:
                print(f'Reranked {i+1}/{len(queries)} queries (k={args.candidate_k})', flush=True)
        torch.cuda.synchronize()
        from evaluation.metrics import evaluate
        before = evaluate({q: rows[:args.candidate_k] for q, rows in candidates.items()}, qrels)
        after = evaluate(ranking, qrels)
        assert abs(before['Recall@100']-after['Recall@100']) < 1e-12
        atomic_json(folder / 'candidate_validation.json', dict(candidate_sets_preserved=True,
                    query_count=len(queries), candidate_k=args.candidate_k,
                    pair_count=sum(len(rows) for rows in ranking.values()), before_topk_metrics=before, after_metrics=after))
        timing = dict(protocol=dict(quality_only=True,warmup=0,repeats=0,latency_scope='cold_reranker_only',
                    latency_includes_batch_rest_and_thermal_wait=True, cooling={k:config[k] for k in
                    ['batch_rest_seconds','cooling_pause_c','cooling_resume_c','thermal_control']}),
                    cold=dict(model_download_cache_resolution_seconds=model.download_time,model_load_seconds=model.load_time,
                              first_pass_queries_seconds=time.perf_counter()-first,total_seconds=time.perf_counter()-started),
                    warm={},RAM_peak_bytes=memory.peak,VRAM_peak_bytes=torch.cuda.max_memory_allocated(),
                    VRAM_reserved_peak_bytes=torch.cuda.max_memory_reserved())
        persist(folder,config,metadata,results,qrels,timing,rankings_override=ranking)
    except BaseException as error:
        log.write(redact_error(traceback.format_exc()))
        status = 'NOT_RUN_RESOURCE_LIMIT' if isinstance(error,MemoryError) or 'out of memory' in str(error).lower() else 'EXECUTION_ERROR'
        atomic_json(folder / 'status.json',dict(status=status,system='qwen3+qwen3',phase=phase,reason=redact_error(str(error))))
        raise
    finally:
        memory.stop.set(); memory.thread.join()
        sys.stdout, sys.stderr = old_stdout, old_stderr; log.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=sorted(DATASETS),required=True)
    p.add_argument('--candidate-k',type=int,choices=[20,50,100],required=True)
    p.add_argument('--candidate-run',required=True)
    p.add_argument('--output',required=True)
    run(p.parse_args())
