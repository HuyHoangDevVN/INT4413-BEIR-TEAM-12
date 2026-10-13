"""One quality pass with cached exact Qwen vectors, frozen revision and cooling."""
import argparse
import json
import sys
import time
from pathlib import Path
import torch
import yaml
from lab.models import Dense
from lab.retrieval import VectorIndex
from lab.data import ROOT, DATASETS, load_dataset
from lab.runner import DEFAULTS, PeakMemory, environment, hardware_inventory, redact_error
from lab.cache import atomic_json
from run_paper_baseline import Tee, persist


def run(args):
    folder = Path(args.output).resolve()
    assert folder.parent == ROOT/'runs' and folder.is_dir()
    config={**DEFAULTS, 'dataset':args.dataset, 'retriever':'qwen3', 'dense_model':'qwen3', 'reranker':None,
        'model_revision':'97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3', 'device':'cuda',
        'batch_size':1, 'cpu_threads':1, 'max_length':512, 'batch_rest_seconds':0, 'progress_interval':1000,
        'thermal_control':True, 'cooling_pause_c':73, 'cooling_resume_c':60, 'warmup':0, 'repeats':0}
    (folder/'config.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    atomic_json(folder/'status.json',dict(status='RUNNING',system='qwen3'))
    atomic_json(folder/'command.json',[sys.executable,*sys.argv])
    atomic_json(folder/'environment.json',environment())
    old=sys.stdout; log=(folder/'run.log').open('w',encoding='utf-8'); sys.stdout=Tee(old,log)
    memory=PeakMemory(); memory.thread.start(); started=time.perf_counter()
    try:
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.set_num_threads(1); torch.cuda.reset_peak_memory_stats()
        hardware=hardware_inventory(); hardware.update(device='cuda',gpu=torch.cuda.get_device_name())
        atomic_json(folder/'hardware.json',hardware)
        corpus,queries,qrels,metadata=load_dataset(args.dataset)
        model=Dense(config,'cuda')
        assert model.name=='Qwen/Qwen3-Embedding-0.6B' and model.revision==config['model_revision']
        vectors,ids,cache,elapsed,hit,manifest=model.corpus(corpus,metadata)
        index=VectorIndex(vectors,ids,'numpy',4096)
        metadata.update(dense_model=model.name,model_revision=model.revision,query_prompt=model.prompt,
            precision=str(next(model.model.parameters()).dtype),effective_max_length=512,
            normalization='l2',tie_policy='stable_corpus_order')
        ranking={}; first=time.perf_counter()
        for i,(qid,text) in enumerate(queries.items()):
            rows=index.search(model.encode([text],query=True)[0],1001)
            rows=[r for r in rows if r['document_id']!=qid][:1000]
            ranking[qid]=[dict(r,rank=j) for j,r in enumerate(rows,1)]
            if (i+1)%25==0:print('Retrieved',i+1,'/',len(queries),flush=True)
        torch.cuda.synchronize()
        timing=dict(protocol=dict(quality_only=True,warmup=0,repeats=0,latency_scope='cold_full_retrieval',
            latency_includes_batch_rest_and_thermal_wait=True), cold=dict(total_seconds=time.perf_counter()-started,
            first_pass_queries_seconds=time.perf_counter()-first,model_load_seconds=model.load_time,
            model_download_cache_resolution_seconds=model.download_time,embedding_time_seconds=elapsed,
            embedding_cache_hit=hit),warm={},RAM_peak_bytes=memory.peak,
            VRAM_peak_bytes=torch.cuda.max_memory_allocated(),VRAM_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
            vector_backend=index.backend,index_size_bytes=vectors.nbytes)
        persist(folder,config,metadata,{q:{r['document_id']:r['score'] for r in rows} for q,rows in ranking.items()},
                qrels,timing,rankings_override=ranking)
    except BaseException as error:
        resource=isinstance(error,MemoryError) or 'out of memory' in str(error).lower()
        atomic_json(folder/'status.json',dict(status='NOT_RUN_RESOURCE_LIMIT' if resource else 'EXECUTION_ERROR',
            reason=redact_error(str(error)),system='qwen3'))
        raise
    finally:
        memory.stop.set(); memory.thread.join(); sys.stdout=old; log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=sorted(DATASETS),required=True)
    p.add_argument('--output',required=True); run(p.parse_args())
