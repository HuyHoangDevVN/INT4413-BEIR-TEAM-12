"""Explicit FP32 CPU baseline work while the shared GPU is unavailable."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from datetime import datetime,timezone
import yaml
from lab.data import ROOT
from lab.cache import atomic_json,file_digest,finalized_artifact_hashes
from lab.process import stop_owned


def candidate_runs(source_session, datasets):
    """Require completed Anserini candidates from the explicitly selected session."""
    entries = json.loads((source_session / 'session.json').read_text(encoding='utf-8'))
    sources = {}
    for dataset in datasets:
        matches = [e for e in entries if e['dataset'] == dataset and e['system'] == 'bm25'
                   and e.get('status') == 'VERIFIED']
        if len(matches) != 1:
            raise ValueError(f'Expected one VERIFIED BM25 source for {dataset}')
        folder = Path(matches[0]['run']).resolve()
        if folder.parent != ROOT / 'runs':
            raise ValueError('Candidate run must belong to this project')
        status = json.loads((folder / 'status.json').read_text(encoding='utf-8'))
        metadata = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
        if status.get('status') != 'VERIFIED' or metadata.get('dataset') != dataset:
            raise ValueError(f'Invalid candidate artifact for {dataset}')
        sources[dataset] = str(folder)
    return sources


def main(args):
    if os.name != 'posix':
        raise RuntimeError('This CPU server supervisor requires Linux')
    import fcntl
    # Validate input before allocating new runs or starting any model.
    sources = candidate_runs(args.source_session.resolve(), args.datasets)
    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    (ROOT / 'runs').mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'reports/remaining_cpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior=finalized_artifact_hashes(ROOT)
    session=ROOT/'reports'/('remaining_cpu_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'));session.mkdir()
    jobs=[]
    atomic_json(session / 'source_session.json', dict(path=str(args.source_session.resolve()),
                datasets=args.datasets, cpu_threads=args.cpu_threads, per_run_seconds=args.per_run_seconds))
    try:
        for dataset in args.datasets:
            for system in ['tasb','ce']:
                folder=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir()
                command=[sys.executable,str(ROOT/'run_paper_neural.py'),'--dataset',dataset,'--system',system,
                    '--output',str(folder),'--device','cpu','--cpu-threads',str(args.cpu_threads)]
                if system=='ce':command+=['--candidate-run',sources[dataset]]
                (folder/'config.yaml').write_text(yaml.safe_dump(dict(dataset=dataset,retriever='paper-tas-b' if system=='tasb' else 'paper-anserini-bm25',reranker=None if system=='tasb' else 'paper-minilm-ce',candidate_k=100)),encoding='utf-8')
                atomic_json(folder/'status.json',dict(status='NOT_RUN'))
                entry=dict(dataset=dataset,system=system,run=str(folder),command=command,status='NOT_RUN');jobs.append(entry)
                atomic_json(session/'session.json',jobs)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':str(args.cpu_threads),'MKL_NUM_THREADS':str(args.cpu_threads),'TOKENIZERS_PARALLELISM':'false','HF_HOME':str(ROOT/'cache/huggingface'),'PYTHONUNBUFFERED':'1'}
                with (session/(folder.name+'.log')).open('w') as log:
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    start=time.monotonic()
                    try:
                        while process.poll() is None:
                            if time.monotonic()-start>args.per_run_seconds:
                                stop_owned(process)
                                atomic_json(folder/'status.json',dict(status='NOT_RUN_RESOURCE_LIMIT',reason='CPU wall-clock budget reached',retry_command=command))
                                break
                            time.sleep(5)
                    except BaseException:stop_owned(process);raise
                entry['status']=json.loads((folder/'status.json').read_text())['status']
                atomic_json(session/'session.json',jobs)
                print(dataset,system,entry['status'],folder,flush=True)
                if entry['status']!='VERIFIED':raise RuntimeError('CPU baseline did not complete')
    finally:
        unchanged=all(file_digest(ROOT/p)==v for p,v in prior.items())
        atomic_json(session/'historical_integrity.json',dict(all_unchanged=unchanged,files_checked=len(prior)))
        assert unchanged


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-session', type=Path, required=True,
                        help='Session directory containing VERIFIED Anserini BM25 entries')
    parser.add_argument('--datasets', nargs='+', choices=['scifact','nfcorpus','fiqa','arguana','trec-covid'],
                        default=['arguana','trec-covid'])
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--per-run-seconds', type=int, default=28800)
    args = parser.parse_args()
    if args.cpu_threads < 1 or args.per_run_seconds < 1:
        parser.error('CPU threads and time budget must be positive')
    main(args)
