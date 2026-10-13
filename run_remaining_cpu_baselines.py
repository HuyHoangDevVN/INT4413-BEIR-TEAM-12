"""Explicit FP32 CPU baseline work while the shared GPU is unavailable."""
import fcntl
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


def main():
    lock=(ROOT/'reports/remaining_cpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=ROOT/'reports/remaining_datasets_20261012_134031'
    entries=json.loads((source/'session.json').read_text())
    prior=finalized_artifact_hashes(ROOT)
    session=ROOT/'reports'/('remaining_cpu_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'));session.mkdir()
    jobs=[]
    try:
        for dataset in ['arguana','trec-covid']:
            for system in ['tasb','ce']:
                folder=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);folder.mkdir()
                command=[sys.executable,str(ROOT/'run_paper_neural.py'),'--dataset',dataset,'--system',system,
                    '--output',str(folder),'--device','cpu','--cpu-threads','8']
                if system=='ce':command+=['--candidate-run',next(e['run'] for e in entries if e['dataset']==dataset and e['system']=='bm25')]
                (folder/'config.yaml').write_text(yaml.safe_dump(dict(dataset=dataset,retriever='paper-tas-b' if system=='tasb' else 'paper-anserini-bm25',reranker=None if system=='tasb' else 'paper-minilm-ce',candidate_k=100)),encoding='utf-8')
                atomic_json(folder/'status.json',dict(status='NOT_RUN'))
                entry=dict(dataset=dataset,system=system,run=str(folder),command=command,status='NOT_RUN');jobs.append(entry)
                atomic_json(session/'session.json',jobs)
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false','HF_HOME':str(ROOT/'cache/huggingface'),'PYTHONUNBUFFERED':'1'}
                with (session/(folder.name+'.log')).open('w') as log:
                    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    start=time.monotonic()
                    try:
                        while process.poll() is None:
                            if time.monotonic()-start>28800:
                                stop_owned(process)
                                atomic_json(folder/'status.json',dict(status='NOT_RUN_RESOURCE_LIMIT',reason='CPU 8-hour limit',retry_command=command))
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


if __name__=='__main__':main()
