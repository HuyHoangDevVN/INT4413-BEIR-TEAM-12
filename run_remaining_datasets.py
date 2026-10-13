"""Sequential remaining-dataset matrix, new artifacts and owned-process thermal limits."""
import fcntl
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime,timezone
import requests
import yaml
from lab.cache import atomic_json,file_digest,finalized_artifact_hashes
from lab.cooling import gpu_temperature
from lab.data import ROOT,load_dataset
from lab.process import stop_owned
from run_paper_baseline import IMAGE


def main(args):
    lock=(ROOT/'reports/remaining_datasets.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    session=ROOT/'reports'/('remaining_datasets_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    session.mkdir()
    prior=finalized_artifact_hashes(ROOT)
    atomic_json(session/'prior_artifacts_sha256.json',prior)
    config=dict(datasets=['fiqa','arguana','trec-covid'],total_budget_seconds=86400,per_run_seconds=28800,
        batch_size=1,batch_rest_seconds=0,cooling_pause_c=73,cooling_resume_c=60,temperature_stop_c=80,
        start_temperature_c=55,quality_only=True)
    atomic_json(session/'session_config.json',config)
    deadline=time.monotonic()+config['total_budget_seconds']; entries=[]; sources={}; telemetry=[]
    if args.resume:
        resumed=args.resume.resolve()
        assert resumed.is_relative_to(ROOT/'reports') and resumed.name.startswith('remaining_datasets_')
        entries=json.loads((resumed/'session.json').read_text())
        atomic_json(session/'resumed_from.json',dict(session=str(resumed)))
    else:
        for dataset in config['datasets']:
            for system,k in [('bm25',1000),('tasb',1000),('ce',100),('qwen',1000),('rerank',20),('rerank',50),('rerank',100)]:
                entries.append(dict(dataset=dataset,system=system,candidate_k=k,status='NOT_RUN',run=None,command=None,reason=None))
    atomic_json(session/'session.json',entries)
    def sample(label):
        temperature=gpu_temperature()
        telemetry.append(dict(label=label,time_utc=datetime.now(timezone.utc).isoformat(),temperature_c=temperature))
        atomic_json(session/'telemetry.json',telemetry)
        return temperature
    def execute(entry,command,gpu):
        entry['command']=command; atomic_json(session/'session.json',entries)
        if gpu:
            while sample(entry['dataset']+'-start-wait')>55:
                if time.monotonic()>=deadline:
                    raise MemoryError('GPU unavailable: did not reach <=55 C within session budget')
                time.sleep(5)
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
             'HF_HOME':str(ROOT/'cache/huggingface'),'PYTHONUNBUFFERED':'1'}
        with (session/(Path(entry['run']).name+'.log')).open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            start=time.monotonic(); reason=None
            try:
                while process.poll() is None:
                    if gpu and sample(entry['dataset']+'-'+entry['system'])>=80:
                        reason='GPU reached hard stop 80 C'
                    if time.monotonic()>=deadline or time.monotonic()-start>config['per_run_seconds']:
                        reason='Wall-clock budget reached'
                    if reason:
                        stop_owned(process); break
                    time.sleep(5)
            except BaseException:
                stop_owned(process); raise
        if reason:
            atomic_json(Path(entry['run'])/'status.json',dict(status='NOT_RUN_RESOURCE_LIMIT',reason=reason,retry_command=command))
        return json.loads((Path(entry['run'])/'status.json').read_text())
    try:
        for dataset in config['datasets']:
            print('Downloading/validating',dataset,flush=True)
            corpus,queries,qrels,meta=load_dataset(dataset,download=True)
            atomic_json(session/(dataset+'_data.json'),dict(**meta,corpus_count=len(corpus),query_count=len(queries),self_overlap=len(set(corpus)&set(queries))))
            del corpus,queries,qrels
        subprocess.run(['docker','pull',IMAGE],check=True)
        name='beir-paper-20211021'
        existing=subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        if existing:
            identity=subprocess.check_output(['docker','inspect','--format','{{.Config.Image}} {{.State.Running}}',name],text=True).strip()
            assert identity==IMAGE+' true', 'Existing container is not the pinned running benchmark; refusing mutation'
        else:
            subprocess.run(['docker','run','-d','--name',name,'--cpus','2','--memory','8g',
                '-e','JAVA_TOOL_OPTIONS=-Xmx1536m','-p','127.0.0.1:18021:8000',IMAGE],check=True)
        for _ in range(60):
            try:
                if requests.get('http://127.0.0.1:18021/docs',timeout=3).ok:break
            except requests.RequestException:pass
            time.sleep(2)
        else:raise RuntimeError('Paper API did not become ready')
        subprocess.run([sys.executable,str(ROOT/'verify_paper_runtime.py'),'--output',str(session/'runtime_verification.json')],cwd=ROOT,check=True)
        # CPU BM25 completes first while shared GPU may still be hot.
        ordered=[e for e in entries if e['system']=='bm25']+[e for e in entries if e['system']!='bm25']
        for entry in ordered:
            if entry['status']!='VERIFIED' and entry['system'] in ['tasb','ce']:
                expected='paper-tas-b' if entry['system']=='tasb' else 'paper-anserini-bm25'
                for candidate in sorted((ROOT/'runs').iterdir(),reverse=True):
                    if not (candidate/'status.json').exists() or json.loads((candidate/'status.json').read_text())['status']!='VERIFIED':continue
                    c=yaml.safe_load((candidate/'config.yaml').read_text())
                    if c['dataset']!=entry['dataset'] or c['retriever']!=expected:continue
                    if c.get('reranker')!=(None if entry['system']=='tasb' else 'paper-minilm-ce'):continue
                    m=json.loads((candidate/'metadata.json').read_text())
                    revision=m.get('model_revision') if entry['system']=='tasb' else m.get('reranker_revision')
                    if revision!=('b12d9352e776979147078a8975a4885042984fd1' if entry['system']=='tasb' else '233902d25c440f23af6f7d6e94d2946bac0bee0a'):continue
                    if m.get('upstream_commit')!='52de18ae6f1ccc7426eb70895c0e43e1552ad493' or m.get('precision')!='torch.float32':continue
                    if entry['system']=='ce':
                        source_meta=json.loads((Path(sources[(entry['dataset'],'bm25')])/'metadata.json').read_text())
                        if m.get('candidate_source_checksum')!=source_meta['upstream_results_checksum']:continue
                    entry.update(run=str(candidate),status='VERIFIED',reason=None)
                    atomic_json(session/'session.json',entries)
                    break
            if entry['status']=='VERIFIED':
                existing=Path(entry['run']); meta=json.loads((existing/'metadata.json').read_text())
                data=json.loads((session/(entry['dataset']+'_data.json')).read_text())
                assert json.loads((existing/'status.json').read_text())['status']=='VERIFIED'
                assert all(meta[k]==data[k] for k in ['corpus_checksum','query_checksum','qrels_checksum'])
                assert file_digest(existing/'ranking.tsv')==meta['ranking_checksum']
                if entry['system'] in ['bm25','qwen']:sources[(entry['dataset'],entry['system'])]=str(existing)
                print('Reusing VERIFIED',entry['dataset'],entry['system'],entry['run'],flush=True)
                continue
            if entry['run']:
                entry.setdefault('previous_attempts',[]).append({k:entry[k] for k in ['run','status','reason','command']})
            entry.update(status='NOT_RUN',reason=None)
            folder=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]); folder.mkdir()
            entry['run']=str(folder)
            system=entry['system']; dataset=entry['dataset']
            retriever='qwen3' if system in ['qwen','rerank'] else ('paper-tas-b' if system=='tasb' else 'paper-anserini-bm25')
            (folder/'config.yaml').write_text(yaml.safe_dump(dict(dataset=dataset,retriever=retriever,
                reranker='qwen3' if system=='rerank' else ('paper-minilm-ce' if system=='ce' else None),candidate_k=entry['candidate_k'])),encoding='utf-8')
            atomic_json(folder/'status.json',dict(status='NOT_RUN',reason=None))
            script={'bm25':'run_paper_baseline.py','tasb':'run_paper_neural.py','ce':'run_paper_neural.py',
                    'qwen':'run_qwen_embedding_quality.py','rerank':'run_qwen_candidates.py'}[system]
            command=[sys.executable,str(ROOT/script),'--dataset',dataset,'--output',str(folder)]
            if system in ['tasb','ce']:command+=['--system',system]
            if system in ['ce','rerank']:
                source=sources[(dataset,'bm25' if system=='ce' else 'qwen')]
                command+=['--candidate-run',source]
            if system=='rerank':command+=['--candidate-k',str(entry['candidate_k'])]
            status=execute(entry,command,system!='bm25')
            entry.update(status=status['status'],reason=status.get('reason'))
            atomic_json(session/'session.json',entries)
            print(dataset,system,entry['candidate_k'],entry['status'],folder,flush=True)
            if entry['status']!='VERIFIED':raise RuntimeError('Experiment failed; remaining jobs retained as NOT_RUN')
            if system in ['bm25','qwen']:sources[(dataset,system)]=str(folder)
        atomic_json(session/'status.json',dict(status='VERIFIED',completed=21))
    except BaseException as error:
        if 'entry' in locals() and entry['run'] and entry['status']=='NOT_RUN':
            entry.update(status='NOT_RUN_RESOURCE_LIMIT' if isinstance(error,MemoryError) else 'EXECUTION_ERROR',reason=str(error))
            atomic_json(Path(entry['run'])/'status.json',dict(status=entry['status'],reason=str(error),retry_command=entry['command']))
        atomic_json(session/'session.json',entries)
        atomic_json(session/'status.json',dict(status='INCOMPLETE',reason=str(error),completed=sum(e['status']=='VERIFIED' for e in entries)))
        raise
    finally:
        unchanged=all(file_digest(ROOT/p)==v for p,v in prior.items())
        atomic_json(session/'historical_integrity.json',dict(files_checked=len(prior),all_unchanged=unchanged,scope='finalized_runs_and_legacy_results'))
        print(session,flush=True)
        assert unchanged


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',type=Path)
    main(parser.parse_args())
