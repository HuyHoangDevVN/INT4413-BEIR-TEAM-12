"""Sequential, thermally supervised quality-only paper baseline session."""
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
import yaml
from lab.cache import atomic_json, file_digest
from lab.data import ROOT
from lab.cooling import gpu_temperature
from lab.process import stop_owned

p = argparse.ArgumentParser()
p.add_argument('--qwen', action='store_true', help='Run frozen Qwen candidates at k=50/100 instead of paper baselines')
args = p.parse_args()
prefix = 'qwen_candidate_session_' if args.qwen else 'paper_neural_session_'
session = ROOT / 'reports' / (prefix + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
session.mkdir(parents=True)
config = dict(batch_size=1, batch_rest_seconds=0, thermal_control=True, start_temperature_c=55, cooling_pause_c=73,
              cooling_resume_c=60, temperature_stop_c=80, cool_start_seconds=0,
              cool_between_seconds=0, total_budget_seconds=14400, per_run_seconds=5400,
              quality_only=True, warmup=0, repeats=0)
atomic_json(session / 'session_config.json', config)
old = {str(p.relative_to(ROOT)): file_digest(p) for folder in [ROOT / 'runs', ROOT / 'results']
       if folder.exists() for p in folder.rglob('*') if p.is_file()}
atomic_json(session / 'prior_artifacts_sha256.json', old)
deadline = time.monotonic() + config['total_budget_seconds']
telemetry, entries = [], []


def sample(label):
    value = dict(label=label, time_utc=datetime.now(timezone.utc).isoformat(), temperature_c=gpu_temperature())
    telemetry.append(value)
    atomic_json(session / 'telemetry.json', telemetry)
    return value['temperature_c']


def cooldown(label, minimum):
    started, stable = time.monotonic(), 0
    while time.monotonic() < deadline and time.monotonic() - started < 600:
        temperature = sample(label)
        stable = stable + 1 if temperature <= 55 else 0
        if stable >= (1 if minimum == 0 else 3) and time.monotonic() - started >= minimum:
            print(label, 'ready at', temperature, 'C', flush=True)
            return
        time.sleep(5)
    raise MemoryError('GPU did not cool to <=55 C within 600 seconds; model not started')


sources = (dict(scifact='20261011_123318_86ae3723', nfcorpus='20261011_141533_2f976292') if args.qwen
           else dict(scifact='20261012_011743_164d4865', nfcorpus='20261012_011854_5304b15c'))
jobs = ([(dataset, 'qwen', k) for k in [50, 100] for dataset in ['scifact', 'nfcorpus']] if args.qwen
        else [(dataset, system, 100) for system in ['tasb', 'ce'] for dataset in ['scifact', 'nfcorpus']])
atomic_json(session / 'planned_experiments.json', [dict(dataset=d,system=s,candidate_k=k) for d,s,k in jobs])
for i, (dataset, system, candidate_k) in enumerate(jobs):
    label = dataset + '-' + system + (f'-k{candidate_k}' if args.qwen else '')
    folder = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    folder.mkdir()
    (folder / 'config.yaml').write_text(yaml.safe_dump(dict(dataset=dataset,
                retriever='qwen3' if args.qwen else ('paper-tas-b' if system == 'tasb' else 'paper-anserini-bm25'),
                reranker='qwen3' if args.qwen else (None if system == 'tasb' else 'paper-minilm-ce'),
                candidate_k=candidate_k)), encoding='utf-8')
    command = ([sys.executable, str(ROOT / 'run_qwen_candidates.py'), '--dataset', dataset,
               '--candidate-k', str(candidate_k), '--output', str(folder)] if args.qwen else
               [sys.executable, str(ROOT / 'run_paper_neural.py'), '--dataset', dataset,
               '--system', system, '--output', str(folder)])
    if system == 'ce' or args.qwen:
        command += ['--candidate-run', str(ROOT / 'runs' / sources[dataset])]
    entry = dict(dataset=dataset, system=system, candidate_k=candidate_k, run=str(folder), command=command, status='NOT_RUN', reason=None)
    entries.append(entry)
    atomic_json(session / 'session.json', entries)
    atomic_json(folder / 'status.json', dict(status='NOT_RUN', system=system))
    try:
        cooldown(label + '-cooldown', 0)
    except MemoryError as error:
        entry.update(status='NOT_RUN_RESOURCE_LIMIT', reason=str(error))
        atomic_json(folder / 'status.json', dict(status=entry['status'], system=system, reason=str(error), retry_command=command))
        atomic_json(session / 'session.json', entries)
        print(entry, flush=True)
        break
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'HF_HOME': str(ROOT / 'cache/huggingface'), 'PYTHONUNBUFFERED': '1'}
    with (session / (label + '.log')).open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        started = time.monotonic()
        reason = None
        try:
            while process.poll() is None:
                if sample(label) >= 80:
                    reason = 'GPU reached hard stop 80 C'
                elif time.monotonic() >= deadline or time.monotonic() - started >= 5400:
                    reason = 'Wall-clock budget reached'
                if reason:
                    stop_owned(process)
                    break
                time.sleep(5)
        except BaseException:
            stop_owned(process)
            raise
    if reason:
        atomic_json(folder / 'status.json', dict(status='NOT_RUN_RESOURCE_LIMIT', system=system,
                    reason=reason, retry_command=command))
    status = json.loads((folder / 'status.json').read_text())
    entry.update(status=status['status'], reason=status.get('reason'))
    atomic_json(session / 'session.json', entries)
    print(label, entry['status'], folder, flush=True)
    if reason:
        break
assert all(file_digest(ROOT / path) == checksum for path, checksum in old.items()), 'Prior artifact changed'
atomic_json(session / 'historical_integrity.json', dict(files_checked=len(old), all_unchanged=True))
print(session, flush=True)
