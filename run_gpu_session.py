"""Bounded sequential GPU session; no CPU fallback or simultaneous models."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from lab.cache import atomic_json
from lab.data import ROOT
from lab.process import stop_owned


def gpu_sample():
    exe = Path('C:/Windows/System32/nvidia-smi.exe') if os.name == 'nt' else Path('nvidia-smi')
    output = subprocess.run([str(exe), '--query-gpu=temperature.gpu,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    if output.returncode:
        raise RuntimeError('GPU telemetry unavailable')
    temperature, memory, utilization = [int(x.strip()) for x in output.stdout.splitlines()[0].split(',')]
    return dict(temperature_c=temperature, total_gpu_memory_used_mib=memory, utilization_percent=utilization)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget-seconds', type=int, default=7200)
    p.add_argument('--per-run-seconds', type=int, default=1200)
    p.add_argument('--temperature-stop', type=int, default=85)
    p.add_argument('--qwen-only', action='store_true')
    p.add_argument('--dataset', default='scifact')
    p.add_argument('--cool-start-seconds', type=int, default=0)
    p.add_argument('--cool-between-seconds', type=int, default=0)
    p.add_argument('--start-temperature', type=int, default=55)
    p.add_argument('--batch-rest-seconds', type=float, default=0)
    args = p.parse_args()
    if args.batch_rest_seconds < 0 or args.start_temperature >= args.temperature_stop:
        p.error('Invalid rest/start temperature')
    session = ROOT / 'reports' / ('gpu_session_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    session.mkdir(parents=True)
    entries, samples = [], []
    deadline = time.monotonic() + args.budget_seconds
    successful = {}
    atomic_json(session/'session_config.json', vars(args))
    def cooldown(label, minimum_seconds):
        started = time.monotonic()
        stable = 0
        while True:
            sample = dict(label=label, elapsed_seconds=time.monotonic()-started, **gpu_sample())
            samples.append(sample)
            atomic_json(session/'telemetry.json', samples)
            stable = stable + 1 if sample['temperature_c'] <= args.start_temperature else 0
            if time.monotonic() - started >= minimum_seconds and stable >= 3:
                print(f'{label}: GPU cooled to {sample["temperature_c"]} C', flush=True)
                return True
            if time.monotonic() >= deadline or time.monotonic()-started >= 600:
                return False
            time.sleep(5)
    experiments = [('minilm', dict(retriever='minilm', batch_size=16, max_length=256)),
                   ('minilm-ce', dict(retriever='bm25', reranker='minilm', candidate_k=20)),
                   ('bge', dict(retriever='bge-m3')),
                   ('hybrid', dict(retriever='hybrid', dense_model='bge-m3')),
                   ('hybrid-bge-ce', dict(retriever='hybrid', dense_model='bge-m3', reranker='bge-reranker-v2-m3', candidate_k=20)),
                   ('qwen', dict(retriever='qwen3', batch_size=2)),
                   ('qwen-ce', dict(retriever='qwen3', reranker='qwen3', candidate_k=20))]
    stopped = None
    if args.qwen_only:
        experiments = experiments[-2:]
        for _, config in experiments:
            config.update(batch_size=1, reranker_batch_size=1, cpu_threads=1,
                          batch_rest_seconds=args.batch_rest_seconds, cooling_pause_c=73, cooling_resume_c=60)
    if args.cool_start_seconds and not cooldown('initial-cooldown', args.cool_start_seconds):
        stopped = 'Initial cooldown did not reach start conditions'
    for label, config in experiments:
        if label in ['hybrid-bge-ce', 'qwen-ce']:
            source = successful.get('hybrid' if label == 'hybrid-bge-ce' else 'qwen')
            if source:
                config['candidate_run'] = str(source)
            else:
                entries.append(dict(label=label, status='NOT_RUN', reason='Verified candidate source unavailable'))
                continue
        command = [sys.executable, str(ROOT/'beir_lab.py'), 'run', '--dataset', args.dataset, '--device', 'cuda',
                   '--batch-size', '4', '--reranker-batch-size', '2', '--max-length', '512', '--reranker-max-length', '512',
                   '--cpu-threads', '2']
        for k,v in config.items():
            command += ['--'+k.replace('_','-'), str(v)]
        entry = dict(label=label, command=command)
        if entries and not stopped and args.cool_between_seconds:
            if not cooldown(label+'-cooldown', args.cool_between_seconds):
                stopped = 'Between-model cooldown did not reach start conditions'
        if stopped or time.monotonic() >= deadline:
            entry.update(status='NOT_RUN_RESOURCE_LIMIT', reason=stopped or 'Session budget reached')
            entries.append(entry)
            continue
        before = {x for x in (ROOT/'runs').iterdir() if x.is_dir()}
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
        started = time.monotonic()
        reason = None
        with (session/(label+'.log')).open('w',encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    sample = dict(label=label, elapsed_seconds=time.monotonic()-started, **gpu_sample())
                    samples.append(sample)
                    atomic_json(session/'telemetry.json', samples)
                    if sample['temperature_c'] >= args.temperature_stop:
                        reason = stopped = f'Temperature stop threshold {args.temperature_stop} C reached'
                    elif time.monotonic() >= deadline or time.monotonic()-started >= args.per_run_seconds:
                        reason = 'Wall-clock budget reached'
                    if reason:
                        stop_owned(process)
                        break
                    time.sleep(5)
            except (Exception, KeyboardInterrupt) as exc:
                reason = stopped = f'Session stopped: {type(exc).__name__}'
                stop_owned(process)
        created = {x for x in (ROOT/'runs').iterdir() if x.is_dir()} - before
        if len(created) == 1:
            folder = created.pop()
            if reason:
                atomic_json(folder/'status.json', dict(status='NOT_RUN_RESOURCE_LIMIT', reason=reason,
                                                      retry_command=subprocess.list2cmdline(command)))
                if not (folder/'timing.json').exists():
                    atomic_json(folder/'timing.json', dict(incomplete=True, elapsed_seconds=time.monotonic()-started,
                                                         RAM_peak_bytes=None, VRAM_peak_bytes=None, cold={}, warm={}))
            entry.update(json.loads((folder/'status.json').read_text()), run=str(folder))
            if entry['status'] == 'VERIFIED':
                successful[label] = folder
        else:
            entry.update(status='RUN_FAILED',reason=reason or 'No unique run artifact')
        entries.append(entry)
        atomic_json(session/'session.json', entries)
        print(label, entry['status'], flush=True)
        time.sleep(5)
    atomic_json(session/'session.json', entries)
    print(session, flush=True)


if __name__ == '__main__':
    main()
