"""Unified auditable BEIR experiments (no training)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import yaml
from lab.cache import atomic_json
from lab.data import ROOT
from lab.runner import DEFAULTS, run, system_id, verify_run
from lab.process import bounded_run


def matrix(path=None):
    configs = yaml.safe_load(Path(path or ROOT/'configs/experiments.yaml').read_text(encoding='utf-8'))['experiments']
    if not isinstance(configs, list) or not configs or any(not isinstance(c, dict) for c in configs):
        raise ValueError('Matrix must contain a nonempty list of experiment configs')
    for config in configs:
        if set(config) - set(DEFAULTS):
            raise ValueError('Unknown experiment field')
    return configs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ['run', 'benchmark']:
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path)
        for key in ['dataset', 'split', 'retriever', 'dense_model', 'reranker', 'device', 'backend', 'fusion', 'data_dir', 'model_revision', 'reranker_revision', 'candidate_run']:
            p.add_argument('--' + key.replace('_', '-'))
        for key in ['top_k', 'candidate_k', 'rrf_k', 'batch_size', 'reranker_batch_size', 'max_length', 'reranker_max_length', 'search_batch_size', 'warmup', 'repeats', 'seed', 'cpu_threads', 'cooling_pause_c', 'cooling_resume_c']:
            p.add_argument('--' + key.replace('_', '-'), type=int)
        for key in ['k1', 'b', 'epsilon']:
            p.add_argument('--' + key, type=float)
        p.add_argument('--batch-rest-seconds', type=float)
        p.add_argument('--download', action='store_true', default=None)
        p.add_argument('--ignore-identical-ids', action=argparse.BooleanOptionalAction, default=None)
        if name == 'benchmark':
            p.add_argument('--all', action='store_true')
            p.add_argument('--matrix', type=Path)
            p.add_argument('--time-budget-seconds', type=int, default=7200)
    p = sub.add_parser('evaluate')
    p.add_argument('--run', required=True, type=Path, help='V2 run directory or its metrics.json')
    args = vars(ap.parse_args())
    command = args.pop('command')
    if command == 'evaluate':
        folder = args['run'] if args['run'].is_dir() else args['run'].parent
        print(json.dumps(verify_run(folder), indent=2))
        return
    path = args.pop('config')
    config = yaml.safe_load(path.read_text(encoding='utf-8')) if path else {}
    # Accept both flat CLI config and documented nested YAML.
    for section, mapping in [('bm25', {'k1':'k1','b':'b','epsilon':'epsilon'}),
                             ('dense', {'model':'dense_model','batch_size':'batch_size'}),
                             ('fusion', {'type':'fusion','k':'rrf_k'}),
                             ('reranker', {'model':'reranker','candidate_k':'candidate_k'})]:
        if isinstance(config.get(section), dict):
            values = config.pop(section)
            config.update({mapping[k]: v for k, v in values.items() if k in mapping})
    if isinstance(config.get('retriever'), dict):
        config['retriever'] = config['retriever']['type']
    budget = args.pop('time_budget_seconds', None)
    matrix_path = args.pop('matrix', None)
    all_systems = args.pop('all', False)
    config.update({k: v for k, v in args.items() if v is not None})
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown config fields: {unknown}')
    if command == 'run':
        folder = run(config)
        status = json.loads((folder / 'status.json').read_text())['status']
        if status != 'VERIFIED':
            sys.exit(1)
        return
    if not all_systems:
        ap.error('benchmark requires --all')
    start = time.monotonic()
    summary = []
    for experiment in matrix(matrix_path):
        resolved = {**DEFAULTS, **config, **experiment}
        remaining = budget - (time.monotonic() - start)
        command_line = [sys.executable, str(ROOT / 'beir_lab.py'), 'run']
        for k, v in resolved.items():
            if v is not None and not isinstance(v, bool):
                command_line.extend(['--' + k.replace('_', '-'), str(v)])
            elif v is True:
                command_line.append('--' + k.replace('_', '-'))
            elif k == 'ignore_identical_ids':
                command_line.append('--no-ignore-identical-ids')
        entry = {'system': system_id(resolved), 'config': resolved, 'command': subprocess.list2cmdline(command_line)}
        if remaining <= 0:
            entry.update(status='NOT_RUN_RESOURCE_LIMIT', reason='Benchmark wall-clock budget exhausted')
        else:
            before = set((ROOT / 'runs').glob('*')) if (ROOT / 'runs').exists() else set()
            try:
                bounded_run(command_line, cwd=ROOT, timeout=remaining)
                created = set((ROOT / 'runs').glob('*')) - before
                if len(created) != 1:
                    raise RuntimeError('Could not identify experiment run')
                folder = created.pop()
                entry.update(json.loads((folder / 'status.json').read_text()), run=str(folder))
            except subprocess.TimeoutExpired:
                created = set((ROOT / 'runs').glob('*')) - before
                for folder in created:
                    atomic_json(folder / 'status.json', {'status': 'NOT_RUN_RESOURCE_LIMIT', 'reason': 'Wall-clock budget reached', 'system': system_id(resolved)})
                entry.update(status='NOT_RUN_RESOURCE_LIMIT', reason='Wall-clock budget reached')
        summary.append(entry)
        atomic_json(ROOT / 'runs' / 'experiment_matrix.json', summary)


if __name__ == '__main__':
    main()
