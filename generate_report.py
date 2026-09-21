"""Generate tables and six figures from successful V2 runs only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lab.data import ROOT
from lab.runner import system_id


def collect(runs):
    rows = []
    for folder in sorted(Path(runs).glob('*')):
        if not folder.is_dir() or not (folder/'status.json').exists():
            continue
        status = json.loads((folder/'status.json').read_text())
        config = yaml.safe_load((folder/'config.yaml').read_text())
        timing = json.loads((folder/'timing.json').read_text()) if (folder/'timing.json').exists() else {}
        metrics = json.loads((folder/'metrics.json').read_text()) if status['status'] in ['VERIFIED','RUN_SUCCESS'] else {}
        row = {'run_id':folder.name, 'dataset':config['dataset'], 'system':system_id(config), 'status':status['status'],
               'reason':status.get('reason'), 'nDCG@10':metrics.get('nDCG@10'), 'Recall@100':metrics.get('Recall@100'),
               'Recall@10':metrics.get('Recall@10'), 'MRR@10':metrics.get('MRR@10'), 'MAP@100':metrics.get('MAP@100'),
               'legacy_exponential_ndcg@10':metrics.get('legacy_exponential_ndcg@10'),
               'Latency P50':timing.get('warm',{}).get('query_latency_p50'), 'Latency P95':timing.get('warm',{}).get('query_latency_p95'),
               'RAM':timing.get('RAM_peak_bytes'), 'VRAM':timing.get('VRAM_peak_bytes'),
               'latency_scope':timing.get('protocol',{}).get('latency_scope','end_to_end')}
        row['latency_includes_cooling'] = timing.get('protocol',{}).get('latency_includes_batch_rest_and_thermal_wait',False)
        row['_protocol'] = timing.get('protocol')
        row['_hardware'] = json.loads((folder/'hardware.json').read_text()) if (folder/'hardware.json').exists() else {}
        metadata = json.loads((folder/'metadata.json').read_text()) if (folder/'metadata.json').exists() else {}
        row['device'] = row['_hardware'].get('device')
        row['precision'] = metadata.get('precision')
        row['max_length'] = config.get('max_length')
        row['_config'] = config
        rows.append(row)
    for row in rows:
        baselines = [r for r in rows if r['dataset']==row['dataset'] and r['system']=='bm25' and r['nDCG@10'] is not None
                     and r['_protocol']==row['_protocol'] and r['_hardware'].get('cpu')==row['_hardware'].get('cpu')
                     and all(r['_config'].get(k)==row['_config'].get(k) for k in ['split','top_k','k1','b','epsilon'])]
        base = baselines[-1] if baselines else None
        row['delta_nDCG@10'] = row['nDCG@10']-base['nDCG@10'] if base and row['nDCG@10'] is not None else None
        row['delta_Recall@100'] = row['Recall@100']-base['Recall@100'] if base and row['Recall@100'] is not None else None
        row['latency_ratio'] = row['Latency P50']/base['Latency P50'] if base and row['Latency P50'] is not None else None
        # Paper systems use exact identities; a reranked pipeline never becomes BM25.
        paper_path = ROOT / 'references' / 'beir_paper_results.csv'
        references = list(csv.DictReader(paper_path.open(encoding='utf-8'))) if paper_path.exists() else []
        reference = next((r for r in references if r['dataset'] == row['dataset'] and r['model'] == 'BM25'
                          and row['system'] == 'paper-anserini-bm25' and r['metric'] == 'nDCG@10'), None)
        neural_reference_path = ROOT / 'references' / 'paper_baselines_2021_verified.csv'
        if reference is None and neural_reference_path.exists():
            neural_references = list(csv.DictReader(neural_reference_path.open(encoding='utf-8')))
            reference = next((r for r in neural_references if r['dataset'] == row['dataset']
                              and r['system_id'] == row['system'] and r['metric'] == 'nDCG@10'), None)
        row['Paper result'] = float(reference['paper_value']) if reference else None
        row['Our run'] = row['nDCG@10']
        row['paper_source'] = reference['source'] if reference else None
    return rows


def plot(rows, out):
    out.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r['nDCG@10'] is not None]
    for key, name in [('nDCG@10','ndcg_by_model'),('Recall@100','recall_by_model'),('delta_nDCG@10','improvement_over_bm25')]:
        fig, ax = plt.subplots(figsize=(max(8,len(valid)*.65),9))
        selected = [r for r in valid if r[key] is not None]
        ax.bar(range(len(selected)), [r[key] for r in selected])
        ax.set_xticks(range(len(selected)), [r['dataset']+' / '+r['system'] for r in selected], rotation=65, ha='right')
        ax.set_ylabel(key)
        if not selected:
            ax.text(.5,.5,'No measured data',transform=ax.transAxes,ha='center')
        fig.tight_layout()
        fig.savefig(out/(name+'.png'),dpi=180)
        plt.close(fig)
    for name in ['ndcg_vs_latency','ndcg_vs_memory']:
        fig, ax = plt.subplots(figsize=(9,6))
        for row in valid:
            if name=='ndcg_vs_latency':
                x = row['Latency P50']
                if x is not None:
                    ax.scatter(x*1000,row['nDCG@10'])
                    ax.annotate(row['dataset']+'/'+row['system'],(x*1000,row['nDCG@10']),fontsize=7)
            else:
                for key, marker in [('RAM','o'),('VRAM','x')]:
                    if row[key] is not None:
                        x=row[key]/1024**3
                        ax.scatter(x,row['nDCG@10'],marker=marker)
                        ax.annotate(row['dataset']+'/'+row['system']+'/'+key,(x,row['nDCG@10']),fontsize=7)
        ax.set_xlabel('Warm P50 milliseconds' if name=='ndcg_vs_latency' else 'Peak GiB (RAM=o, VRAM=x)')
        ax.set_ylabel('nDCG@10')
        fig.tight_layout()
        fig.savefig(out/(name+'.png'),dpi=180)
        plt.close(fig)
    datasets=sorted({r['dataset'] for r in valid})
    systems=sorted({r['system'] for r in valid})
    fig,ax=plt.subplots(figsize=(max(7,len(systems)),8))
    if datasets and systems:
        values=np.full((len(datasets),len(systems)),np.nan)
        for r in valid:
            values[datasets.index(r['dataset']),systems.index(r['system'])]=r['nDCG@10']
        heat=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=1,aspect='auto')
        ax.set_xticks(range(len(systems)),systems,rotation=65,ha='right')
        ax.set_yticks(range(len(datasets)),datasets)
        for i in range(len(datasets)):
            for j in range(len(systems)):
                ax.text(j,i,f'{values[i,j]:.3f}' if np.isfinite(values[i,j]) else 'N/A',ha='center')
        fig.colorbar(heat,ax=ax,label='nDCG@10')
    else:
        ax.text(.5,.5,'No measured data',transform=ax.transAxes,ha='center')
    fig.tight_layout()
    fig.savefig(out/'dataset_heatmap.png',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,default=ROOT/'runs')
    p.add_argument('--output',type=Path,default=ROOT/'reports')
    args=p.parse_args()
    rows=collect(args.runs)
    args.output.mkdir(parents=True,exist_ok=True)
    columns=[k for k in rows[0] if not k.startswith('_')] if rows else ['dataset','system','status']
    with (args.output/'results_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in rows)
    display=['dataset','system','status','device','max_length','Paper result','Our run','Recall@100','latency_scope','latency_includes_cooling','Latency P50','Latency P95','RAM','VRAM','delta_nDCG@10','latency_ratio']
    def format_value(v):
        return 'N/A' if v is None else f'{v:.6g}' if isinstance(v,float) else str(v)
    lines=['# Results from V2 runs','', 'Latency in seconds; memory in bytes. Missing values are N/A.',
           'Historical results/*.json are NOT_RUN / historical unverified, excluded from new measurements.', '',
           '| '+' | '.join(display)+' |','| '+' | '.join(['---']*len(display))+' |']
    lines.extend('| '+' | '.join(format_value(r.get(k)) for k in display)+' |' for r in rows)
    matrix_path=args.runs/'experiment_matrix.json'
    if matrix_path.exists():
        lines.extend(['','## Experiment matrix status',''])
        for e in json.loads(matrix_path.read_text()):
            lines.append(f'- {e["system"]}: {e["status"]}; {e.get("reason") or "completed"}')
    lines.extend(['','Paper result and Our run are separate evidence lanes; references CSV is not an experiment.'])
    (args.output/'results_summary.md').write_text('\n'.join(lines),encoding='utf-8')
    plot(rows,ROOT/'plots')
    print(args.output/'results_summary.md')


if __name__=='__main__':
    main()
