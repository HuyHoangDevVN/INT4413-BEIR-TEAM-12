"""Combine two historical datasets and 21 completed new runs without rewriting artifacts."""
import argparse
import csv
import json
from datetime import datetime,timezone
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
from report_verified_comparison import ROOT,RUNS,SYSTEMS,METRICS,snapshot,verify_readonly,write_csv
from lab.data import load_dataset
from lab.runner import system_id


def main(args):
    before=snapshot()
    session=args.session.resolve()
    assert session.is_relative_to(ROOT/'reports')
    entries=json.loads((session/'session.json').read_text())
    assert len(entries)==21 and all(e['status']=='VERIFIED' for e in entries)
    assert json.loads((session/'historical_integrity.json').read_text())['all_unchanged']
    runs={d:list(ids) for d,ids in RUNS.items()}
    order=[('bm25',1000),('tasb',1000),('ce',100),('qwen',1000),('rerank',20),('rerank',50),('rerank',100)]
    for dataset in ['fiqa','arguana','trec-covid']:
        runs[dataset]=[Path(next(e['run'] for e in entries if e['dataset']==dataset and (e['system'],e['candidate_k'])==key)).name for key in order]
    references=list(csv.DictReader((ROOT/'references/paper_baselines_2021_five_datasets.csv').open(encoding='utf-8-sig')))
    rows=[]; papers=[]; coverage=[]
    for dataset,ids in runs.items():
        corpus,queries,qrels,data=load_dataset(dataset)
        coverage.append(dict(dataset=dataset,corpus_count=len(corpus),query_count=len(queries),metadata=data))
        for label,run_id in zip(SYSTEMS,ids):
            folder=ROOT/'runs'/run_id
            measured,_,meta=verify_readonly(folder,corpus,queries,qrels,data)
            config=yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
            base=measured if label==SYSTEMS[0] else base
            row=dict(dataset=dataset,system=label,system_id=system_id(config),run_id=run_id,status='VERIFIED',**measured,
                ndcg_delta_bm25=measured['nDCG@10']-base['nDCG@10'],
                ndcg_relative_percent_bm25=(measured['nDCG@10']/base['nDCG@10']-1)*100)
            rows.append(row)
            for ref in references:
                if ref['dataset']==dataset and ref['system_id']==row['system_id']:
                    papers.append(dict(**ref,measured_value=measured['nDCG@10'],
                        rounded_match=round(measured['nDCG@10'],3)==float(ref['paper_value']),run_id=run_id))
    assert len(rows)==35 and len(papers)==15
    output=ROOT/'reports/paper_reproduction'/('five_datasets_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    write_csv(output/'measured_metrics.csv',rows); write_csv(output/'paper_reference.csv',papers)
    (output/'dataset_evidence.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# BEIR: so sánh 7 hệ thống trên 5 dataset', '',
        'Đủ 35 run VERIFIED: 14 run SciFact/NFCorpus được giữ nguyên, cộng 21 run FiQA/ArguAna/TREC-COVID. Báo cáo đọc lại ranking, checksum, coverage và đối chiếu cả năm metric với pytrec_eval; không chạy inference mới khi tạo báo cáo.', '',
        '| Hệ thống | SciFact | NFCorpus | FiQA | ArguAna | TREC-COVID |', '|---|---:|---:|---:|---:|---:|']
    for label in SYSTEMS:
        lines.append('| '+label+' | '+' | '.join(f"{next(r['nDCG@10'] for r in rows if r['dataset']==d and r['system']==label):.6f}" for d in runs)+' |')
    lines+=['', 'Bảng trên là nDCG@10 gain tuyến tính. Dataset có cùng corpus/query/qrels checksum giữa các hệ thống; binary relevance >0, giữ mọi query trong test split. CSV giữ số thực gốc và gain mũ riêng, không dùng gain mũ để xếp hạng chính.', '']
    for dataset in runs:
        lines += [f'## {dataset}', '', '| Hệ thống | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | MAP@100 | Delta nDCG so BM25 | Tương đối |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['dataset']==dataset:
                lines.append('| '+r['system']+' | '+' | '.join(f'{r[m]:.6f}' for m in METRICS)+f" | {r['ndcg_delta_bm25']:+.6f} | {r['ndcg_relative_percent_bm25']:+.2f}% |")
    lines += ['', '## Tái hiện paper và nguồn tham chiếu', '',
        '| Dataset | System ID | BEIR 2021 nDCG | Run nDCG | Khớp 3 chữ số |', '|---|---|---:|---:|---|']
    for r in papers:
        lines.append(f"| {r['dataset']} | {r['system_id']} | {float(r['paper_value']):.3f} | {r['measured_value']:.6f} | {'Có' if r['rounded_match'] else 'Không'} |")
    lines += ['', 'Nguồn số tham chiếu: [BEIR 2021 v4, Table 2](https://arxiv.org/html/2104.08663v4), được lưu riêng trong CSV với system ID đầy đủ. ArguAna BM25 Anserini multifield đo khoảng 0,414, khác 0,315 của bảng năm 2021; số 0,414 nDCG và 0,943 Recall@100 được xác nhận trong [Pyserini reproduction](https://castorini.github.io/pyserini/2cr/beir.html). Không đổi kết quả để ép khớp paper, không coi matching rounded scores là runtime byte-identical.', '',
        '## Điều kiện và hạn chế', '',
        'Ba dataset bổ sung chạy server Ubuntu có A30, batch 1, context 512, không nghỉ cố định. TAS-B/CE FP32; một số baseline dùng CPU 8 thread khi GPU nóng do workload dùng chung, device thực tế được lưu hardware/config từng run. Qwen GPU BF16; revision model giữ nguyên như hai dataset đầu. Bắt đầu GPU ≤55°C, tạm dừng 73→60°C, stop 80°C. Phiên quality-only không đo warm latency/QPS; không so tốc độ giữa CPU/GPU hoặc với run Qwen laptop có nghỉ cố định.', '',
        'ArguAna giữ nguyên 1.406 query và qrels, có 5 relevant document thiếu trong corpus chính thức: không bịa document hoặc bỏ query, lưu missing policy trong metadata; self-ID được bỏ trước candidate selection. TREC-COVID giữ hai nhãn -1 unjudged, zero gain/relevance không dương, tương thích pytrec_eval. Xem docs/REMAINING_DATASET_RUN.md và dataset_evidence.json.', '',
        'TREC-COVID index gây OOM container 3 GiB; retry mới với 8 GiB riêng cho container benchmark, Java heap 1536 MiB và CPU 2. BM25 parameters/index/search source không đổi. Attempt lỗi không có metric hoàn chỉnh. Các run cũ và JSON V1 vẫn giữ nguyên.', '',
        'Bằng chứng phủ 5 dataset, không phải toàn bộ BEIR; ba nhóm domain rộng gồm khoa học/y sinh, tài chính, lập luận. Không gọi đây là zero-shot nghiêm ngặt: checkpoint có thể đã thấy BEIR trong training. Các kết luận thống kê cũ chỉ áp dụng SciFact/NFCorpus; ba dataset mới chưa được kiểm định trong báo cáo này.', '',
        'Quyết định cũ top-50 SciFact/top-100 NFCorpus được giữ nguyên theo báo cáo final_conclusions_20261012_075023. Không tự áp dụng quyết định này cho domain mới chỉ vì cùng model.', '',
        '## Run nguồn', '', '| Dataset | Hệ thống | Run ID |', '|---|---|---|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['system']} | [{r['run_id']}](../../../runs/{r['run_id']}/metrics.json) |")
    lines+=['',f'Phiên mới: `{session.relative_to(ROOT)}`. Tái tạo: `.\\.venv-paper\\Scripts\\python.exe report_five_datasets.py --session {session.relative_to(ROOT)}`. integrity.json xác nhận artifact nguồn giữ nguyên trong lần tạo báo cáo.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,5,figsize=(18,5),sharey=True)
    labels=['BM25','TAS-B','BM25+CE100','Qwen','Qwen+20','Qwen+50','Qwen+100']
    for ax,dataset in zip(axes,runs):
        selected=[r for r in rows if r['dataset']==dataset]
        ax.barh(labels,[r['nDCG@10'] for r in selected],color=['#68757f']*3+['#2474b5']+['#13856c']*3)
        ax.set_title(dataset);ax.set_xlim(0,1);ax.set_xlabel('nDCG@10')
    axes[0].invert_yaxis();fig.tight_layout();fig.savefig(output/'ndcg_comparison.png',dpi=180);plt.close(fig)
    assert snapshot()==before
    (output/'integrity.json').write_text(json.dumps(dict(all_unchanged=True,source_files_checked=len(before),
        runs_verified=35,artifact_sha256=before),indent=2),encoding='utf-8')
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--session',type=Path,required=True)
    main(parser.parse_args())
