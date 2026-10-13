"""Create a new report from immutable measured artifacts; never run inference."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytrec_eval
import yaml
from evaluation.metrics import evaluate
from lab.cache import file_digest
from lab.data import ROOT, load_dataset
from lab.runner import read_ranking, system_id

SYSTEMS = ['Anserini BM25', 'TAS-B', 'Anserini BM25 → MiniLM CE top-100',
           'Qwen embedding', 'Qwen embedding → Qwen reranker top-20',
           'Qwen embedding → Qwen reranker top-50', 'Qwen embedding → Qwen reranker top-100']
RUNS = {
    'scifact': ['20261012_011743_164d4865', '20261012_031344_b3d097b2', '20261012_031511_57b74dda',
                '20261011_123318_86ae3723', '20261011_125607_76912e7b', '20261012_034346_41582e11', '20261012_042342_4498124d'],
    'nfcorpus': ['20261012_011854_5304b15c', '20261012_031430_7c681d4e', '20261012_032010_b259b035',
                 '20261011_141533_2f976292', '20261011_143638_8d39eaff', '20261012_040125_fd25658a', '20261012_050442_11865b28'],
}
METRICS = ['nDCG@10', 'Recall@10', 'Recall@100', 'MRR@10', 'MAP@100']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def snapshot():
    return {str(p.relative_to(ROOT)): file_digest(p) for base in ['runs', 'results']
            for p in (ROOT/base).rglob('*') if p.is_file()}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def verify_readonly(folder, corpus, queries, qrels, data):
    assert read_json(folder/'status.json')['status'] == 'VERIFIED', folder
    meta = read_json(folder/'metadata.json')
    assert all(meta[k] == data[k] for k in ['corpus_checksum', 'query_checksum', 'qrels_checksum'])
    for name, key in [('ranking.tsv', 'ranking_checksum'), ('query_coverage.json', 'query_coverage_checksum'),
                      ('upstream_results.json', 'upstream_results_checksum')]:
        if key in meta:
            assert file_digest(folder/name) == meta[key], (folder, name)
    ranking = read_ranking(folder/'ranking.tsv')
    assert set(ranking) == set(queries)
    assert all(r['document_id'] in corpus for rows in ranking.values() for r in rows)
    measured = evaluate(ranking, qrels)
    assert measured == read_json(folder/'metrics.json')
    trec = {q: {r['document_id']: float(len(rows)-i) for i, r in enumerate(rows)} for q, rows in ranking.items()}
    values = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut.10', 'recall.10,100', 'map_cut.100'}).evaluate(trec)
    for metric, key in [('nDCG@10', 'ndcg_cut_10'), ('Recall@10', 'recall_10'),
                        ('Recall@100', 'recall_100'), ('MAP@100', 'map_cut_100')]:
        assert abs(sum(values[q][key] for q in qrels)/len(qrels)-measured[metric]) < 1e-9
    top10 = {q: dict(list(rows.items())[:10]) for q, rows in trec.items()}
    values = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank'}).evaluate(top10)
    assert abs(sum(values[q]['recip_rank'] for q in qrels)/len(qrels)-measured['MRR@10']) < 1e-9
    return measured, ranking, meta


def main():
    before = snapshot()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    output = ROOT/'reports/paper_reproduction'/('complete_comparison_'+stamp)
    rows, deltas, evidence, performance, papers = [], [], [], [], []
    references = list(csv.DictReader((ROOT/'references/paper_baselines_2021_verified.csv').open(encoding='utf-8-sig')))
    for r in csv.DictReader((ROOT/'references/beir_paper_results.csv').open(encoding='utf-8-sig')):
        if r['dataset'] in RUNS and r['model'] == 'BM25':
            references.append(dict(dataset=r['dataset'], system_id='paper-anserini-bm25', metric=r['metric'],
                                   paper_value=r['paper_value'], source=r['source'], table=r['table']))
    for dataset, ids in RUNS.items():
        corpus, queries, qrels, data = load_dataset(dataset)
        rankings = {}
        for index, (label, run_id) in enumerate(zip(SYSTEMS, ids)):
            folder = ROOT/'runs'/run_id
            metrics, ranking, meta = verify_readonly(folder, corpus, queries, qrels, data)
            rankings[index] = ranking
            config = yaml.safe_load((folder/'config.yaml').read_text(encoding='utf-8'))
            row = dict(dataset=dataset, system=label, system_id=system_id(config), run_id=run_id, status='VERIFIED', **metrics)
            rows.append(row)
            evidence.append(dict(dataset=dataset, system=label, run_id=run_id, query_count=len(queries),
                corpus_count=len(corpus), metric_recomputed=True, pytrec_eval='matched_all_five_metrics',
                metadata=meta, config=config))
            timing = read_json(folder/'timing.json')
            hardware = read_json(folder/'hardware.json') if (folder/'hardware.json').exists() else {}
            performance.append(dict(dataset=dataset, system=label, run_id=run_id,
                device=hardware.get('gpu') or hardware.get('device'),
                cold_total_seconds=timing.get('cold', {}).get('total_seconds'),
                latency_scope=timing.get('protocol', {}).get('latency_scope', 'end_to_end'),
                warm_p50_seconds=timing.get('warm', {}).get('query_latency_p50'),
                warm_p95_seconds=timing.get('warm', {}).get('query_latency_p95'),
                warm_qps=timing.get('warm', {}).get('queries_per_second'),
                ram_peak_bytes=timing.get('RAM_peak_bytes'), vram_peak_bytes=timing.get('VRAM_peak_bytes')))
            if index in [2, 4, 5, 6]:
                source_index, depth = (0, 100) if index == 2 else (3, [20, 50, 100][index-4])
                prefix = {q: r[:depth] for q, r in rankings[source_index].items()}
                assert all({r['document_id'] for r in ranking[q]} == {r['document_id'] for r in prefix[q]} for q in queries)
                assert abs(evaluate(prefix, qrels)['Recall@100']-metrics['Recall@100']) < 1e-12
                evidence[-1]['candidate_set_preserved'] = True
            for reference in references:
                if reference['dataset'] == dataset and reference['system_id'] == row['system_id']:
                    value = float(reference['paper_value'])
                    assert round(metrics[reference['metric']], 3) == value
                    papers.append({**reference, 'run_id': run_id, 'measured_value': metrics[reference['metric']], 'rounded_match': True})
        selected = rows[-7:]
        for index, row in enumerate(selected):
            comparisons = [('Anserini BM25', selected[0])]
            if index >= 3:
                comparisons += [('TAS-B', selected[1]), ('Anserini BM25 → MiniLM CE top-100', selected[2])]
            if index >= 4:
                comparisons += [('Qwen embedding', selected[3]), ('Qwen reranker top-20', selected[4])]
            if index == 6:
                comparisons += [('Qwen reranker top-50', selected[5])]
            for reference_name, reference in comparisons:
                for metric in METRICS:
                    delta = row[metric]-reference[metric]
                    deltas.append(dict(dataset=dataset, system=row['system'], baseline=reference_name, metric=metric,
                                       absolute_delta=delta, relative_percent=delta/reference[metric]*100))
    assert len(rows) == 14 and len(papers) == 6
    output.mkdir()
    write_csv(output/'measured_metrics.csv', rows)
    write_csv(output/'improvements.csv', deltas)
    write_csv(output/'paper_reference.csv', papers)
    write_csv(output/'performance.csv', performance)
    (output/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# So sánh ba baseline BEIR 2021 với Qwen trên SciFact/NFCorpus', '',
             'Báo cáo dùng 14 run thực đã VERIFIED: Anserini BM25, TAS-B, Anserini BM25 → MiniLM CE top-100, Qwen embedding và Qwen reranker top-20/50/100. Không thực hiện inference mới. Tất cả ranking được đọc lại, kiểm tra checksum/dataset/coverage, tính lại metric và đối chiếu cả năm metric với pytrec_eval trong lần tạo báo cáo này.', '',
             '## Thiết kế và điều kiện so sánh', '',
             'SciFact: 5.183 document, 300 query test. NFCorpus: 3.633 document, 323 query test. Các hệ thống trên cùng dataset có cùng corpus/query/qrels checksum. nDCG@10 dùng gain tuyến tính; Recall/MRR/MAP dùng relevance >0; query thiếu ranking tính 0. Gain mũ chỉ nằm trong cột legacy_exponential_ndcg@10 của CSV, không dùng để xếp hạng chính.', '',
             'Ba baseline gốc là ba hệ thống được chọn từ paper, không phải toàn bộ hệ thống BEIR. BM25 dùng Anserini/Pyserini 0.12.0, k1=0.9, b=0.4, title/contents weight 1.0; không thay bằng rank_bm25 V1. TAS-B dùng msmarco-distilbert-base-tas-b, dot-product không normalization. CE dùng ms-marco-MiniLM-L-6-v2, chỉ rerank candidate BM25 top-100. Upstream paper-era commit 52de18ae6f1ccc7426eb70895c0e43e1552ad493 được giữ nguyên; chưa có bằng chứng đây là SHA tạo ra bảng paper, runtime hiện tại có adapter tương thích.', '',
             'Qwen dùng Qwen3-Embedding-0.6B (query instruction; document không instruction), Qwen3-Reranker-0.6B (template yes/no chính thức). Revision embedding 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, reranker e61197ed45024b0ed8a2d74b80b4d909f1255473. Context 512, BF16, batch 1; reranker dùng prefix của cùng ranking embedding đóng băng và giữ nguyên candidate set. Prompt và revision chi tiết trong evidence.json/artifact nguồn.', '',
             '## Chất lượng đo thực tế', '']
    lines += ['Delta tuyệt đối = metric hệ thống − metric baseline; tăng tương đối (%) = delta / metric baseline × 100. Metric ở thang 0–1, các giá trị hiển thị làm tròn 6 chữ số; CSV giữ độ chính xác gốc. Không nhầm delta 0,01 với tăng tương đối 1%.', '']
    for dataset in RUNS:
        selected = [r for r in rows if r['dataset'] == dataset]
        lines += [f'### {dataset}', '', '| Hệ thống | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | MAP@100 |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in selected:
            lines.append('| '+r['system']+' | '+' | '.join(f'{r[m]:.6f}' for m in METRICS)+' |')
        lines += ['', '| Hệ thống | Delta nDCG so BM25 | Tăng tương đối | Delta Recall@100 so BM25 |', '|---|---:|---:|---:|']
        for r in selected[1:]:
            ndcg, recall = r['nDCG@10']-selected[0]['nDCG@10'], r['Recall@100']-selected[0]['Recall@100']
            lines.append(f"| {r['system']} | {ndcg:+.6f} | {ndcg/selected[0]['nDCG@10']*100:+.2f}% | {recall:+.6f} |")
        lines += ['', f'![nDCG {dataset}](ndcg_{dataset}.png)', '']
    lines += ['## Qwen đối chiếu từng baseline gốc', '',
              '| Dataset | Qwen | Delta nDCG so TAS-B | Tăng so TAS-B | Delta nDCG so BM25 + CE100 | Tăng so BM25 + CE100 |',
              '|---|---|---:|---:|---:|---:|']
    for dataset in RUNS:
        s = [r for r in rows if r['dataset'] == dataset]
        for r in s[3:]:
            tas, ce = r['nDCG@10']-s[1]['nDCG@10'], r['nDCG@10']-s[2]['nDCG@10']
            lines.append(f"| {dataset} | {r['system']} | {tas:+.6f} | {tas/s[1]['nDCG@10']*100:+.2f}% | {ce:+.6f} | {ce/s[2]['nDCG@10']*100:+.2f}% |")
    lines += ['', 'Qwen embedding vượt CE100 về nDCG trên cả hai dataset, nhưng MRR@10 trên NFCorpus (0,562534) thấp hơn CE100 (0,581858). Vì vậy kết luận tốt hơn phụ thuộc metric, không phải toàn bộ chất lượng đều tăng. Qwen top-20 có MAP@100 thấp hơn embedding trên NFCorpus do cắt candidate; top-50/100 khắc phục giới hạn này trong kết quả đã đo.', '',
              '## Candidate-depth và diễn giải Recall', '',
              '| Dataset | So sánh | Delta nDCG@10 | Tăng tương đối | Delta Recall@100 |', '|---|---|---:|---:|---:|']
    for dataset in RUNS:
        s = [r for r in rows if r['dataset'] == dataset]
        for a, b in [(4, 3), (5, 4), (6, 4), (6, 5)]:
            delta = s[a]['nDCG@10']-s[b]['nDCG@10']
            lines.append(f"| {dataset} | {s[a]['system']} so với {s[b]['system']} | {delta:+.6f} | {delta/s[b]['nDCG@10']*100:+.2f}% | {s[a]['Recall@100']-s[b]['Recall@100']:+.6f} |")
    lines += ['', 'Recall@100 tăng khi mở rộng top-20→50→100 vì candidate chứa thêm document liên quan. Trong cùng candidate set, reranker không thay đổi Recall@100; cả CE và Qwen đã được kiểm tra trực tiếp từng query. Qwen top-100 giữ Recall@100 bằng embedding top-100. Top-20 có thể thấp hơn baseline CE top-100 dù nDCG tốt hơn, vì candidate budget khác nhau; không suy diễn thành lỗi mất document.', '',
              'SciFact: top-50 gần bằng top-100 về nDCG (chỉ thêm 0,000374 điểm khi lên 100); MAP@100 của top-100 thấp hơn top-50 khoảng 0,000185 điểm. NFCorpus: top-100 tiếp tục cải thiện nDCG thêm 0,002662 điểm và MAP so với top-50. Chưa tính confidence interval hay paired significance test; không gọi các delta nhỏ là cải thiện có ý nghĩa thống kê.', '',
              '## Đối chiếu paper', '',
              '| Dataset | System ID | Paper nDCG@10 | Run nDCG@10 | Khớp làm tròn 3 chữ số |', '|---|---|---:|---:|---|']
    for p in papers:
        lines.append(f"| {p['dataset']} | {p['system_id']} | {float(p['paper_value']):.3f} | {p['measured_value']:.6f} | Có |")
    lines += ['', 'Nguồn: BEIR 2021 v4, Table 2, https://arxiv.org/html/2104.08663v4 . Số tham chiếu chỉ có nDCG@10, nằm riêng trong paper_reference.csv và được ánh xạ bằng system ID đầy đủ. Sáu số làm tròn khớp bảng không chứng minh runtime hoặc ranking byte-identical với tác giả. Qwen là experiment mở rộng, không gán số paper BM25 cho Qwen/CE.', '',
              '## Chi phí và giới hạn phép đo', '',
              'TAS-B/MiniLM CE và Qwen top-50/100 chạy A30/Ubuntu; Qwen embedding/top-20 dùng artifact laptop RTX 3070 BF16 có nghỉ 0,1 giây/batch và phiên bản PyTorch khác. Top-50/100 nghỉ cố định 0, pause 73→60°C, stop 80°C, bắt đầu model khi ≤55°C. Khác GPU/backend có thể gây sai khác số học nhỏ; so chất lượng được cùng dữ liệu, chưa có phép đo tốc độ công bằng trên cùng hardware/protocol.', '',
              '| Dataset | Qwen candidate k | Cold reranker (phút, gồm chờ nhiệt) | RAM peak GiB | VRAM allocated peak GiB |', '|---|---:|---:|---:|---:|']
    for r in performance:
        if r['run_id'] in [RUNS[d][i] for d in RUNS for i in [5, 6]]:
            k = 50 if 'top-50' in r['system'] else 100
            lines.append(f"| {r['dataset']} | {k} | {r['cold_total_seconds']/60:.2f} | {r['ram_peak_bytes']/2**30:.2f} | {r['vram_peak_bytes']/2**30:.2f} |")
    lines += ['', 'Phiên Qwen top-50/100 khoảng 2 giờ 20 phút kể cả chờ giữa model; nhiệt cao nhất quan sát 74°C, 40 lần chờ nhiệt, không chạm 80°C. Có workload dùng chung server trong lượt cuối. Các lượt server là quality-only, chưa đo warm latency/QPS. performance.csv giữ số đo có sẵn và ô trống cho chưa đo; cold CE/reranker-only không gồm retrieval, không cộng hay chia thành latency toàn pipeline. Không tính tỷ lệ latency với BM25 khi protocol khác.', '',
              '## Nhận xét theo RQ1–RQ4', '',
              '- RQ1 — Chất lượng: Qwen embedding vượt TAS-B và Anserini BM25 trên cả hai dataset về nDCG; Qwen reranker top-100 đạt nDCG cao nhất trong bảy cấu hình đã đo. CSV improvements chứa delta tuyệt đối/tương đối của cả năm metric với từng baseline.',
              '- RQ2 — Tổng quát hóa: bằng chứng chỉ gồm hai dataset thiên về khoa học/y sinh, chưa có domain tài chính, lập luận hoặc nhiều domain độc lập. Không kết luận tốt hơn trên toàn bộ BEIR.',
              '- RQ3 — Chi phí: candidate-k tăng làm tăng số cặp phải chấm; top-50 là lựa chọn đáng cân nhắc cho SciFact nếu ưu tiên nDCG và ngân sách, top-100 phù hợp khi ưu tiên Recall@100. Thời gian cold có chờ nhiệt chỉ phản ánh vận hành phiên này; cần warm benchmark cùng A30/protocol để so chi phí công bằng.',
              '- RQ4 — Khả năng tái lập: mỗi dòng có run ID, config, revision, checksum, ranking và môi trường. Project không fine-tune checkpoint nhưng BEIR có thể xuất hiện trong training của model mới; chưa đủ bằng chứng zero-shot nghiêm ngặt hoặc loại trừ data contamination.', '',
              '## Run nguồn và cách tái tạo báo cáo', '', '| Dataset | Hệ thống | Run ID |', '|---|---|---|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['system']} | [{r['run_id']}](../../../runs/{r['run_id']}/metrics.json) |")
    lines += ['', 'Chạy `.\\.venv-paper\\Scripts\\python.exe report_verified_comparison.py` để tạo thư mục báo cáo mới. Script chỉ đọc artifact; không tải model, không chạy GPU, không sửa status/validation/ranking cũ. integrity.json xác nhận hash trước/sau cho toàn bộ runs/ và results/. Sáu JSON V1 lịch sử không có evidence đầy đủ bị loại khỏi bảng chính, vẫn được giữ nguyên trong results/ và legacy/v1/results/.', '',
              'Đầu ra: measured_metrics.csv, improvements.csv, paper_reference.csv, performance.csv, evidence.json, integrity.json và hai biểu đồ. Các báo cáo trước vẫn giữ nguyên. Experiment chưa chạy trong báo cáo này: FiQA/ArguAna, hybrid→Qwen, warm benchmark đồng nhất và significance test.']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for dataset in RUNS:
        selected = [r for r in rows if r['dataset'] == dataset]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(['BM25', 'TAS-B', 'BM25 + MiniLM CE100', 'Qwen embedding', 'Qwen + reranker20',
                       'Qwen + reranker50', 'Qwen + reranker100'], [r['nDCG@10'] for r in selected],
                       color=['#68757f']*3+['#2474b5']+['#13856c']*3)
        ax.bar_label(bars, labels=[f"{r['nDCG@10']:.4f}" for r in selected], padding=4)
        ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel('nDCG@10 (linear gain)'); ax.set_title(dataset)
        fig.tight_layout(); fig.savefig(output/f'ndcg_{dataset}.png', dpi=180); plt.close(fig)
    after = snapshot()
    assert before == after, 'Historical artifact changed while reporting'
    legacy = list((ROOT/'results').glob('*.json'))
    assert len(legacy) == 6 and all(p.read_bytes() == (ROOT/'legacy/v1/results'/p.name).read_bytes() for p in legacy)
    (output/'integrity.json').write_text(json.dumps(dict(files_checked=len(before), all_unchanged=True,
        legacy_json_count=6, runs_verified=14, pytrec_eval='matched_all_five_metrics', artifact_sha256=before), indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
