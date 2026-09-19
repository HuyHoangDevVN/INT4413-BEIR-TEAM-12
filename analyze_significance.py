"""Update a copied report using saved rankings only; no model execution."""
import json
import shutil
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from evaluation.metrics import evaluate
from evaluation.significance import paired_inference, holm
from report_verified_comparison import ROOT, RUNS, SYSTEMS, METRICS, snapshot, verify_readonly, write_csv
from lab.data import load_dataset
from lab.cache import file_digest

BASE_REPORT = ROOT/'reports/paper_reproduction/complete_comparison_20261012_064753'


def main():
    before = snapshot()
    base_hashes = {p.name: file_digest(p) for p in BASE_REPORT.iterdir() if p.is_file()}
    output = BASE_REPORT.parent/('significance_comparison_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    comparisons = [(i, j, 'qwen_vs_original_baseline') for i in range(3, 7) for j in range(3)]
    comparisons += [(5, 4, 'candidate_depth'), (6, 4, 'candidate_depth'), (6, 5, 'candidate_depth')]
    comparisons += [(i, 3, 'reranker_vs_embedding') for i in range(4, 7)]
    assert len(comparisons) == 18
    results, per_query, provenance = [], [], []
    for dataset, ids in RUNS.items():
        corpus, queries, qrels, data = load_dataset(dataset)
        qids = sorted(qrels)
        scores = []
        for label, run_id in zip(SYSTEMS, ids):
            metrics, ranking, meta = verify_readonly(ROOT/'runs'/run_id, corpus, queries, qrels, data)
            query_metrics = [evaluate({q: ranking[q]}, {q: qrels[q]}) for q in qids]
            assert all(abs(np.mean([m[k] for m in query_metrics])-metrics[k]) < 1e-12 for k in METRICS)
            scores.append(np.array([m['nDCG@10'] for m in query_metrics]))
            per_query += [dict(dataset=dataset, query_id=q, system=label, run_id=run_id, **m)
                          for q, m in zip(qids, query_metrics)]
            provenance.append(dict(dataset=dataset, run_id=run_id, ranking_sha256=meta['ranking_checksum'],
                                   query_count=len(qids), pytrec_eval='matched_all_five_metrics'))
        for first, second, group in comparisons:
            row = dict(dataset=dataset, system=SYSTEMS[first], baseline=SYSTEMS[second], comparison_group=group,
                       metric='nDCG@10', first_run=ids[first], baseline_run=ids[second])
            row.update(paired_inference(scores[first], scores[second], seed=42+len(results)))
            results.append(row)
        print(dataset, 'verified; paired inference complete', flush=True)
    assert len(results) == 36
    for row, adjusted in zip(results, holm([r['p_raw'] for r in results])):
        row.update(p_holm=float(adjusted), significant_holm=bool(adjusted < .05),
                   significant_positive=bool(adjusted < .05 and row['mean_delta'] > 0))
    output.mkdir()
    for path in BASE_REPORT.iterdir():
        if path.is_file():
            shutil.copy2(path, output/path.name)
    write_csv(output/'significance.csv', results)
    write_csv(output/'per_query_metrics.csv', per_query)
    method = dict(primary_metric='nDCG@10', unit='paired_query_within_dataset', alpha=.05,
        alternative='two_sided', family_size=36, multiplicity='Holm across all 36 nDCG comparisons, both datasets',
        permutations=100000, exact_if_nonzero_pairs_at_most=16, monte_carlo_plus_one=True,
        bootstrap_samples=20000, bootstrap_ci='paired percentile 95%, pointwise and unadjusted',
        seeds=[r['seed'] for r in results], numpy_version=np.__version__,
        analysis_type='post_hoc_existing_fixed_test_runs; not preregistered', provenance=provenance)
    (output/'statistical_method.json').write_text(json.dumps(method, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['## Kiểm tra ý nghĩa thống kê từ ranking đã lưu', '',
        'Metric chính được chọn trước khi xem kết quả kiểm định là **nDCG@10 gain tuyến tính**. Đơn vị là query: SciFact 300, NFCorpus 323. Dùng đầy đủ query có qrels, kể cả query ranking rỗng; không coi document hoặc các lượt timing lặp là mẫu độc lập. Đây là phân tích hậu nghiệm các test run hiện có, không phải nghiên cứu đăng ký trước.', '',
        '**Phương pháp:** paired randomization hai phía trên hiệu nDCG từng query (đổi dấu ngẫu nhiên, 100.000 mẫu; duyệt chính xác nếu ≤16 cặp khác 0). P Monte Carlo dùng (số mẫu cực đoan + 1)/(100.000 + 1), không xuất p=0. Bootstrap ghép cặp 20.000 mẫu query có hoàn lại cho CI 95% percentile của mean delta. Seed 42–77 cố định, tính theo chunk trên CPU.', '',
        '**Hiệu chỉnh nhiều phép thử:** một family gồm 36 phép nDCG trên cả hai dataset: 24 Qwen so với ba baseline gốc, 6 so giữa top-20/50/100, 6 reranker so với embedding. Kết luận có ý nghĩa dựa vào p Holm <0,05 và mean delta >0. CI 95% là pointwise chưa hiệu chỉnh; không dùng CI riêng để thay thế quyết định Holm. Holm kiểm soát family-wise error dưới các giả định hợp lệ của p-value.', '',
        'Tham khảo phương pháp: [nghiên cứu randomization trong IR của Smucker và cộng sự](https://ciir-publications.cs.umass.edu/getpdf.php?id=885), [R documentation: p.adjust/Holm](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).', '',
        '| Dataset | Qwen so với baseline | Delta nDCG | CI 95% pointwise | p raw | p Holm | Tăng có ý nghĩa? | W/L/T query |',
        '|---|---|---:|---|---:|---:|---|---|']
    for r in results:
        short = r['system'].replace('Qwen embedding → Qwen reranker ', 'Qwen reranker ')
        base = r['baseline'].replace('Qwen embedding → Qwen reranker ', 'Qwen reranker ')
        lines.append(f"| {r['dataset']} | {short} vs {base} | {r['mean_delta']:+.6f} | [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] | {r['p_raw']:.6g} | {r['p_holm']:.6g} | {'Có' if r['significant_positive'] else 'Chưa đủ bằng chứng'} | {r['wins']}/{r['losses']}/{r['ties']} |")
    lines += ['', 'W/L/T là số query có delta >1e-12 / <-1e-12 / còn lại. Trọng số mỗi query bằng nhau; delta mean, CI và kiểm định dùng số thực chưa làm tròn. P-value không phải xác suất giả thuyết đúng hoặc thước đo độ lớn cải thiện.', '', '### Diễn giải sau hiệu chỉnh', '']
    lines += ['Tổng hợp: Qwen reranker top-20/50/100 vượt cả ba baseline về nDCG có ý nghĩa sau Holm trên cả hai dataset; reranker cũng vượt embedding có ý nghĩa. Embedding vượt TAS-B có ý nghĩa trên hai dataset, vượt BM25 có ý nghĩa trên NFCorpus nhưng chưa đủ bằng chứng trên SciFact; chưa vượt BM25+MiniLM CE có ý nghĩa trên dataset nào. Cả sáu phép so candidate-depth chưa đủ bằng chứng tăng có ý nghĩa sau Holm.', '',
              'NFCorpus top-100 so với top-20 có p raw≈0,0377 và CI pointwise không chứa 0, nhưng p Holm≈0,3396: **không** kết luận tăng có ý nghĩa sau nhiều phép thử. Điều này minh họa vì sao không chọn riêng một p raw thuận lợi rồi bỏ hiệu chỉnh.', '']
    for dataset in RUNS:
        for index in range(3, 7):
            subset = [r for r in results if r['dataset'] == dataset and r['system'] == SYSTEMS[index]
                      and r['comparison_group'] == 'qwen_vs_original_baseline']
            passed = [r['baseline'] for r in subset if r['significant_positive']]
            failed = [r['baseline'] for r in subset if not r['significant_positive']]
            lines.append(f"- {dataset}, {SYSTEMS[index]}: tăng có ý nghĩa so với {', '.join(passed) if passed else 'không baseline nào trong ba baseline'}; chưa đủ bằng chứng so với {', '.join(failed) if failed else 'không có baseline còn lại'}.")
        depth = [r for r in results if r['dataset'] == dataset and r['comparison_group'] == 'candidate_depth']
        for r in depth:
            lines.append(f"- {dataset}, {r['system']} so với {r['baseline']}: delta {r['mean_delta']:+.6f}, p Holm={r['p_holm']:.6g}; {'tăng có ý nghĩa' if r['significant_positive'] else 'chưa đủ bằng chứng tăng có ý nghĩa'}.")
    lines += ['', '**Giới hạn:** không bác bỏ H0 không chứng minh hai hệ thống tương đương; chưa làm equivalence test. Bootstrap/query randomization giả định query là đơn vị có thể lấy mẫu/hoán đổi phù hợp; query cùng chủ đề có thể tương quan. CI và p chỉ phản ánh bất định theo query trên hai test set, không phản ánh biến động training/model seed, domain mới hoặc thiếu qrels. Không kết luận causal hoặc zero-shot sạch contamination. Top-20 laptop và top-50/100 A30 khác backend; kiểm định không tách riêng ảnh hưởng số học của GPU khỏi candidate-depth.', '',
        'Bốn metric phụ và legacy gain mũ được lưu trong per_query_metrics.csv, nhưng **chưa kiểm định ý nghĩa thống kê**; không suy từ nDCG sang Recall/MRR/MAP. Tỷ lệ Recall@100 bất biến trong cùng candidate set là tính chất đã kiểm tra trực tiếp, không cần kiểm định để chứng minh.', '',
        'Tái tạo: `.\\.venv-paper\\Scripts\\python.exe analyze_significance.py`. Chỉ dùng CPU và ranking có sẵn, không tải/nạp/chạy model. statistical_method.json lưu seed/protocol; significance.csv lưu 36 phép thử; integrity.json xác nhận toàn bộ artifact nguồn và báo cáo cũ giữ nguyên.', '']
    section = '\n'.join(lines)
    (output/'statistical_analysis.md').write_text(section, encoding='utf-8')
    text = (BASE_REPORT/'comparison.md').read_text(encoding='utf-8')
    text = text.replace('Chưa tính confidence interval hay paired significance test; không gọi các delta nhỏ là cải thiện có ý nghĩa thống kê.',
                        'Đã kiểm tra CI và paired significance ở phần thống kê bên dưới; quyết định dùng p Holm, không chỉ dựa vào delta.')
    text = text.replace('và significance test.', 'và equivalence test. Significance nDCG đã được bổ sung bên dưới.')
    text = text.replace('## Run nguồn và cách tái tạo báo cáo', section+'\n## Run nguồn và cách tái tạo báo cáo')
    (output/'comparison.md').write_text(text, encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    for ax, dataset in zip(axes, RUNS):
        subset = [r for r in results if r['dataset'] == dataset and
                  r['baseline'] == 'Anserini BM25' and r['comparison_group'] == 'qwen_vs_original_baseline']
        for i, r in enumerate(subset):
            ax.plot([r['ci95_low'], r['ci95_high']], [i, i], color='#2474b5')
            ax.scatter(r['mean_delta'], i, color='#13856c' if r['significant_positive'] else '#68757f')
        ax.set_yticks(range(4), ['Embedding', 'Reranker20', 'Reranker50', 'Reranker100'])
        ax.invert_yaxis(); ax.axvline(0, color='black', linestyle='--', linewidth=.8)
        ax.set_title(dataset); ax.set_xlabel('Mean nDCG@10 delta vs Anserini BM25')
    fig.suptitle('Paired bootstrap 95% pointwise CI; significance uses Holm-adjusted p')
    fig.tight_layout(); fig.savefig(output/'ndcg_delta_ci.png', dpi=180); plt.close(fig)
    assert snapshot() == before
    assert all(file_digest(BASE_REPORT/name) == value for name, value in base_hashes.items())
    integrity = dict(source_artifact_files_checked=len(before), all_unchanged=True,
        previous_report_unchanged=True, previous_report_sha256=base_hashes, artifact_sha256=before,
        runs_verified=14, nDCG_tests=36, gpu_model_execution=False)
    (output/'integrity.json').write_text(json.dumps(integrity, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
