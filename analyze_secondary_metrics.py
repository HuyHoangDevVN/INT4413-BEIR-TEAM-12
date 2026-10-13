"""Extend the saved nDCG report with three paired query-level metrics."""
import json
import csv
import shutil
from datetime import datetime, timezone
import numpy as np
from evaluation.metrics import evaluate
from evaluation.significance import paired_inference, holm
from report_verified_comparison import ROOT, RUNS, SYSTEMS, METRICS, snapshot, verify_readonly, write_csv
from lab.data import load_dataset
from lab.cache import file_digest

BASE = ROOT/'reports/paper_reproduction/significance_comparison_20261012_070142'
SECONDARY = ['Recall@100', 'MRR@10', 'MAP@100']


def verdict(row, key='p_holm_secondary108'):
    if row[key] >= .05:
        return 'Chưa đủ bằng chứng'
    if row['mean_delta'] > 0:
        return 'Tăng có ý nghĩa'
    if row['mean_delta'] < 0:
        return 'Giảm có ý nghĩa'
    return 'Không đổi'


def main():
    before = snapshot()
    prior = {p.name: file_digest(p) for p in BASE.iterdir() if p.is_file()}
    comparisons = [(i, j, 'qwen_vs_original_baseline') for i in range(3, 7) for j in range(3)]
    comparisons += [(5, 4, 'candidate_depth'), (6, 4, 'candidate_depth'), (6, 5, 'candidate_depth')]
    comparisons += [(i, 3, 'reranker_vs_embedding') for i in range(4, 7)]
    results, per_query, score_cache = [], [], {}
    for dataset, ids in RUNS.items():
        corpus, queries, qrels, data = load_dataset(dataset)
        qids = sorted(qrels)
        scores = []
        for label, run_id in zip(SYSTEMS, ids):
            metrics, ranking, _ = verify_readonly(ROOT/'runs'/run_id, corpus, queries, qrels, data)
            values = [evaluate({q: ranking[q]}, {q: qrels[q]}) for q in qids]
            assert all(abs(np.mean([m[k] for m in values])-metrics[k]) < 1e-12 for k in METRICS)
            scores.append({k: np.array([m[k] for m in values]) for k in SECONDARY})
            score_cache[(dataset, run_id)] = scores[-1]
            per_query += [dict(dataset=dataset, query_id=q, system=label, run_id=run_id, **m) for q,m in zip(qids, values)]
        for metric in SECONDARY:
            for first, second, group in comparisons:
                row = dict(dataset=dataset, system=SYSTEMS[first], baseline=SYSTEMS[second], comparison_group=group,
                           metric=metric, first_run=ids[first], baseline_run=ids[second])
                row.update(paired_inference(scores[first][metric], scores[second][metric], seed=1000+len(results)))
                results.append(row)
        print(dataset, 'rankings verified; three secondary metrics analyzed', flush=True)
    assert len(results) == 108
    ndcg = list(csv.DictReader((BASE/'significance.csv').open(encoding='utf-8-sig')))
    assert len(ndcg) == 36
    pilot_secondary = holm([r['p_raw'] for r in results])
    pilot_global = holm([float(r['p_raw']) for r in ndcg] + [r['p_raw'] for r in results])[36:]
    for index, row in enumerate(results):
        row['pilot_p_raw'] = row['p_raw']
        row['monte_carlo_refined'] = False
        # Refine near-threshold Monte Carlo decisions with a fresh fixed-size sample.
        if row['permutation_method'].startswith('monte_carlo') and any(.025 < p < .075 for p in [pilot_secondary[index], pilot_global[index]]):
            row.update(paired_inference(score_cache[(row['dataset'], row['first_run'])][row['metric']],
                score_cache[(row['dataset'], row['baseline_run'])][row['metric']], seed=2000+index, permutations=1000000))
            row['monte_carlo_refined'] = True
    global_p = holm([float(r['p_raw']) for r in ndcg] + [r['p_raw'] for r in results])
    for row, p, global_adjusted in zip(results, holm([r['p_raw'] for r in results]), global_p[36:]):
        row.update(p_holm_secondary108=float(p), p_holm_all144=float(global_adjusted))
        row.update(conclusion_secondary108=verdict(row), conclusion_all144=verdict(row, 'p_holm_all144'))
    output = BASE.parent/('multimetric_significance_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    for p in BASE.iterdir():
        if p.is_file():
            shutil.copy2(p, output/p.name)
    write_csv(output/'secondary_significance.csv', results)
    # Original nDCG p-values and primary-family decisions remain untouched in significance.csv.
    all_rows = [dict(dataset=r['dataset'], system=r['system'], baseline=r['baseline'], metric=r['metric'],
        mean_delta=float(r['mean_delta']), p_raw=float(r['p_raw']), p_holm_all144=float(p)) for r,p in zip(ndcg, global_p[:36])]
    all_rows += [{k:r[k] for k in all_rows[0]} for r in results]
    write_csv(output/'all_metrics_global144.csv', all_rows)
    write_csv(output/'per_query_metrics.csv', per_query)
    assert (output/'per_query_metrics.csv').read_bytes() == (BASE/'per_query_metrics.csv').read_bytes()
    protocol = dict(metrics=SECONDARY, n_queries={'scifact':300, 'nfcorpus':323},
        method='paired two-sided sign-flip randomization; paired percentile bootstrap', alpha=.05,
        secondary_family_size=108, secondary_correction='Holm across all 3 secondary metrics and both datasets',
        sensitivity_family_size=144, sensitivity_correction='Holm including 36 previous nDCG tests',
        primary_nDCG_family_size=36, primary_nDCG_results_preserved=True,
        permutations=100000, exact_if_nonzero_pairs_at_most=16, bootstrap_samples=20000,
        refinement='Fresh 1000000 permutations, seed 2000+comparison_index if pilot Holm108 or Holm144 in (0.025,0.075)',
        ci='95% pointwise unadjusted; not simultaneous', seeds=[r['seed'] for r in results],
        numpy_version=np.__version__, post_hoc=True, model_execution=False)
    (output/'secondary_statistical_method.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['## Kiểm định bổ sung Recall@100, MRR@10 và MAP@100', '',
        'Dùng lại cùng 14 ranking, đầy đủ 300 query SciFact và 323 query NFCorpus; mean của metric theo query khớp artifact và cả năm metric được đối chiếu lại pytrec_eval. Không chạy model, không dùng các lần timing làm mẫu độc lập.', '',
        'Phương pháp giống nDCG: paired randomization hai phía 100.000 mẫu, exact nếu ≤16 cặp khác 0; p Monte Carlo plus-one; CI paired percentile bootstrap 95% với 20.000 mẫu. Seed mặc định 1000–1107. Nếu p Holm108 hoặc Holm144 pilot nằm trong (0,025; 0,075), tính lại bằng 1.000.000 mẫu độc lập với seed 2000+index rồi hiệu chỉnh lại toàn family; CSV lưu pilot và kết quả cuối. Đây là refinement hậu nghiệm để giảm sai số Monte Carlo gần ngưỡng, không đổi dữ liệu hay lựa chọn comparison. CI pointwise chưa hiệu chỉnh. Phân tích bổ sung hậu nghiệm, không đăng ký trước.', '',
        '**Family bổ sung gồm 108 phép thử**: cùng 36 so sánh trên mỗi metric × 3 metric, hiệu chỉnh Holm toàn bộ 108, alpha 0,05. Kết luận nDCG family 36 trong bản trước được giữ nguyên. Cột p Holm144 là kiểm tra độ nhạy khi xét chung cả 144 phép thử; không chọn riêng family thuận lợi cho từng dòng.', '',
        'Dấu delta xác định tăng/giảm; p hiệu chỉnh xác định bằng chứng khác biệt. Không bác bỏ H0 không chứng minh tương đương. Không suy từ CI pointwise sang quyết định đã hiệu chỉnh. Chỉ xét bất định theo query, chưa xét training seed, query phụ thuộc chủ đề, domain mới hay qrels thiếu.', '']
    lines += ['### Kết luận chính cho Qwen top-100', '']
    for dataset in RUNS:
        for metric in SECONDARY:
            subset = [r for r in results if r['dataset']==dataset and r['metric']==metric
                and r['system']==SYSTEMS[6] and r['comparison_group']=='qwen_vs_original_baseline']
            passed = [r['baseline'] for r in subset if r['conclusion_secondary108']=='Tăng có ý nghĩa']
            failed = [r['baseline'] for r in subset if r['conclusion_secondary108']!='Tăng có ý nghĩa']
            lines.append(f"- {dataset}, {metric}: tăng có ý nghĩa so với {', '.join(passed) if passed else 'không baseline nào'}; chưa đủ bằng chứng tăng so với {', '.join(failed) if failed else 'không còn baseline nào trong ba baseline'}.")
    lines += ['']
    for metric in SECONDARY:
        lines += [f'### {metric}', '',
            '| Dataset | Hệ thống vs baseline | Delta | CI 95% pointwise | p raw | p Holm108 | p Holm144 | Kết luận Holm108 | W/L/T |',
            '|---|---|---:|---|---:|---:|---:|---|---|']
        for r in results:
            if r['metric'] != metric:
                continue
            label = (r['system']+' vs '+r['baseline']).replace('Qwen embedding → Qwen reranker', 'Qwen reranker')
            lines.append(f"| {r['dataset']} | {label} | {r['mean_delta']:+.6f} | [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] | {r['p_raw']:.6g} | {r['p_holm_secondary108']:.6g} | {r['p_holm_all144']:.6g} | {r['conclusion_secondary108']} | {r['wins']}/{r['losses']}/{r['ties']} |")
    lines += ['', '### Tổng hợp Qwen top-100 so với ba baseline', '',
        '| Dataset | Metric | Baseline | Delta | p Holm108 | p Holm144 | Kết luận |', '|---|---|---|---:|---:|---:|---|']
    for r in results:
        if r['system'] == SYSTEMS[6] and r['comparison_group'] == 'qwen_vs_original_baseline':
            lines.append(f"| {r['dataset']} | {r['metric']} | {r['baseline']} | {r['mean_delta']:+.6f} | {r['p_holm_secondary108']:.6g} | {r['p_holm_all144']:.6g} | {r['conclusion_secondary108']} |")
    lines += ['', '### Candidate-depth và kiểm tra Recall', '']
    for metric in SECONDARY:
        for r in results:
            if r['metric'] == metric and r['comparison_group'] == 'candidate_depth':
                lines.append(f"- {r['dataset']}, {metric}, {r['system']} vs {r['baseline']}: delta {r['mean_delta']:+.6f}; {r['conclusion_secondary108']}; p Holm108={r['p_holm_secondary108']:.6g}, p Holm144={r['p_holm_all144']:.6g}.")
    equal = [r for r in results if r['metric']=='Recall@100' and r['system']==SYSTEMS[6] and r['baseline']==SYSTEMS[3]]
    assert len(equal)==2 and all(r['mean_delta']==0 and r['p_raw']==1 and r['ties']==r['n_queries'] for r in equal)
    changed = [r for r in results if r['conclusion_secondary108'] != r['conclusion_all144']]
    lines += ['', 'Qwen reranker top-100 và embedding có Recall@100 **bằng nhau ở từng query** trên cả hai dataset: delta=0, CI=[0,0], p=1. Đây là do giữ nguyên candidate set top-100, không phải bằng chứng reranker nâng recall. Top-20/50 bị giới hạn candidate; tăng recall khi tăng K là hiệu ứng thêm candidate, không phải model reranker bổ sung document.', '',
        f'Số kết luận bổ sung đổi khi chuyển từ Holm108 sang Holm144: {len(changed)}.']
    for r in changed:
        lines.append(f"- {r['dataset']}, {r['metric']}, {r['system']} vs {r['baseline']}: {r['conclusion_secondary108']} → {r['conclusion_all144']}.")
    lines += ['', 'Recall@10 và legacy_exponential_ndcg@10 vẫn chỉ mô tả, chưa kiểm định. Các hạn chế về query dependence, test-set post-hoc, khác GPU giữa top-20 và top-50/100, thiếu qrels và zero-shot của phần nDCG tiếp tục áp dụng.', '',
        'Tái tạo: `.\\.venv-paper\\Scripts\\python.exe analyze_secondary_metrics.py`. secondary_significance.csv lưu 108 phép bổ sung, all_metrics_global144.csv lưu sensitivity toàn bộ, secondary_statistical_method.json lưu protocol/seed. significance.csv và statistical_method.json giữ nguyên kết quả nDCG trước; integrity.json chứng minh artifact và bản báo cáo trước không bị sửa.', '']
    section = '\n'.join(lines)
    (output/'secondary_statistical_analysis.md').write_text(section, encoding='utf-8')
    primary=(BASE/'statistical_analysis.md').read_text(encoding='utf-8')
    primary=primary.replace('Bốn metric phụ và legacy gain mũ được lưu trong per_query_metrics.csv, nhưng **chưa kiểm định ý nghĩa thống kê**; không suy từ nDCG sang Recall/MRR/MAP.',
        'Recall@100, MRR@10 và MAP@100 đã được kiểm định riêng trong [báo cáo bổ sung](secondary_statistical_analysis.md); Recall@10 và legacy gain mũ vẫn chỉ mô tả.')
    (output/'statistical_analysis.md').write_text(primary, encoding='utf-8')
    text=(BASE/'comparison.md').read_text(encoding='utf-8')
    text=text.replace('Bốn metric phụ và legacy gain mũ được lưu trong per_query_metrics.csv, nhưng **chưa kiểm định ý nghĩa thống kê**; không suy từ nDCG sang Recall/MRR/MAP.',
        'Recall@100, MRR@10 và MAP@100 đã được kiểm định riêng trong phần bổ sung bên dưới; Recall@10 và legacy gain mũ vẫn chỉ mô tả. Không suy kết luận của một metric sang metric khác.')
    text=text.replace('## Run nguồn và cách tái tạo báo cáo', section+'\n## Run nguồn và cách tái tạo báo cáo')
    (output/'comparison.md').write_text(text, encoding='utf-8')
    assert snapshot()==before
    assert all(file_digest(BASE/name)==value for name,value in prior.items())
    (output/'integrity.json').write_text(json.dumps(dict(all_unchanged=True, source_artifact_files_checked=len(before),
        previous_report_unchanged=True, previous_report_sha256=prior, artifact_sha256=before,
        secondary_tests=108, global_sensitivity_tests=144, model_execution=False), indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
