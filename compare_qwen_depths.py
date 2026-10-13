"""Verify completed Qwen candidate-depth artifacts and compare quality only."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lab.data import ROOT, load_dataset
from lab.cache import file_digest
from lab.runner import verify_run, read_ranking
from evaluation.metrics import evaluate


def main():
    session = ROOT / 'reports/qwen_candidate_session_20261012_034345'
    entries = json.loads((session / 'session.json').read_text())
    assert len(entries) == 4 and all(e['status'] == 'VERIFIED' for e in entries)
    assert json.loads((session / 'historical_integrity.json').read_text())['all_unchanged']
    ids = {'scifact': {20: '20261011_125607_76912e7b'},
           'nfcorpus': {20: '20261011_143638_8d39eaff'}}
    for entry in entries:
        ids[entry['dataset']][entry['candidate_k']] = Path(entry['run']).name
    output = ROOT / 'reports/paper_reproduction/qwen_depths_20261012'
    output.mkdir(exist_ok=True)
    rows = []
    for dataset, depths in ids.items():
        baseline = None
        for depth, run_id in sorted(depths.items()):
            folder = ROOT / 'runs' / run_id
            assert json.loads((folder / 'status.json').read_text())['status'] == 'VERIFIED'
            # Only new runs are rewritten by verification; historical artifacts stay intact.
            metrics = verify_run(folder) if depth != 20 else json.loads((folder / 'metrics.json').read_text())
            meta = json.loads((folder / 'metadata.json').read_text())
            if depth == 20:
                _, _, qrels, _ = load_dataset(dataset)
                assert file_digest(folder / 'ranking.tsv') == meta['ranking_checksum']
                assert evaluate(read_ranking(folder / 'ranking.tsv'), qrels) == metrics
            signature = {k: meta[k] for k in ['corpus_checksum', 'query_checksum', 'qrels_checksum', 'reranker_revision']}
            if baseline is None:
                baseline, reference = metrics, signature
            assert signature == reference
            if depth != 20:
                validation = json.loads((folder / 'candidate_validation.json').read_text())
                assert validation['candidate_sets_preserved']
                assert abs(validation['before_topk_metrics']['Recall@100'] - metrics['Recall@100']) < 1e-12
            rows.append(dict(dataset=dataset, candidate_k=depth, run_id=run_id,
                ndcg10=metrics['nDCG@10'], recall100=metrics['Recall@100'], map100=metrics['MAP@100'],
                ndcg10_delta_top20=metrics['nDCG@10']-baseline['nDCG@10'],
                ndcg10_relative_percent_top20=(metrics['nDCG@10']/baseline['nDCG@10']-1)*100,
                recall100_delta_top20=metrics['Recall@100']-baseline['Recall@100']))
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Qwen reranker: top-20, top-50 và top-100', '',
             '| Dataset | Candidate k | nDCG@10 | Delta top-20 | Tương đối | Recall@100 | MAP@100 | Run ID |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['candidate_k']} | {r['ndcg10']:.6f} | {r['ndcg10_delta_top20']:+.6f} | {r['ndcg10_relative_percent_top20']:+.2f}% | {r['recall100']:.6f} | {r['map100']:.6f} | {r['run_id']} |")
    lines += ['', 'SciFact: top-50 tăng nDCG 1,48% so với top-20; lên top-100 chỉ thêm 0,000374 điểm nDCG, MAP@100 giảm nhẹ 0,000185 điểm so với top-50. NFCorpus: top-100 đạt nDCG cao nhất, tăng 2,51% so với top-20 và thêm 0,002662 điểm so với top-50. Chưa kiểm tra ý nghĩa thống kê của các delta nhỏ.', '',
              '| Dataset | Candidate k | Cold reranker (phút) | RAM peak (GiB) | VRAM allocated peak (GiB) |',
              '|---|---:|---:|---:|---:|']
    for entry in entries:
        timing = json.loads((ROOT / 'runs' / Path(entry['run']).name / 'timing.json').read_text())
        lines.append(f"| {entry['dataset']} | {entry['candidate_k']} | {timing['cold']['total_seconds']/60:.2f} | {timing['RAM_peak_bytes']/2**30:.2f} | {timing['VRAM_peak_bytes']/2**30:.2f} |")
    thermal = json.loads((session / 'thermal_summary.json').read_text())
    lines += ['', f"Phiên kéo dài khoảng 2 giờ 20 phút kể cả chờ giữa model. Nhiệt cao nhất quan sát {thermal['peak_observed_c']}°C; {thermal['cooling_pause_count']} lần chờ nhiệt, không chạm ngưỡng dừng cứng. {json.loads((session/'historical_integrity.json').read_text())['files_checked']} tệp artifact trước phiên giữ nguyên. GPU có tiến trình dùng chung xuất hiện trong lượt cuối; thời gian này không phải benchmark server độc quyền.",
              '', 'Top-50/100 chạy thực trên A30/Ubuntu, batch 1, BF16, context 512, không nghỉ cố định. Tạm dừng khi 73°C tới 60°C; dừng cứng 80°C. Candidate là prefix của cùng ranking Qwen embedding đã đóng băng; reranking giữ nguyên candidate set. Recall@100 tăng do mở rộng candidate, không phải do reranker bổ sung document.',
              '', 'Top-20 dùng lại kết quả laptop đã VERIFIED, cùng revision/prompt/context/precision nhưng khác phiên bản PyTorch và GPU. So sánh chất lượng có giới hạn sai khác số học giữa backend; không so latency/QPS như cùng protocol. Chỉ đo cold reranker-only, gồm thời gian chờ nhiệt; chưa có warm benchmark.',
              '', 'Bốn run mới được đọc lại ranking/checksum và tính lại metric; bằng chứng pytrec_eval nằm trong validation.json. Kết quả lịch sử được bảo toàn theo historical_integrity.json của phiên server.',
              '', f'Bằng chứng phiên: `{session.relative_to(ROOT)}`. Tái tính: `.\\.venv-paper\\Scripts\\python.exe compare_qwen_depths.py`.']
    (output / 'comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, dataset in zip(axes, ids):
        selected = [r for r in rows if r['dataset'] == dataset]
        bars = ax.bar([str(r['candidate_k']) for r in selected], [r['ndcg10'] for r in selected], color=['#68757f', '#2474b5', '#13856c'])
        ax.bar_label(bars, labels=[f"{r['ndcg10']:.4f}" for r in selected], padding=3)
        ax.set_title(dataset); ax.set_xlabel('Candidate k'); ax.set_ylim(0, 1)
    axes[0].set_ylabel('nDCG@10 (linear gain)'); fig.tight_layout()
    fig.savefig(output / 'ndcg_comparison.png', dpi=180); plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
