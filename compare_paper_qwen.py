"""Compare verified paper BM25 runs with preserved Qwen runs; no new inference."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from lab.cache import atomic_json
from lab.data import ROOT
from lab.runner import verify_run

RUNS = {
    'scifact': ['20261012_011743_164d4865', '20261011_123318_86ae3723', '20261011_125607_76912e7b'],
    'nfcorpus': ['20261012_011854_5304b15c', '20261011_141533_2f976292', '20261011_143638_8d39eaff'],
}
PUBLISHED = {'scifact': 0.665, 'nfcorpus': 0.325}
output = ROOT / 'reports/paper_reproduction'
output.mkdir(exist_ok=True)
rows = []
for dataset, run_ids in RUNS.items():
    baseline = None
    checksum = None
    for label, run_id in zip(['paper-anserini-bm25', 'qwen3-embedding-0.6b', 'qwen3-embedding-0.6b+qwen3-reranker-0.6b+k20'], run_ids):
        folder = ROOT / 'runs' / run_id
        assert json.loads((folder / 'status.json').read_text())['status'] == 'VERIFIED'
        metrics = verify_run(folder)
        metadata = json.loads((folder / 'metadata.json').read_text())
        if baseline is None:
            baseline, checksum = metrics, {key: metadata[key] for key in ['corpus_checksum', 'query_checksum', 'qrels_checksum']}
            assert round(metrics['nDCG@10'], 3) == PUBLISHED[dataset]
        assert all(metadata[key] == value for key, value in checksum.items())
        rows.append(dict(dataset=dataset, system=label, run_id=run_id,
                         ndcg10=metrics['nDCG@10'], recall100=metrics['Recall@100'], map100=metrics['MAP@100'],
                         delta_ndcg10=metrics['nDCG@10'] - baseline['nDCG@10'],
                         relative_ndcg10_percent=100 * (metrics['nDCG@10'] / baseline['nDCG@10'] - 1),
                         paper_bm25_ndcg10=PUBLISHED[dataset], latency_comparison=None))
with (output / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
lines = ['# BM25 paper BEIR 2021 và Qwen', '',
         '| Dataset | System | nDCG@10 | Delta BM25 | Relative | Recall@100 | MAP@100 | Run |',
         '|---|---|---:|---:|---:|---:|---:|---|']
for row in rows:
    lines.append(f"| {row['dataset']} | {row['system']} | {row['ndcg10']:.6f} | {row['delta_ndcg10']:+.6f} | {row['relative_ndcg10_percent']:+.2f}% | {row['recall100']:.6f} | {row['map100']:.6f} | {row['run_id']} |")
lines += ['', 'Baseline: Anserini/Pyserini 0.12.0, k1=0.9/b=0.4, title/contents riêng weight=1.0, top-1000. Hai baseline khớp Table 2 khi làm tròn 3 chữ số.',
          '', 'Upstream SHA 52de18ae6f1ccc7426eb70895c0e43e1552ad493 là commit cuối trước paper v4; chưa chứng minh SHA thực nghiệm gốc. Python/dependency của wrapper là phiên bản hiện tại được ghi trong artifact, không phải runtime byte-identical năm 2021.',
          '', 'Qwen dùng lại run VERIFIED đã lưu, BF16/context 512; reranker chỉ top-20, giữ đúng candidate set. Không so latency với BM25 CPU/Docker vì protocol và thiết bị khác nhau. Chưa chạy TAS-B, DPR, ANCE hay BM25+CE top-100 của paper trong lượt này.',
          '', 'Sáu JSON lịch sử giữ nguyên. Lượt lỗi Java heap và các lượt kiểm tra lặp lưu riêng; chỉ run IDs trong bảng là đối chứng được chọn.',
          '', 'Nguồn: https://arxiv.org/html/2104.08663v4 (Section 4, Table 2, Table 6); checkout upstream/beir nguyên trạng. Xem docs/PAPER_REPRODUCTION_AUDIT.md.',
          '', 'Lệnh tính lại: `.\\.venv-paper\\Scripts\\python.exe compare_paper_qwen.py`.']
(output / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
historical = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() ==
              hashlib.sha256((ROOT / 'legacy/v1/results' / p.name).read_bytes()).hexdigest()
              for p in (ROOT / 'results').glob('*.json')}
assert len(historical) == 6 and all(historical.values())
atomic_json(output / 'historical_integrity.json', historical)
print(json.dumps(rows, indent=2))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for axis, (dataset, run_ids) in zip(axes, RUNS.items()):
    selected = [row for row in rows if row['dataset'] == dataset]
    bars = axis.bar(['BM25 paper', 'Qwen embedding', 'Qwen + reranker 20'], [r['ndcg10'] for r in selected], color=['#68757f', '#2474b5', '#13856c'])
    axis.bar_label(bars, labels=[f"{r['ndcg10']:.4f}" for r in selected], padding=4)
    axis.set_title(dataset)
    axis.set_ylim(0, 1)
    axis.tick_params(axis='x', labelrotation=15)
axes[0].set_ylabel('nDCG@10 (linear gain)')
fig.tight_layout()
fig.savefig(output / 'ndcg_comparison.png', dpi=180)
plt.close(fig)
