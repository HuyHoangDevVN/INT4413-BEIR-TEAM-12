"""Read and verify the complete two-dataset baseline comparison; no inference."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lab.data import ROOT
from lab.runner import verify_run

systems = ['Anserini BM25', 'TAS-B', 'Anserini BM25 → MiniLM CE top-100', 'Qwen embedding', 'Qwen embedding → Qwen reranker top-20']
runs = {
    'scifact': ['20261012_011743_164d4865', '20261012_031344_b3d097b2', '20261012_031511_57b74dda', '20261011_123318_86ae3723', '20261011_125607_76912e7b'],
    'nfcorpus': ['20261012_011854_5304b15c', '20261012_031430_7c681d4e', '20261012_032010_b259b035', '20261011_141533_2f976292', '20261011_143638_8d39eaff'],
}
published = {'scifact': [.665, .643, .688], 'nfcorpus': [.325, .319, .350]}
output = ROOT / 'reports/paper_reproduction/full_baselines_20261012'
output.mkdir(exist_ok=True)
rows = []
for dataset, ids in runs.items():
    base = None
    for i, (system, run_id) in enumerate(zip(systems, ids)):
        folder = ROOT / 'runs' / run_id
        assert json.loads((folder / 'status.json').read_text())['status'] == 'VERIFIED'
        metrics = verify_run(folder)
        metadata = json.loads((folder / 'metadata.json').read_text())
        data = {k: metadata[k] for k in ['corpus_checksum', 'query_checksum', 'qrels_checksum']}
        if base is None:
            base, checksum = metrics, data
        assert data == checksum
        paper = published[dataset][i] if i < 3 else None
        if paper is not None:
            assert round(metrics['nDCG@10'], 3) == paper
        rows.append(dict(dataset=dataset, system=system, run_id=run_id, paper_ndcg10=paper,
                         ndcg10=metrics['nDCG@10'], recall100=metrics['Recall@100'], map100=metrics['MAP@100'],
                         delta_ndcg10=metrics['nDCG@10']-base['nDCG@10'],
                         relative_ndcg10_percent=(metrics['nDCG@10']/base['nDCG@10']-1)*100,
                         device=json.loads((folder/'hardware.json').read_text()).get('gpu') if (folder/'hardware.json').exists() else None))
with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines=['# Tái hiện ba baseline BEIR 2021 và so sánh Qwen', '',
       '| Dataset | System | Paper nDCG@10 | Run nDCG@10 | Delta BM25 | Relative | Recall@100 | MAP@100 | Run ID |',
       '|---|---|---:|---:|---:|---:|---:|---:|---|']
for row in rows:
    paper='—' if row['paper_ndcg10'] is None else f"{row['paper_ndcg10']:.3f}"
    lines.append(f"| {row['dataset']} | {row['system']} | {paper} | {row['ndcg10']:.6f} | {row['delta_ndcg10']:+.6f} | {row['relative_ndcg10_percent']:+.2f}% | {row['recall100']:.6f} | {row['map100']:.6f} | {row['run_id']} |")
lines += ['', 'Cả sáu baseline làm tròn nDCG@10 tới 3 chữ số khớp Table 2: https://arxiv.org/html/2104.08663v4 . Mỗi run đã đọc lại ranking, kiểm tra checksum/coverage và khớp cả năm metric với pytrec_eval.',
          '', 'TAS-B và MiniLM CE chạy trên A30/Ubuntu, Python 3.12.7/PyTorch 2.6.0+cu124, FP32/context 512, batch 1. Theo yêu cầu mới, không nghỉ cố định; vẫn tạm nghỉ từ 73 đến 60 C, dừng 80 C và bắt đầu model mới khi <=55 C. Chỉ chạy quality first pass; không có warm latency/QPS.',
          '', 'CE dùng candidate Anserini đã VERIFIED và giữ nguyên top-100 (tối đa 100 document/query; query lexical no-match giữ rỗng). Latency chỉ có cold reranker-only, không đo lại BM25. Qwen dùng lại run laptop BF16/top-20, có nghỉ cố định; không so latency giữa hai thiết bị/protocol.',
          '', 'Upstream commit 52de18ae6f1ccc7426eb70895c0e43e1552ad493 giữ nguyên. Wrapper nạp trực tiếp file SentenceBERT nguyên bản để tránh package __init__ kéo TensorFlow không liên quan; CE shim đổi keyword sentences sang positional inputs cho SentenceTransformers hiện tại. Model revisions và source checksums ở metadata của từng run. Không tuyên bố runtime byte-identical năm 2021.',
          '', 'Qwen embedding tốt hơn TAS-B trên cả hai dataset; Qwen+reranker có nDCG cao nhất trong các hệ thống đã đo. Recall@100 của Qwen reranker top-20 thấp hơn CE top-100 trên cả hai dataset; không coi candidate khác nhau là cùng protocol. Chưa đủ bằng chứng zero-shot nghiêm ngặt hay nhiều domain độc lập.',
          '', 'Phiên server hoàn chỉnh: reports/paper_neural_session_20261012_031344. Nhiệt tối đa 73 C, một lần tạm nghỉ tới 60 C, không chạm 80 C. 42 tệp artifact trước phiên giữ nguyên; sáu JSON lịch sử laptop vẫn giữ nguyên. Lượt lỗi TensorFlow và lượt đổi cấu hình không có metric hoàn chỉnh.',
          '', 'Tái tính bảng: `.\\.venv-paper\\Scripts\\python.exe compare_all_baselines.py`. Không thực hiện inference mới.']
(output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
fig, axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
for ax,dataset in zip(axes,runs):
    selected=[r for r in rows if r['dataset']==dataset]
    bars=ax.bar(['BM25','TAS-B','BM25+CE100','Qwen','Qwen+CE20'],[r['ndcg10'] for r in selected],color=['#68757f','#948a72','#5366a7','#2474b5','#13856c'])
    ax.bar_label(bars,labels=[f"{r['ndcg10']:.4f}" for r in selected],padding=3)
    ax.set_title(dataset); ax.set_ylim(0,1); ax.tick_params(axis='x',labelrotation=15)
axes[0].set_ylabel('nDCG@10 (linear gain)'); fig.tight_layout()
fig.savefig(output/'ndcg_comparison.png',dpi=180); plt.close(fig)
print(output)
