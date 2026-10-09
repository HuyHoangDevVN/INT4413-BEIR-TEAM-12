# Kết quả baseline và Qwen đã kiểm chứng

Snapshot: 2026-10-12T16:02:31.291779+00:00. Không đại diện tiến độ live của server.

Metric dưới đây được tính lại từ ranking đã lưu và đối chiếu pytrec_eval trong gói bằng chứng local. Ranking/qrels đầy đủ không nằm trong Git; `provenance.json` ghi nguồn và checksum của các bảng. SHA-256 chứng minh tính toàn vẹn, không tự chứng minh thí nghiệm đúng.

| Dataset | Pipeline | Run ID | nDCG@10 | Recall@100 | MRR@10 | MAP@100 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| scifact | Anserini BM25 | 20261012_011743_164d4865 | 0.664705 | 0.907556 | 0.630220 | 0.621740 |
| scifact | TAS-B | 20261012_031344_b3d097b2 | 0.642755 | 0.891000 | 0.612123 | 0.604591 |
| scifact | Anserini BM25 → MiniLM CE top-100 | 20261012_031511_57b74dda | 0.687546 | 0.907556 | 0.657901 | 0.648352 |
| scifact | Qwen embedding | 20261011_123318_86ae3723 | 0.696704 | 0.953333 | 0.664597 | 0.660031 |
| scifact | Qwen embedding → Qwen reranker top-20 | 20261011_125607_76912e7b | 0.751778 | 0.876667 | 0.727794 | 0.716871 |
| scifact | Qwen embedding → Qwen reranker top-50 | 20261012_034346_41582e11 | 0.762875 | 0.926667 | 0.736164 | 0.729349 |
| scifact | Qwen embedding → Qwen reranker top-100 | 20261012_042342_4498124d | 0.763249 | 0.953333 | 0.735913 | 0.729164 |
| nfcorpus | Anserini BM25 | 20261012_011854_5304b15c | 0.325450 | 0.250047 | 0.532516 | 0.146694 |
| nfcorpus | TAS-B | 20261012_031430_7c681d4e | 0.318865 | 0.280119 | 0.520582 | 0.145868 |
| nfcorpus | Anserini BM25 → MiniLM CE top-100 | 20261012_032010_b259b035 | 0.350424 | 0.250047 | 0.581858 | 0.160925 |
| nfcorpus | Qwen embedding | 20261011_141533_2f976292 | 0.356216 | 0.327960 | 0.562534 | 0.170417 |
| nfcorpus | Qwen embedding → Qwen reranker top-20 | 20261011_143638_8d39eaff | 0.380999 | 0.212877 | 0.594987 | 0.161476 |
| nfcorpus | Qwen embedding → Qwen reranker top-50 | 20261012_040125_fd25658a | 0.387883 | 0.274494 | 0.601267 | 0.179711 |
| nfcorpus | Qwen embedding → Qwen reranker top-100 | 20261012_050442_11865b28 | 0.390546 | 0.327960 | 0.604603 | 0.188403 |

## Điều kiện diễn giải

- `quality.csv` giữ mọi hồ sơ, kể cả partial/failed/summary-only; không lấy điểm test cao nhất làm cấu hình chính.
- `comparisons.csv` chứa delta hợp lệ và run ID đối chứng; điểm paper nằm riêng trong `references/`.
- `statistical_summary.csv` là kiểm định đã lưu: nDCG Holm36, metric bổ sung Holm108 và sensitivity Holm144.
- Top-50 SciFact/top-100 NFCorpus là lựa chọn post-hoc, không phải tuning trên dev độc lập.
- Top-20/50 có Recall@100 trên output đã cắt; reranker100 giữ Recall@100 của cùng 100 candidates.
- Laptop/A30/CPU và scope timing khác nhau: không kết luận speedup trực tiếp từ `efficiency.csv`.
- Hai dataset có đủ matrix baseline/Qwen. FiQA/ArguAna/TREC-COVID trong snapshot chỉ có một phần hệ thống.
- Không chứng minh strict zero-shot hoặc cải thiện trên toàn BEIR.
