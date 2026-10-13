# Hướng dẫn 14 slide

Mỗi slide giữ tối đa khoảng 6 dòng; notes chứa phần diễn giải. Các suggested figure chỉ sử dụng khi có measured data.

## Slide 1
- Title: BEIR: từ 2021 đến retrieval hiện đại
- Key message: Đánh giá chất lượng và chi phí cùng nhau
- Content: Chủ đề, tên nhóm, RQ1–RQ4
- Suggested figure: Sơ đồ pipeline
- Speaker notes: Giới thiệu phạm vi project, không hứa kết quả chưa đo.
- Source: EXPERIMENT_DESIGN.md

## Slide 2
- Title: Nhu cầu tìm kiếm ngoài domain
- Key message: Điểm cao in-domain chưa đảm bảo tổng quát hóa
- Content: Lexical gap; domain shift; training vs test
- Suggested figure: Ví dụ query/document
- Speaker notes: Dùng một ví dụ khoa học và một ví dụ y sinh.
- Source: https://arxiv.org/abs/2104.08663

## Slide 3
- Title: BEIR là benchmark
- Key message: Nhiều tác vụ và domain dùng cùng khung đánh giá
- Content: Corpus, queries, qrels; SciFact/NFCorpus
- Suggested figure: Bảng dataset
- Speaker notes: Hai dataset đang chạy chỉ là một phần BEIR.
- Source: docs/CODE_AUDIT.md

## Slide 4
- Title: Zero-shot và giới hạn chứng minh
- Key message: Không fine-tune đích không chứng minh không contamination
- Content: Checkpoint pretrained; revision; training overlap
- Suggested figure: Sơ đồ training/evaluation
- Speaker notes: Nêu rõ model hiện đại có thể đã thấy dữ liệu benchmark.
- Source: docs/EXPERIMENT_DESIGN.md

## Slide 5
- Title: BM25 lexical baseline
- Key message: Baseline mạnh và có thể kiểm tra công thức
- Content: TF, IDF, length normalization; k1/b
- Suggested figure: Sơ đồ postings
- Speaker notes: Tokenizer và tham số khác paper có thể tạo chênh lệch.
- Source: lab/retrieval.py

## Slide 6
- Title: Dense retrieval
- Key message: Bi-encoder cho phép cache corpus vectors
- Content: MiniLM, BGE-M3, Qwen3; cosine
- Suggested figure: Query/document embeddings
- Speaker notes: Query prompt Qwen khác document; không encode corpus mỗi run.
- Source: lab/models.py

## Slide 7
- Title: Hybrid RRF
- Key message: Kết hợp bằng rank tránh raw-score mismatch
- Content: BM25 + dense; k=60; rank từ 1
- Suggested figure: Sơ đồ hai nhánh
- Speaker notes: RRF không học trọng số theo test qrels.
- Source: lab/retrieval.py

## Slide 8
- Title: Reranking
- Key message: Chất lượng đổi lấy inference trên candidates
- Content: MiniLM CE, BGE, Qwen; K=20/50/100
- Suggested figure: Pipeline top-K
- Speaker notes: Giữ candidate set thì Recall@K không đổi.
- Source: lab/models.py

## Slide 9
- Title: Evaluator đúng
- Key message: Định nghĩa metric cần cố định trước benchmark
- Content: Linear nDCG; Recall; MRR; MAP; missing queries
- Suggested figure: Ví dụ graded relevance
- Speaker notes: NFCorpus có relevance 2 nên gain mũ và tuyến tính khác nhau.
- Source: evaluation/metrics.py

## Slide 10
- Title: Thiết kế và hardware
- Key message: Cùng protocol mới so quality-cost được
- Content: Revision, checksum, config, device; cold/warm
- Suggested figure: Run artifact tree
- Speaker notes: Báo CPU/GPU thực dùng thay vì chỉ nói máy có GPU.
- Source: runs/*/hardware.json

## Slide 11
- Title: Kết quả thực đo
- Key message: Chỉ dùng run thành công có evidence
- Content: nDCG@10/Recall@100; trạng thái
- Suggested figure: plots/ndcg_by_model.png
- Speaker notes: Đọc run ID và phạm vi dataset; không lấy lịch sử làm kết quả mới.
- Source: reports/results_summary.csv

## Slide 12
- Title: Chi phí và candidate depth
- Key message: Đo latency cùng chất lượng
- Content: P50/P95, QPS, RAM/VRAM; K sweep
- Suggested figure: plots/ndcg_vs_latency.png
- Speaker notes: Warm không chứa corpus encoding; cold log riêng.
- Source: runs/*/timing.json

## Slide 13
- Title: Trả lời RQ và hạn chế
- Key message: Domain-specific evidence quyết định kết luận
- Content: Delta vs BM25; hybrid; rerank; chưa chạy
- Suggested figure: plots/dataset_heatmap.png
- Speaker notes: Nếu modern chưa chạy, RQ1–RQ4 còn thiếu bằng chứng.
- Source: reports/results_summary.md

## Slide 14
- Title: Kết luận và bước tiếp
- Key message: Tiếp tục theo evidence đã có
- Content: Phát hiện, hạn chế, experiment kế tiếp
- Suggested figure: Bảng quality-cost
- Speaker notes: Nêu lý do chọn dataset hoặc model tiếp theo, không overclaim.
- Source: reports/IMPLEMENTATION_REPORT.md
