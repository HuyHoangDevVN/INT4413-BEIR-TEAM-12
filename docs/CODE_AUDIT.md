# Audit mã nguồn V1 và kế hoạch sửa V2

Ngày kiểm tra: 2026-10-07. Chủ sở hữu logic: các script Python tại project này; không có Git repository. Đã đọc README, requirements, utils, run_bm25, run_dense, run_rerank, collect_results, plot và cả 10 cell notebook.

## Luồng hiện tại
Dataset BEIR JSONL (hoặc GZIP), qrels TSV test → lọc query có qrels → BM25 hoặc normalized MiniLM cosine → tùy chọn CrossEncoder trên top-100 → evaluator → JSON tổng hợp. SciFact có 5.183 documents; NFCorpus có 3.633 documents. Dữ liệu local đầy đủ. Sáu JSON lịch sử chỉ lưu hai metric, không lưu ranking/environment nên chưa xác minh lại được.

## Correctness
- utils.dcg_at_k dùng gain `2**rel-1`; khác gain tuyến tính của TREC ndcg_cut. NFCorpus có relevance 2 nên sự khác biệt có ý nghĩa.
- recall_at_k dùng membership và len(gold), đếm cả relevance 0 nếu dataset có; duplicate ranking có thể đếm nhiều lần.
- evaluator lặp trên results nên query thiếu ranking không được tính vào mẫu số.
- collect_results và notebook dùng model.split('+')[0], gán BM25 paper cho reranker; số tham chiếu cũ phải kiểm tra lại với paper trước khi sử dụng.
- loader bỏ qua TSV lỗi và overwrite ID trùng; thiếu validation.
- BM25 script tie ổn định theo corpus; notebook dùng argsort với tie không đảm bảo giống script.

## Reproducibility và hiệu năng
- Không lưu ranking, model revision, checksum, seed, hardware hoặc timing; kết quả bị overwrite theo tên model.
- BM25 get_scores quét corpus và sort toàn bộ cho mỗi query; giữ làm reference trước khi thêm postings.
- Dense encode corpus mỗi run và giữ ma trận queries × documents; không có cache hoặc chunked search.
- Reranker batch trong mỗi query, đúng chỉ score candidates nhưng không log chi phí.
- Notebook duplicate logic nên dễ drift.
- requirements không pin; Python mặc định 3.8 32-bit, Python 3.12 có torch CPU và ST 2.7.0. Có RTX 3070 Laptop nhưng torch.cuda.is_available=False; không suy ra neural GPU đã khả dụng.
- utils import thực hiện network probe, SSL_VERIFY=False và patch SSL toàn cục; cần bỏ, dùng TLS verified và CA bundle.

## Bảo toàn và thay đổi
Giữ sáu JSON results nguyên trạng, baseline model/tokenizer/formula BM25, CLI cũ và bản source trong legacy/v1. V2 lưu runs riêng và gain mũ dưới tên legacy_exponential_ndcg@10. Xây evaluator/tests trước optimization; bổ sung adapters/cache/index/CLI, benchmark cold/warm và report có trạng thái. Source, fixture tests và model thực chạy là các bằng chứng khác nhau. Không có kết quả mới tại thời điểm viết audit này.
