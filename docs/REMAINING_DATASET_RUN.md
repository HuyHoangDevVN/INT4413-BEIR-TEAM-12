# Chạy ba dataset còn lại — 12/10/2026

Phạm vi: FiQA, ArguAna, TREC-COVID; mỗi dataset gồm Anserini BM25, TAS-B, BM25→MiniLM CE top-100, Qwen embedding và Qwen reranker top-20/50/100. Đây là 21 cấu hình bổ sung của kế hoạch 5 dataset, không phải toàn bộ 18 tập BEIR. Chỉ nhận kết quả VERIFIED có ranking, checksum và đối chiếu pytrec_eval.

Supervisor `run_remaining_datasets.py` chạy tuần tự, không nghỉ cố định; GPU chỉ bắt đầu khi ≤55°C, tạm dừng 73→60°C, dừng cứng 80°C. Budget tối đa 24 giờ cho phiên, 8 giờ mỗi worker; run bị dừng không có metric hoàn chỉnh. `--resume reports/remaining_datasets_SESSION` tạo phiên tiếp tục mới, chỉ tái sử dụng run VERIFIED sau kiểm tra checksum dữ liệu/ranking; các attempt cũ được giữ lại. Source upstream và revision model vẫn pin như các lượt SciFact/NFCorpus. Qwen embedding dùng exact NumPy theo chunk, cache corpus; không chạy lại embedding cho từng candidate-k.

Dataset tải từ URL BEIR chính thức bằng HTTPS có xác minh TLS. Mọi member ZIP phải nằm trong thư mục dataset được yêu cầu; không được ghi sang dữ liệu offline khác. Dữ liệu hiện có:

| Dataset | Corpus | Query test |
|---|---:|---:|
| FiQA | 57.638 | 648 |
| ArguAna | 8.674 | 1.406 |
| TREC-COVID | 171.332 | 50 |

ArguAna có 5 qrel document không tồn tại trong corpus chính thức đã tải. BEIR xác nhận một trường hợp tại [issue #172](https://github.com/beir-cellar/beir/issues/172); năm ID thực tế được audit và lưu trong `lab.data.ARGUANA_MISSING`/metadata. Giữ nguyên tất cả 1.406 query/qrels và denominator; không tạo document hay bỏ query. Chỉ năm ID được audit được chấp nhận, missing ID khác vẫn bị từ chối. Query có relevant document không thể truy xuất nhận metric 0. ArguAna còn có query/document cùng ID: lọc self trước chọn candidate; lưu raw score thật và explicit rank.

TREC-COVID có hai nhãn -1 (`38/9hbib8b3`, `50/svo94kuo`), tương ứng unjudged theo [trec_eval formats](https://github.com/usnistgov/trec_eval/blob/main/formats.c). Giữ raw qrel -1 để khớp upstream; evaluator dùng gain 0/relevance >0 và đã đối chiếu fixture với pytrec_eval. Nhãn dưới -1 hoặc duplicate vẫn bị từ chối. Những trường hợp này được ghi metadata, không sửa JSONL/TSV gốc.

TREC-COVID làm container benchmark giới hạn 3 GiB OOM (exit 137/OOMKilled=true) khi lập index. Chỉ container `beir-paper-20211021` pin đúng image được nâng giới hạn RAM lên 8 GiB; Java heap vẫn 1536 MiB, CPU vẫn 2, k1=0.9/b=0.4 và title/contents weight 1.0. Giới hạn RAM thực tế được đọc từ Docker và lưu config/metadata mỗi run; metric của các lượt 3 GiB trước không bị sửa. Không restart hoặc sửa container dịch vụ khác.

Các lượt setup lỗi và run OOM không có metric hoàn chỉnh; được giữ trong session/attempt riêng. Không tự nâng trạng thái của kết quả chưa chạy. Kết quả mới phải được đồng bộ và xác minh lại trước khi đưa vào bảng 5 dataset. Không suy đoán winner hoặc số paper cho các dataset chưa hoàn tất.

BM25 FiQA/TREC-COVID đo 0.236067/0.655854, làm tròn khớp Table 2 BEIR 2021. BM25 ArguAna đo 0.414152, không khớp số 0.315 trong Table 2 năm 2021; nhưng khớp 0.414 nDCG và 0.943 Recall@100 của BM25 multifield trong [Pyserini reproduction](https://castorini.github.io/pyserini/2cr/beir.html). Không sửa số đo hoặc gọi đây là tái hiện khớp bảng 2021 trên ArguAna. Giữ nguồn năm 2021 và số reproduction riêng; so sánh Qwen với baseline thực chạy cùng dữ liệu/protocol.

GPU dùng chung lên 80°C trong lượt Qwen embedding FiQA; supervisor dừng riêng process benchmark, giữ `NOT_RUN_RESOURCE_LIMIT`, không có metric hoàn chỉnh. Sau khi benchmark dừng GPU vẫn 82–88°C và có workload của tài khoản khác, nên không gọi đây là nhiệt do riêng benchmark. Các baseline FP32 còn lại được chạy CPU rõ ràng (`--device cpu --cpu-threads 8`, batch 1, CUDA_VISIBLE_DEVICES rỗng trong riêng child process). Qwen vẫn chờ điều kiện GPU; không tự bỏ ngưỡng nhiệt. GPU queue tái sử dụng baseline CPU VERIFIED sau kiểm tra revision và checksum; không tạo kết quả giả và không silent fallback. Historical integrity chỉ chụp finalized runs/legacy results để hai phiên CPU/GPU không coi file đang chạy là artifact bất biến.
