# Tái hiện baseline BEIR 2021

## Cập nhật 12/10/2026: baseline neural hoàn tất

TAS-B dot product và Anserini → MiniLM-L6-v2 top-100 đã VERIFIED trên SciFact/NFCorpus ở server A30. nDCG làm tròn khớp Table 2: TAS-B 0.643/0.319, CE 0.688/0.350. Phần mô tả "chưa chạy" dưới đây là trạng thái lượt BM25 trước đó. Xem reports/paper_reproduction/full_baselines_20261012/comparison.md.

Server Python 3.12.7/PyTorch 2.6.0+cu124, FP32/context 512/batch 1. Bỏ nghỉ cố định theo yêu cầu mới; vẫn giám sát 73→60°C, dừng 80°C. Cấu hình laptop/Qwen giữ nguyên. Wrapper load file SentenceBERT nguyên trạng bằng importlib vì __init__ upstream kéo TensorFlow không dùng; CE shim chuyển keyword sentences sang positional inputs. Không sửa source upstream, không coi shim là bằng chứng mock: bốn model runs thực encode/predict và đọc lại ranking khớp pytrec_eval.

Paper: https://arxiv.org/html/2104.08663v4, Section 4 và Table 6.

- Baseline chính BM25 là Anserini, k1=0.9, b=0.4; title/contents riêng, mỗi field weight=1.0. Phát biểu trước rằng paper dùng Elasticsearch là sai; Elasticsearch là một implementation khác của upstream.
- BM25+CE dùng top-100, checkpoint cross-encoder/ms-marco-MiniLM-L-6-v2, context 512. MiniLM dense V1 không phải baseline dense được nêu trong paper; TAS-B là một checkpoint dense của paper. Lượt này tái hiện baseline BM25 trên hai dataset, chưa chạy TAS-B hay BM25+CE của paper.
- Upstream checkout nguyên trạng tại upstream/beir, commit 52de18ae6f1ccc7426eb70895c0e43e1552ad493 (2021-09-22), commit cuối trước ngày v4 (2021-10-21). Đây là lựa chọn paper-era có thể tái lập, không có bằng chứng đây là SHA tác giả dùng để tạo bảng paper.
- examples/beir-pyserini/Dockerfile pin pyserini==0.12.0 và Java 11. main.py dùng SimpleSearcher, k1=0.9/b=0.4, contents/title=1.0. Image tác giả đã resolve digest sha256:c18cb7f60db4390978d3260bb9e80c797bf11e07e102a407323c7c5c804c83bf; cần kiểm tra source/version bên trong trước khi chạy.
- Môi trường wrapper Python 3.12 riêng .venv-paper dùng cùng third-party packages của .venv hiện có, không thay package toàn máy hay sửa checkout upstream. Không tuyên bố tái hiện byte-identical runtime 2021.
- Wrapper chỉ thêm offline data, artifact và điều tiết CPU/RAM; giữ index/search Pyserini và evaluator upstream. Không chạy neural trong lượt BM25 này. Cấu hình Qwen cũ giữ nguyên: GPU batch=1, BF16, context=512, nghỉ 0.1 giây/batch, tạm nghỉ 73 đến 60 C, dừng 80 C; nghỉ đầu 60 giây/giữa model 120 giây, bắt đầu <=55 C. Timing có nghỉ chỉ dùng so sánh cùng protocol.
- Dữ liệu/kết quả cũ giữ nguyên. Qwen hiện có BF16/top-20 là experiment mở rộng; so sánh quality cùng qrels, không coi candidate/precision/latency là cùng cấu hình với paper CE top-100.
- Image thực tế đã kiểm tra: pyserini 0.12.0, main.py/config.py khớp source checkout khi chuẩn hóa newline. Container chạy 2 CPU/3 GiB RAM. NFCorpus tìm cả tập một lần vượt Java heap mặc định; retry với heap 1536 MiB và nhóm 20 query thành công, không đổi ranking/thuật toán.
- NFCorpus có 15 query lexical no-match; upstream pytrec_eval vẫn tính đủ 323 query. query_coverage.json lưu số dòng kể cả 0, checksum được xác minh; không chèn document giả hoặc loại query. Run lỗi giữ trạng thái riêng và không đưa metric vào báo cáo hoàn chỉnh.
