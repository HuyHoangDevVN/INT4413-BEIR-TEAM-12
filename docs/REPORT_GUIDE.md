# Hướng dẫn báo cáo Word

1. **Bối cảnh:** nhu cầu tìm thông tin đúng domain, lexical gap và chi phí neural. Nêu BEIR là benchmark, dẫn paper gốc.
2. **Vấn đề nghiên cứu:** trình bày RQ1–RQ4; định nghĩa “cải thiện” bằng delta metric và “chi phí” bằng protocol cụ thể.
3. **BEIR benchmark:** mô tả corpus/query/qrels, split test, SciFact/NFCorpus; không coi hai dataset là toàn benchmark.
4. **Mô hình:** BM25, MiniLM bi-encoder/CE, BGE-M3, Qwen3, hybrid RRF và reranker; dẫn model card đúng checkpoint.
5. **Thiết kế thực nghiệm:** trích config, hardware, phiên bản, revision, checksums, max_length, candidate-k; giải thích linear gain và cold/warm.
6. **Kết quả:** dùng `reports/results_summary.csv` và run ID. Giữ Paper result và Our run thành hai cột; không lấy JSON lịch sử thành VERIFIED. Đưa Recall@10, MRR/MAP vào bảng bổ sung.
7. **Phân tích:** trả lời từng RQ bằng dataset cụ thể; nếu chỉ BM25 đã chạy thì RQ so modern chưa trả lời được. Báo cáo cả giảm và tăng, không cherry-pick.
8. **Quality-cost trade-off:** dùng scatter nDCG-latency và nDCG-memory; candidate 20/50/100 so với nguồn candidate cùng K. Chỉ chia latency khi protocol/hardware tương thích.
9. **Hạn chế:** hai domain, qrels incomplete, tokenizer, truncation, model training contamination, CPU/GPU availability, RSS sampling, serial QPS, model chưa chạy.
10. **Kết luận:** chỉ kết luận trong phạm vi measurements; đề xuất experiment tiếp theo sau khi xem số liệu. Không viết “modern models solved BEIR”.

Nguồn dữ liệu cho bảng/hình là run artifacts; mô tả thuật ngữ xem EXPERIMENT_DESIGN.md. Chèn hình PNG từ plots ở độ phân giải gốc, đặt caption kèm dataset, run ID, metric gain và warm/cold. Không viết báo cáo trước khi evaluator/tests ổn định.

Mẫu câu: “Trong experiment [run ID], hệ thống X đạt nDCG@10 cao hơn Y trên SciFact; kết quả này chưa chứng minh ưu thế trên các domain khác.” Khi chưa chạy: “Integration đã có; experiment chưa thực hiện vì [reason].”

Tài liệu tham khảo: [BEIR 2021](https://arxiv.org/abs/2104.08663), [BEIR evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py) và model cards liên kết trong EXPERIMENT_DESIGN.md. Các DOCX hiện có không được tự động sửa trong lần nâng cấp này.
