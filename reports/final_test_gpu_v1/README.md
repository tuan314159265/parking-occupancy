# Lượt chạy lại trên Colab GPU (kiểm tra tái lập)

Chạy lại gói khóa `v1.0-final` (hash khóa validation `931ebc8a…972a`) trên Google Colab, `device=0`, PyTorch 2.15.0+cu132. Không chọn lại tham số (`parameter_selection=false`).

**Kết quả:** `aligned_predictions.jsonl` của cả 4 phương pháp trùng byte-for-byte với lượt CPU trong [`../final_test_v1`](../final_test_v1), nên F1/precision/recall/accuracy giống hệt. Chỉ thời gian khác (thiết bị khác nhau).

| Phương pháp | F1 | ms/ảnh CPU local | ms/ảnh Colab |
|---|---:|---:|---:|
| HOG + SVM | 0.9654 | 167.8 | 347.9 |
| YOLO full 640 | 0.3679 | 165.3 | 170.1 |
| Hybrid | 0.6517 | 386.5 | 210.5 |
| YOLO full pixel-matched | 0.6249 | 377.9 | 185.3 |

HOG không dùng GPU nên số của nó phản ánh CPU của runtime Colab. Thư mục này chỉ giữ bảng, metrics, timing và hash; dự đoán từng ô và ảnh minh họa giống lượt CPU nên không lặp lại (`evidence_sha256.json` ghi hash của bộ đầy đủ trên Drive).
