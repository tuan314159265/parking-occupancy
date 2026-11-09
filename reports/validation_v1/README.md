# Kết quả validation v1

Đã chạy đủ 1.182 ảnh train và 270 validation (15.120 ô); đây là số liệu dùng chọn tham số, không phải test độc lập.
YOLO11n cùng checkpoint ở tất cả nhánh. Tập test không được chạy lại trong task này.

| Phương pháp | F1 | Accuracy | Coverage | MAE ô trống | ms/ảnh CPU |
|---|---:|---:|---:|---:|---:|
| HOG + SVM | 0.9888 | 0.9415 | 0.9538 | 3.093 | 267.4 |
| YOLO full 640 | 0.5691 | 0.5855 | 0.8713 | 17.970 | 242.5 |
| Hybrid 640 | 0.7439 | 0.6999 | 0.8920 | 12.674 | 504.4 |
| Full pixel-matched | 0.7418 | 0.6978 | 0.8819 | 12.311 | 430.8 |

## Cấu hình đã khóa

- HOG/SVM: C=0.01, ngưỡng margin=-0.25.
- Hybrid: padding=16 px; overlap yêu cầu=0.2.
- Confidence hậu xử lý dùng chung=0.15; YOLO confidence floor cố định=0,15.
- Quy tắc ghép: `{"ambiguity_margin": 0.0, "lower_fraction": 0.67, "min_overlap": 0.2, "min_score": 0.15, "nms_iou": 0.5}`.
- Checkpoint SHA256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- Toàn bộ 9 cấu hình HOG và 144 tổ hợp joint có trong hog_search.csv và detector_search.csv, gồm cả cấu hình không cải thiện.

## Đối chiếu pixel và diễn giải

Hybrid so với full 640: ΔF1=+0.1748; so với toàn ảnh khớp pixel: ΔF1=+0.0020.
Pixel tensor của hybrid và đối chứng bằng nhau trên từng ảnh; padding/overlap hiệu dụng và fallback nằm trong tile_plans.json. Pixel bằng nhau không đồng nghĩa FLOPs/latency bằng nhau.
Không quy toàn bộ lợi ích so với full 640 cho ROI; cần đọc cùng kết quả full pixel-matched và thời gian. Ngân sách pixel không áp dụng cho HOG/SVM.
Timing là một lượt đo sau warm-up trên CPU local, có thể chịu tải của máy; không so trực tiếp với timing GPU Colab. Kết quả validation có thiên lệch chọn tham số.

## Khóa và lịch sử test

locked_config.json giữ checkpoint, split, model, tham chiếu train, engine và phiên bản; model HOG/geometry reference cụ thể nằm trong outputs/validation_v1/ và cần giữ để chạy test đúng khóa.
Notebook cũ đã lưu output test tại main 293893d. Task này không dùng điểm test để xếp hạng nhưng không gọi test v1 là holdout chưa quan sát; xem docs/validation.md về lịch sử.
Ảnh/feature/checkpoint/bundle lớn không được đưa vào bản báo cáo Git này. Bằng chứng chi tiết per-slot/per-frame ở outputs/validation_v1/selected/.
