# Trạng thái bàn giao — 28/10/2026

1. Khung repo local: đã tạo data/configs/src/outputs, notebook Run All, hướng dẫn môi trường, khung LaTeX, contracts dữ liệu/hình học/detector/đánh giá. Đã cài editable trong .venv và chạy 8 kiểm thử thành công. Đã kiểm tra cú pháp các cell notebook; chưa chạy trọn notebook trên Colab và chưa biên dịch LaTeX vì máy chưa có pdflatex. Mã được bàn giao qua repository GitHub tuan314159265/parking-occupancy, nhánh main.
2. PKLot: đã xác nhận URL chính thức, kích thước archive và XML thật. Loader/audit và pipeline tải tiếp → giải nén full-frame → audit đã triển khai. Archive còn đang tải chậm; chưa có danh sách mẫu hợp lệ của bộ đầy đủ. CNRPark-EXT chưa tải.

Phiên này có downloader đang chạy và tiến trình chờ archive đủ kích thước để tự giải nén/audit (tối đa chờ 24 giờ). Xem outputs/pipeline_status.json. Khi hoàn tất, xem outputs/full/valid_samples.jsonl, issues.jsonl, audit_summary.json và provenance.json. Nếu phiên bị dừng hoặc trạng thái không tiến triển, dùng lệnh prepare trong README để tải tiếp, sau khi chắc rằng downloader cũ đã dừng.

Kiểm tra tạm outputs/partial chỉ chứa các file có trong đầu archive chưa hoàn tất. Chưa ghép được cặp ảnh/XML tại thời điểm kiểm tra; lỗi thiếu đối ứng trong tập tạm không chứng minh ảnh/XML bị thiếu ở bộ gốc. Không sử dụng thống kê tạm để báo cáo chất lượng toàn bộ PKLot.

## Cập nhật luồng GitHub → Colab

Mã xử lý được gom vào parking_module.py tại gốc repository. Notebook chỉ cần OWNER/REPO/REF, tải file từ GitHub và import trực tiếp. Không yêu cầu ZIP repo hay cài package trên Colab. Các module src cũ chỉ chuyển tiếp đến module gốc để tương thích CLI local. Repo đích: https://github.com/tuan314159265/parking-occupancy; notebook đã cấu hình sẵn.

## 31/10/2026 — task 3

Đã rà dữ liệu, cập nhật loader v2, audit lại, khóa configs/pklot_v1.json và xuất train/val/test cùng JSON layout. Thí nghiệm chính 1.182/270/270 frame; chuyển UFPR04 → UFPR05 có 420/90/90. Notebook 3.2 tải manifest dùng chung từ GitHub. Xem docs/dataset-protocol.md để biết thống kê và kiểm chứng. Phần split dữ liệu đã có; các phương pháp và đánh giá thực nghiệm vẫn là bước tiếp theo.

## 04/11/2026 — task 11

Đã hiện thực evaluator chung, ghép khóa ảnh/ô, xử lý unknown/missing, confusion matrix, precision/recall/F1, accuracy tổng, coverage, tỉ lệ báo trống sai, MAE đếm, timing theo ảnh gốc và xuất JSON/CSV. Có phân nhóm camera/thời tiết, FrameTimer cho nhiều tile, CLI và notebook 6.1. Công thức được kiểm chứng bằng ví dụ nhỏ; đây là hạ tầng đánh giá, chưa phải kết quả thực nghiệm của task 12/13.

## 04/11/2026 — task 15

Đã viết mục tiêu, input/output, phạm vi camera cố định, phân biệt phát hiện xe và trạng thái ô; mô tả PKLot, nhãn/polygon, kiểm tra chất lượng, tập con và giao thức theo ngày/camera. Nội dung ở docs/introduction-and-data.md, report/sections/introduction.tex và data.tex; đã tích hợp vào notebook mục 1 và 3. Đây là phần mở đầu/dữ liệu, các mục phương pháp/kết quả của báo cáo vẫn chờ hoàn thiện.

## 08/11/2026 — tích hợp pipeline và trực quan hóa

Đã review và merge nhánh feat/parking-line-geometry (4.2, 4.3, 4.6, 6.2 của Văn). Đã tăng tốc analyze_parking_geometry khoảng 23 lần với đầu ra không đổi, và giữ LF cho manifest trên Windows. Đã thêm cờ hình học yếu so với tham chiếu train (build_geometry_reference, assess_geometry_quality, apply_geometry_policy) cùng 11 test. Đã hoàn thiện notebook mục 5 (pipeline, hình 6 panel, CSV/JSON), 6.2b, 6.3, 7–9. Toàn bộ notebook đã chạy trên 141 ảnh PKLot thật lấy từ đầu archive, nhưng nhãn không khớp ảnh nên chưa có số liệu thực nghiệm; cần Run All trên Colab với dataset đầy đủ. Xem docs/pipeline-integration.md.


## 09/11/2026 — task 12 (within_camera)

Đã fetch/review main 293893d, giữ bản sửa local riêng, chạy validation thật trên đủ 1.182 train / 270 validation. Đã thử 9 cấu hình HOG và 144 tổ hợp detector; 66 tổ hợp không tăng mean F1 so với mặc định, vẫn được lưu. Thêm YOLO toàn ảnh khớp số pixel tensor thật với tiling; benchmark lại bốn phương pháp end-to-end. Đã khóa checkpoint, split, model, tham chiếu hình học, phiên bản và quy tắc trước lượt test mới. Kết quả và toàn bộ grid nhẹ ở reports/validation_v1; cache/model/chi tiết ở outputs/validation_v1. 113 kiểm thử đạt. Không chạy lại test trong task này; notebook cũ đã có output test nên không gọi test v1 là holdout chưa quan sát. Giao thức chuyển camera chưa chạy validation mới.

## 11/11/2026 — hợp nhất validation và test vào notebook chính

Notebook 01 đã ghép kiểm tra khóa, runner test, bảng tổng/theo nhóm, ảnh đúng/sai/unknown và checksum vào mục 6.3; mục 7 hiển thị lại bằng chứng. Mặc định RUN_VALIDATION_SEARCH=False để dùng artifact đã có, RUN_FINAL_TEST=False để chủ động chạy test. Ưu tiên code/split/plan local hoặc bundle; tự tìm đúng outputs/validation_v1 của lượt local trước. ZIP khôi phục chứa notebook chính và đầy đủ split metadata, không chứa dataset. Đã chạy toàn bộ 33 code cell theo thứ tự với mạng bị chặn và hai cờ suy luận tắt, xác nhận đủ 270 ảnh đầu vào. Bộ 131 kiểm thử đạt tại lần kiểm tra tích hợp. Các nhận xét số liệu test cũ được giữ trong khối lịch sử, không nhận là số liệu khóa mới. Chưa thực hiện inference test mới.
