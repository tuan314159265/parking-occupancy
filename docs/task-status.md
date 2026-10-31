# Trạng thái bàn giao — 28/10/2026

1. Khung repo local: đã tạo data/configs/src/outputs, notebook Run All, hướng dẫn môi trường, khung LaTeX, contracts dữ liệu/hình học/detector/đánh giá. Đã cài editable trong .venv và chạy 8 kiểm thử thành công. Đã kiểm tra cú pháp các cell notebook; chưa chạy trọn notebook trên Colab và chưa biên dịch LaTeX vì máy chưa có pdflatex. Mã được bàn giao qua repository GitHub tuan314159265/parking-occupancy, nhánh main.
2. PKLot: đã xác nhận URL chính thức, kích thước archive và XML thật. Loader/audit và pipeline tải tiếp → giải nén full-frame → audit đã triển khai. Archive còn đang tải chậm; chưa có danh sách mẫu hợp lệ của bộ đầy đủ. CNRPark-EXT chưa tải.

Phiên này có downloader đang chạy và tiến trình chờ archive đủ kích thước để tự giải nén/audit (tối đa chờ 24 giờ). Xem outputs/pipeline_status.json. Khi hoàn tất, xem outputs/full/valid_samples.jsonl, issues.jsonl, audit_summary.json và provenance.json. Nếu phiên bị dừng hoặc trạng thái không tiến triển, dùng lệnh prepare trong README để tải tiếp, sau khi chắc rằng downloader cũ đã dừng.

Kiểm tra tạm outputs/partial chỉ chứa các file có trong đầu archive chưa hoàn tất. Chưa ghép được cặp ảnh/XML tại thời điểm kiểm tra; lỗi thiếu đối ứng trong tập tạm không chứng minh ảnh/XML bị thiếu ở bộ gốc. Không sử dụng thống kê tạm để báo cáo chất lượng toàn bộ PKLot.

## Cập nhật luồng GitHub → Colab

Mã xử lý được gom vào parking_module.py tại gốc repository. Notebook chỉ cần OWNER/REPO/REF, tải file từ GitHub và import trực tiếp. Không yêu cầu ZIP repo hay cài package trên Colab. Các module src cũ chỉ chuyển tiếp đến module gốc để tương thích CLI local. Repo đích: https://github.com/tuan314159265/parking-occupancy; notebook đã cấu hình sẵn.

## 31/10/2026 — task 3

Đã rà dữ liệu, cập nhật loader v2, audit lại, khóa configs/pklot_v1.json và xuất train/val/test cùng JSON layout. Thí nghiệm chính 1.182/270/270 frame; chuyển UFPR04 → UFPR05 có 420/90/90. Notebook 3.2 tải manifest dùng chung từ GitHub. Xem docs/dataset-protocol.md để biết thống kê và kiểm chứng. Phần split dữ liệu đã có; các phương pháp và đánh giá thực nghiệm vẫn là bước tiếp theo.
