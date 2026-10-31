# Giao diện phiên bản 0.2

`Frame`: sample_id là đường dẫn XML tương đối bỏ suffix; đường dẫn ảnh/XML tương đối dataset_root. Camera giữ nguyên tên thư mục (gồm PUC, PUCPR, UFPR04, UFPR05); date ISO theo timestamp tên file khi hợp lệ, directory_date giữ ngày thư mục và date_source ghi nguồn; Cloudy chuẩn hóa thành overcast. Có thể khôi phục metadata gốc qua xml_path.

`ParkingSpace`: space_id duy nhất trong frame, occupied boolean (XML 1=có xe, 0=trống), polygon theo thứ tự contour trong XML. Tọa độ pixel trên ảnh gốc: gốc trái trên, x sang phải, y xuống dưới; biên hình học cho phép x=width/y=height. Không thay contour bằng rotatedRect hoặc hộp xe. Vùng không hợp lệ làm cả frame bị loại và ghi lý do; không bỏ nhãn lỗi âm thầm.

`VehicleDetector.predict(image_path) -> list[VehicleDetection]`: xyxy pixel ảnh gốc, confidence [0,1], class_name. Sau resize/letterbox phải chuyển về tọa độ gốc. Adapter chịu trách nhiệm kiểm tra đầu ra. Chưa chọn hay cài detector; cấu hình detector=null là chủ ý. Bước ghép detection với ô dùng polygon và xuất OccupancyPrediction, chưa triển khai trong hai đầu việc này.

`OccupancyPrediction`: khóa (sample_id, space_id), occupied_score [0,1]. Bước đánh giá sau này phải join đúng khóa, phát hiện thiếu/trùng và chốt ngưỡng trên validation trước khi chạy test. Hàm occupancy_metrics nhận hai dãy binary đã căn chỉnh.

Đánh giá trạng thái ô: occupied là lớp dương, confusion matrix TP/TN/FP/FN, accuracy, precision, recall, F1 (mẫu số 0 trả 0). Báo cáo tổng và theo camera/thời tiết; không tính vehicle mAP từ XML PKLot. Cần nhãn hộp xe độc lập nếu đánh giá detector.

Split chính thức của dự án: dùng manifest đã khóa ở data/manifests/pklot_v1/ và cấu hình configs/pklot_v1.json. Có within_camera và transfer_ufpr04_to_ufpr05; cùng ngày không giao nhau giữa train/val/test, camera đích không nằm trong train/validation của thí nghiệm chuyển camera. Cột split do split_for() của audit chỉ là xem trước, không dùng cho thực nghiệm. Xem docs/dataset-protocol.md về ngân sách, phiên bản và quy tắc chọn tập con.

Audit cache: audit_summary.json có audit_version và seed; valid_samples.jsonl giữ đường dẫn tương đối. ensure_dataset kiểm tra một frame khi dùng cache. Dữ liệu phải bất biến; force_audit=True để quét lại sau thay đổi. cached_audit_dir cho phép chủ dataset chia sẻ manifest dưới đường dẫn mount khác; kết quả được ghi riêng tại output.

Manifest dự án tách labels (space_id → 0/1) khỏi hình học dùng chung. geometry_id trỏ vào slot_geometries.json; có nhiều layout trên cùng camera. read_project_frame(record, root) đối chiếu nhãn và layout đã khóa với XML thật. Không dùng geometry_id làm định danh xe.
