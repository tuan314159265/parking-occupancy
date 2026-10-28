# Giao diện phiên bản 0.1

`Frame`: sample_id là đường dẫn XML tương đối bỏ suffix; đường dẫn ảnh/XML tương đối dataset_root. Camera giữ nguyên tên thư mục (gồm PUC, PUCPR, UFPR04, UFPR05); date ISO; Cloudy chuẩn hóa thành overcast. Có thể khôi phục metadata gốc qua xml_path.

`ParkingSpace`: space_id duy nhất trong frame, occupied boolean (XML 1=có xe, 0=trống), polygon theo thứ tự contour trong XML. Tọa độ pixel trên ảnh gốc: gốc trái trên, x sang phải, y xuống dưới; biên hình học cho phép x=width/y=height. Không thay contour bằng rotatedRect hoặc hộp xe. Vùng không hợp lệ làm cả frame bị loại và ghi lý do; không bỏ nhãn lỗi âm thầm.

`VehicleDetector.predict(image_path) -> list[VehicleDetection]`: xyxy pixel ảnh gốc, confidence [0,1], class_name. Sau resize/letterbox phải chuyển về tọa độ gốc. Adapter chịu trách nhiệm kiểm tra đầu ra. Chưa chọn hay cài detector; cấu hình detector=null là chủ ý. Bước ghép detection với ô dùng polygon và xuất OccupancyPrediction, chưa triển khai trong hai đầu việc này.

`OccupancyPrediction`: khóa (sample_id, space_id), occupied_score [0,1]. Bước đánh giá sau này phải join đúng khóa, phát hiện thiếu/trùng và chốt ngưỡng trên validation trước khi chạy test. Hàm occupancy_metrics nhận hai dãy binary đã căn chỉnh.

Đánh giá trạng thái ô: occupied là lớp dương, confusion matrix TP/TN/FP/FN, accuracy, precision, recall, F1 (mẫu số 0 trả 0). Báo cáo tổng và theo camera/thời tiết; không tính vehicle mAP từ XML PKLot. Cần nhãn hộp xe độc lập nếu đánh giá detector.

Split dự án: SHA256(seed:date), bucket 70/15/15 theo ngày trên tất cả camera; tỷ lệ thực tế có thể lệch vì số ngày hữu hạn. Đây là giao thức dự án, không phải bản sao split chính thức 50/50. Tất cả ô và camera cùng ngày thuộc cùng split. Có thể thêm leave-one-camera-out sau này, cần tệp cấu hình riêng.
