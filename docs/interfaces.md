# Giao diện phiên bản 0.3

`Frame`: sample_id là đường dẫn XML tương đối bỏ suffix; đường dẫn ảnh/XML tương đối dataset_root. Camera giữ nguyên tên thư mục (gồm PUC, PUCPR, UFPR04, UFPR05); date ISO theo timestamp tên file khi hợp lệ, directory_date giữ ngày thư mục và date_source ghi nguồn; Cloudy chuẩn hóa thành overcast. Có thể khôi phục metadata gốc qua xml_path.

`ParkingSpace`: space_id duy nhất trong frame, occupied boolean (XML 1=có xe, 0=trống), polygon theo thứ tự contour trong XML. Tọa độ pixel trên ảnh gốc: gốc trái trên, x sang phải, y xuống dưới; biên hình học cho phép x=width/y=height. Không thay contour bằng rotatedRect hoặc hộp xe. Vùng không hợp lệ làm cả frame bị loại và ghi lý do; không bỏ nhãn lỗi âm thầm.

`VehicleDetector.predict(image_path) -> list[VehicleDetection]`: xyxy pixel ảnh gốc, confidence [0,1], class_name. Sau resize/letterbox phải chuyển về tọa độ gốc. Adapter chịu trách nhiệm kiểm tra đầu ra. Chưa chọn hay cài detector; cấu hình detector=null là chủ ý. Bước ghép detection với ô dùng polygon và xuất OccupancyPrediction, chưa triển khai trong hai đầu việc này.

`OccupancyPrediction`: khóa (sample_id, space_id), occupied_score [0,1] hoặc None cho chưa xác định. evaluate_occupancy ghép theo khóa, từ chối trùng/ngoài manifest, tính ô thiếu là unknown (hoặc báo lỗi khi missing_policy="error"). Dictionary dự đoán cũng nhận state=0/1/None và slot_id là alias của space_id. Chốt ngưỡng trên validation trước khi test. occupancy_metrics là helper nhị phân cũ; evaluate_occupancy là evaluator dùng chung.

Đánh giá trạng thái ô: occupied là lớp dương. Ma trận chính 2×3 giữ unknown; accuracy tổng coi unknown là sai, coverage báo tỉ lệ xác định. Recall/F1 chính phạt unknown trên ô có xe; covered_only để chẩn đoán. Có tỉ lệ báo trống sai với hai mẫu số, MAE đếm ô trống/có xe, và timing theo ảnh gốc gồm mọi tile. Báo tổng, theo camera/thời tiết. Chi tiết công thức và schema trong docs/evaluation.md; không tính vehicle mAP từ XML PKLot.

Split chính thức của dự án: dùng manifest đã khóa ở data/manifests/pklot_v1/ và cấu hình configs/pklot_v1.json. Có within_camera và transfer_ufpr04_to_ufpr05; cùng ngày không giao nhau giữa train/val/test, camera đích không nằm trong train/validation của thí nghiệm chuyển camera. Cột split do split_for() của audit chỉ là xem trước, không dùng cho thực nghiệm. Xem docs/dataset-protocol.md về ngân sách, phiên bản và quy tắc chọn tập con.

Audit cache: audit_summary.json có audit_version và seed; valid_samples.jsonl giữ đường dẫn tương đối. ensure_dataset kiểm tra một frame khi dùng cache. Dữ liệu phải bất biến; force_audit=True để quét lại sau thay đổi. cached_audit_dir cho phép chủ dataset chia sẻ manifest dưới đường dẫn mount khác; kết quả được ghi riêng tại output.

Manifest dự án tách labels (space_id → 0/1) khỏi hình học dùng chung. geometry_id trỏ vào slot_geometries.json; có nhiều layout trên cùng camera. read_project_frame(record, root) đối chiếu nhãn và layout đã khóa với XML thật. Không dùng geometry_id làm định danh xe.

Validation dùng `run_validation_search` với plan đã đăng ký. `validation_pixel_shape` quy đổi ngân sách 1/2/4 tile vuông sang tensor toàn ảnh đúng số pixel. Confidence floor của model cố định; ngưỡng confidence tìm kiếm lọc box sau gộp, trước suy trạng thái. SVM xuất state từ decision_margin, không coi margin là occupied_score [0,1]. `verify_validation_lock` đối chiếu checkpoint/split/model/tham chiếu/engine/phiên bản trước `run_locked_test`; chỉ run đầy đủ tạo khóa. Xem docs/validation.md và reports/validation_v1.
