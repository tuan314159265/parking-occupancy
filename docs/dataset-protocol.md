# Bộ dữ liệu dùng chung: PKLot v1

Tất cả phương pháp dùng cùng manifest trong `data/manifests/pklot_v1/`. Không dùng cột split xem trước của audit và không tự chia lại trên từng máy. Cấu hình nguồn ở `configs/pklot_v1.json`; seed 42. Bộ này là giao thức của dự án, không phải split chính thức của PKLot.

| Giao thức | Train | Validation | Test | Phạm vi |
|---|---:|---:|---:|---|
| within_camera | 1.182 | 270 | 270 | Cả PUCPR, UFPR04 và UFPR05 có mặt ở mỗi split |
| transfer_ufpr04_to_ufpr05 | 420 | 90 | 90 | Train/validation chỉ UFPR04; test chỉ UFPR05 |

UFPR04 và UFPR05 là hai góc nhìn **cùng bãi UFPR**. Giao thức thứ hai kiểm tra chuyển góc nhìn, không chứng minh chuyển sang bãi khác. Phải huấn luyện lại HOG/SVM và chọn ngưỡng/quy tắc chỉ trên train/validation của giao thức tương ứng. Không dùng model, ngưỡng hay quy tắc đã chỉnh trên UFPR05 để báo kết quả chuyển góc nhìn. Hai giao thức là hai thí nghiệm độc lập; không gộp các manifest của chúng.

## Quy tắc chia và giới hạn máy

- Nhóm theo **ngày thu nhận**, không theo crop, thời tiết hoặc từng frame. Mọi camera có cùng ngày chỉ thuộc một split trong mỗi giao thức. Giữ nguyên mọi ô của một frame cùng split.
- Trong giao thức chính, chia ngày xấp xỉ 70/15/15. Chọn trong 256 phương án xác định bởi seed để đủ camera/thời tiết và gần ngân sách. Chỉ dùng số lượng và metadata, không dùng nhãn trạng thái, dự đoán hay điểm số test để chọn phương án.
- Trong giao thức chuyển camera, giữ toàn bộ ngày của UFPR05 cho test; bỏ các ngày đó khỏi nguồn trước khi chia nguồn UFPR04 xấp xỉ 80/20.
- Giới hạn mỗi cặp camera/thời tiết: tối đa 140 train, 30 validation, 30 test. Chọn luân phiên các ngày, sắp thứ tự frame bằng hash seed/sample_id để không lấy hết một ngày dài trước các ngày khác.
- Ngân sách giao thức chính tối đa 1.800 ảnh, thực tế 1.722. Nhóm train UFPR04/rainy thiếu 22 và UFPR05/rainy thiếu 56 so với cap; không lấy bù từ test và không lặp ảnh.
- Validation/test có đúng 30 frame cho mỗi camera/thời tiết được dùng. Đây là tập con để so sánh có kiểm soát, không đại diện tỉ lệ thời tiết tự nhiên của toàn PKLot. Nếu tổng hợp accuracy/F1, báo rõ giao thức và phân bố này.

## Dữ liệu được kiểm tra

Audit loader v2 trên archive local: 12.416 XML; **11.638 frame hợp lệ**, 778 frame bị loại do nhãn/id ô không hợp lệ, và 1 ảnh thiếu XML. Không suy đoán nhãn thiếu hoặc bỏ âm thầm một ô lỗi. Kết quả audit gọn nằm ở `inventory_audit.json`; log từng lỗi và manifest toàn bộ để ở outputs local, không push dữ liệu lớn.

Có XML dùng `Point` thay cho `point`, loader hỗ trợ cả hai. Ngày chuẩn lấy từ tên file có timestamp hợp lệ; trường `directory_date` giữ nguyên tên thư mục, `date_source` ghi nguồn ngày. Có 195 frame hợp lệ mà ngày chuẩn khác ngày thư mục, bao gồm tên thư mục tháng 14; không đổi tên dữ liệu gốc.

## Manifest và hình học

Mỗi dòng trong `train.jsonl`, `val.jsonl`, `test.jsonl` gồm:

- `sample_id`, `image_path`, `xml_path`: đường dẫn tương đối dataset root, dùng được qua shortcut Drive khác nhau.
- `camera`, `lot`, `weather`, `date`, `directory_date`, `date_source`, `width`, `height`.
- `split`, `geometry_id`, `labels`: ánh xạ space_id → 0 (trống) hoặc 1 (có xe).

`slot_geometries.json` chứa 5 layout: PUCPR 1, UFPR04 3 và UFPR05 1. Mỗi layout giữ camera, kích thước ảnh và polygon theo space_id. Dùng `geometry_id` của từng frame, không gán một layout cố định cho mọi ảnh của camera. Polygon là vùng ô đỗ, không phải hộp xe; JSON hình học không chứa nhãn occupancy.

Giao diện dùng trên Colab:

```python
project = parking.fetch_project_splits(OWNER, REPO, REF, WORK_DIR / "splits/pklot_v1")
frames = project["protocols"]["within_camera"]["train"]
layouts = project["geometries"]
frame = parking.read_project_frame(frames[0], DATASET_DIR)
```

Notebook mục 3.2 xuất sẵn TRAIN_FRAMES, VAL_FRAMES, TEST_FRAMES, SLOT_GEOMETRIES và MANIFEST_DIR. Chỉ tải metadata khoảng 1,8 MB từ GitHub; ảnh vẫn đọc từ dataset cá nhân hoặc Drive chung. Không ghi vào thư mục dataset chung.

## Khóa phiên bản và tái lập

`split_lock.json` ghi config, SHA256 manifest nguồn, danh sách ngày, số frame/ô/nhãn và SHA256 từng artifact. Bộ tải xác minh checksum và các điều kiện: không trùng sample/ngày giữa split, camera đích chỉ có ở test của thí nghiệm chuyển camera, nhãn khớp ID ô hình học.

Đã tái sinh hai lần với cùng dữ liệu cho cùng checksum; đã đọc lại toàn bộ **1.763 frame duy nhất** của cả hai thí nghiệm để đối chiếu nhãn và hình học với XML thật. Không có ngày giao nhau trong train/validation/test của từng thí nghiệm.

Tái lập từ dataset đã giải nén:

```bash
python parking_module.py audit --root data/raw/PKLot --output outputs/task3_audit
python parking_module.py splits --manifest outputs/task3_audit/valid_samples.jsonl --config configs/pklot_v1.json --output data/manifests/pklot_v1
```

Nếu đã tồn tại lock, chương trình chỉ dùng lại khi nguồn/config/checksum không đổi. Đổi ngân sách, seed hoặc inventory phải tạo phiên bản/thư mục mới để cả nhóm cập nhật cùng nhau. `--overwrite` dành cho lúc xây dựng bộ mới, không tự gọi trong notebook. Pin REF vào commit SHA khi chạy thí nghiệm chính thức.
