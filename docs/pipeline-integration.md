# Tích hợp pipeline và trực quan hóa (task của Vinh): những gì đã thực hiện

**Ngày:** 08/11/2026
**Phạm vi:** merge nhánh `feat/parking-line-geometry` (Văn) vào `main`, review code; hoàn thiện mục 5 (pipeline), 6.2b, 6.3, 7, 8, 9 của notebook `notebooks/01_pklot_run_all.ipynb`; bổ sung hàm dùng chung vào `parking_module.py`.

## 1. Bối cảnh: các phần của nhóm được tích hợp

| Mục notebook | Task / người làm | Pipeline gọi lại |
|---|---|---|
| 3.1–3.2 | Tuấn: loader, audit, split `pklot_v1`, JSON vùng ô | `read_project_frame`, `TRAIN/VAL/TEST_FRAMES`, `SLOT_GEOMETRIES` |
| 4.1 | Hữu: HOG + SVM | `predict_hog_frame`, `HOG_SVM_BUNDLE` |
| 4.2 | Văn: Canny + Hough, cờ chất lượng hình học | `parking.analyze_parking_geometry`, `GEOMETRY_CONFIG` |
| 4.3 | Văn: ROI màu, tile, quy đổi tọa độ | `parking.make_parking_tiles`, `parking.tile_boxes_to_image`, `TILE_CONFIG` |
| 4.4 | Hữu: YOLO11n full-frame | `FULLFRAME_DETECTOR` |
| 4.5 | Hữu (+ Văn sửa dùng tiler chung): hybrid ROI/tile + global NMS | `predict_hybrid_frame` |
| 4.6, 6.2 | Văn: ghép xe–ô, chọn cấu hình chung trên validation | `parking.infer_parking_occupancy`, `FROZEN_OCCUPANCY_CONFIG`, `OCCUPANCY_VAL` |
| 6.1 | Tuấn: bộ đánh giá chung | `parking.evaluate_occupancy`, `FrameTimer`, `export_evaluation` |
| **5, 6.2b, 6.3, 7–9** | **Vinh: tích hợp pipeline và trực quan hóa** | (phần mô tả dưới đây) |

## 2. Review nhánh `feat/parking-line-geometry` (10 commit của Văn)

**Kết luận: đạt, đã merge** (`git merge --no-ff`, commit `effd2bd`).

Điểm tốt:
- Hàm dùng chung nằm trong module, có kiểm tra đầu vào chặt và không đọc nhãn occupancy.
- Quyết định ghép xe–ô không phụ thuộc thứ tự đầu vào. Mỗi xe và mỗi ô chỉ tham gia tối đa một cặp ghép; trường hợp gần hòa thì không gán (abstain).
- Cùng một quy tắc cho full-frame và hybrid. Tiler bảo đảm mọi ô và vùng ngữ cảnh nằm trọn trong ít nhất một tile. Có 53 test mới, gồm cả test chạy trực tiếp cell notebook.
- Mọi layout `pklot_v1` đều là polygon lồi, đúng giả định của `score_vehicle_spaces`. Đã kiểm tra cả 5 layout.

Vấn đề tìm thấy và đã xử lý:
1. **Hiệu năng `analyze_parking_geometry`:** mất 3–6 s/ảnh PUCPR (100 ô, 400–800 đoạn), vì mỗi cạnh so với mọi đoạn cùng hướng. Đã thêm bộ lọc bounding box. Đầu ra **giống hệt từng byte** trên 12 ảnh PKLot thật, nhanh hơn khoảng **23 lần** (commit `c500b0e`).
2. **Cờ tuyệt đối không đủ để đánh dấu hình học yếu:** trên ảnh thật, chỉ khoảng 40–65% ô có cờ `supported` ngay cả ở ảnh bình thường, vì vạch sơn không luôn trùng cạnh polygon XML. Mục 5 bổ sung đánh giá so với tham chiếu train (mục 3 bên dưới), không sửa hàm của Văn.
3. **Checksum manifest trên Windows:** `core.autocrlf` đổi manifest sang CRLF nên lệch SHA256 trong `split_lock.json`. Đã thêm `.gitattributes` để giữ LF (commit `3697841`).

4. **Đầu ra ở hai gốc khác nhau:** trước đây ghi vào cả `OUTPUT_DIR/<PROTOCOL>/...` và `WORK_DIR/outputs/...`. Đã gom 4.2 (`geometry/`), 4.6/6.2 (`occupancy/`) và 6.1 (`evaluation/`) về `OUTPUT_DIR/<PROTOCOL>/`, và cập nhật các test tương ứng.
5. **`imgsz` cố định ở 640 bỏ sót xe nhỏ (PUCPR):** cell 4.4 giờ chọn `imgsz` trong `YOLO_IMGSZ_CANDIDATES = [640, 960, 1280]` trên validation, theo F1 occupancy (quy tắc ghép mặc định của 4.6 + evaluator chung; hòa thì chọn kích thước nhỏ hơn). Kết quả của kích thước được chọn dùng lại luôn, không chạy thêm. Hybrid dùng cùng detector nên cùng `imgsz`. Cách chọn được ghi trong `yolo_fullframe/run_config.json`. Trên 24 ảnh thật của dữ liệu thử, số box/ảnh tăng từ 9,9 (640) lên 15,7 (960) và 26,9 (1280).

Góp ý chưa sửa:
- `tests/test_tiles.py` ghi vào `/tmp/parking-tiles-preview.png`, nên test này lỗi trên Windows (trên Colab/Linux thì chạy được).
- 4.4 truyền `half=False`, và Ultralytics in cảnh báo deprecated ở mỗi lần gọi (không ảnh hưởng kết quả).

## 3. Phần đã làm

### 3.1. Hàm dùng chung mới trong `parking_module.py` (commit `26b1106`, 11 test trong `tests/test_geometry_quality.py`)
- `build_geometry_reference(images, layout, ...)`: từ ảnh train của một `geometry_id`, tính ảnh median, tỉ lệ `supported` và median `boundary_support` của từng ô, cùng median số đoạn thẳng. **Không dùng nhãn.**
- `assess_geometry_quality(geometry, reference)`: tạo cờ hình học yếu so với tham chiếu.
  - Cấp frame: `layout_shift_suspected` (phase correlation giữa gradient ảnh và ảnh tham chiếu lệch quá 6 px), `few_line_segments` (số đoạn < 25% median tham chiếu).
  - Cấp ô: `boundary_lost` (ô thường `supported` nhưng ảnh này mất vạch). Ô vốn không bao giờ có vạch thì không bị gắn cờ.
  - Không có tham chiếu (ví dụ UFPR05 trong giao thức chuyển camera): không gắn cờ và ghi `reference_available=False`.
- `apply_geometry_policy(predictions, quality, policy)`: với `"unknown"`, ô **vacant** có cờ yếu chuyển thành **unknown**; ô occupied hoặc unknown không bị đổi. Với `"report_only"`, trạng thái giữ nguyên và chỉ ghi lý do (dùng cho phân tích).

Hiệu chỉnh trên 48 ảnh PKLot thật:

| Biến thể ảnh | Frame bị gắn cờ | Ô bị gắn cờ |
|---|---|---|
| Ảnh gốc | 0–4,5% | 2–10% |
| Dịch 10 px | 100% | — |
| Dịch 7 px | 64–91% | — |
| Giảm độ sáng còn 30% | 100% (`few_line_segments`) | — |

### 3.2. Thay đổi tối thiểu ở mục 4 (hành vi cũ giữ nguyên)
- 4.4: tách `FullFrameYOLODetector.predict_image(image)`; `predict(image_path)` đọc ảnh rồi gọi lại hàm này. Bỏ dòng `# TODO` thừa.
- 4.4: thêm bước chọn `imgsz` trên validation (xem mục 2, điểm 5).
- 4.5: `predict_hybrid_frame(frame, image=None)`, nhận ảnh đã đọc để pipeline không đọc ảnh hai lần.

### 3.3. Mục 5: pipeline tích hợp (`run_pipeline_frame`, `run_pipeline`)
Một ảnh gốc đi qua các bước sau, đo bằng `parking.FrameTimer`:

| Giai đoạn đo | Bước |
|---|---|
| preprocess | Đọc ảnh BGR → `analyze_parking_geometry` → `assess_geometry_quality` |
| model | HOG/SVM (`predict_hog_frame`), hoặc YOLO toàn ảnh (`predict_image`), hoặc hybrid (`predict_hybrid_frame`: tile → YOLO từng tile → `tile_boxes_to_image` → global NMS) |
| postprocess | `infer_parking_occupancy` (HOG: margin > 0) → `apply_geometry_policy` → đếm |

Các phần khác của mục 5:
- **Thông báo:** `⚠ [Hình học yếu] <sample>: <lý do>`, kèm số ô đã chuyển sang UNKNOWN. Tất cả được ghi vào `warnings.txt`.
- **Trực quan hóa** `visualize_pipeline_result`, 6 panel:
  1. Ảnh gốc.
  2. Canny trong vùng bãi.
  3. Đoạn Hough + ô mất vạch + độ lệch camera.
  4. ROI và tile.
  5. Box xe trên ảnh gốc + phần dưới box dùng để ghép.
  6. Ô tô màu: xanh = trống, đỏ = có xe, cam = chưa xác định.
- **Xuất file:** `slots.csv`, `frames.jsonl`, `predictions.jsonl`, `timings.jsonl`, `warnings.txt`, `run_config.json`, `evaluation/`.
- **Demo:** cả ba phương pháp trên một ảnh validation; kết quả ở `pipeline/demo/`.
- `rescore_result` tính lại trạng thái theo chính sách hoặc cấu hình khác mà không chạy lại model.

### 3.4. Mục 6.2b, 6.3, 7, 8, 9
- **6.2b:** dùng lại `OCCUPANCY_VAL` (cấu hình chung Văn đã chọn) và `HOG_VAL_PREDICTIONS`, nên không chạy lại YOLO. Cell tính cờ hình học trên validation, so sánh hai chính sách `unknown` và `report_only`, thống kê cờ theo camera/thời tiết, rồi ghi `pipeline/locked_config.json`.
- **6.3:** chạy pipeline đầy đủ của 3 phương pháp trên test với cấu hình đã khóa, có timing và đánh giá. Có bảng `report_only` để phân tích. Mỗi cặp camera × thời tiết có một ảnh minh họa cho cả ba phương pháp.
- **7:** bảng tổng hợp val/test, F1 theo camera/thời tiết, đánh đổi chất lượng–thời gian, và `observations.txt`. Nhận xét chỉ được sinh từ số đo.
- **8–9:** kết luận, giới hạn, hướng phát triển; hướng dẫn tái lập và `run_manifest.json` (SHA256 của mọi đầu ra).

## 4. Kiểm chứng
- **Dữ liệu thử:** 60 MB đầu archive PKLot chính thức, gồm 141 ảnh thật của PUCPR, UFPR04, UFPR05, đủ 5 layout. Mỗi ảnh được ghép với XML gần nhất về thời gian, nên polygon là thật nhưng **nhãn không khớp ảnh**.
- **Chạy notebook:** mọi code cell từ 4.1 đến 9 chạy thành công trên dữ liệu thử (60 train / 24 val / 24 test). Chỉ các cell Drive/tải module/tải split được thay bằng dữ liệu local. Môi trường: Python 3.13, OpenCV 5.0, torch CPU, Ultralytics 8.4.
- **Unit test:** 102 test, 95 đạt. 7 lỗi chỉ xảy ra trên Windows: 6 lỗi `test_project_splits` đã có trên `main` từ trước merge, và 1 test ghi `/tmp`. Các test chạy cell notebook (hybrid, occupancy, validation) đều đạt sau khi chỉnh 4.4/4.5.
- **Chưa có số liệu thực nghiệm.** Các con số F1 từ dữ liệu thử không có ý nghĩa vì nhãn không khớp ảnh. Cần chạy **Run All trên Colab** với PKLot đầy đủ.

## 5. Cách chạy
1. Push `main` (có `parking_module.py` mới) lên GitHub. Notebook tải module từ `REF`.
2. Trên Colab: **Runtime → Run all** (chi tiết ở README, mục "Chạy notebook hoàn chỉnh trên Colab"). Có thể đặt `PIPELINE_MAX_FRAMES = 30` ở 6.3 để chạy thử nhanh; kết quả lúc đó được ghi `partial_run: true`.
3. Đầu ra pipeline nằm ở `OUTPUT_DIR/<PROTOCOL>/pipeline/`: `geometry_reference/`, `demo/`, `val/`, `test/`, `figures/`, `results/`, `locked_config.json`, `run_manifest.json`.

## 6. Cập nhật 09/11/2026: validation chính thức

Task 12 bổ sung runner độc lập, grid giới hạn, đối chứng toàn ảnh khớp pixel, khóa bằng checksum và cổng test rõ ràng. Mặc định VALIDATION_ONLY_MODE=True để bỏ demo nặng bị trùng; RUN_FINAL_TEST=False. Các cell/hàm minh họa cũ vẫn giữ khi cần chạy riêng. Số liệu thật và lịch sử test cũ xem reports/validation_v1/README.md; hướng dẫn hiện hành ở docs/validation.md. Các kết quả/giới hạn mô tả bên trên là lịch sử lần tích hợp cũ, không thay thế báo cáo validation mới.
