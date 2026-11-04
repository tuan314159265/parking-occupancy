# Bộ đánh giá trạng thái ô đỗ

Dùng `parking.evaluate_occupancy` cho HOG/SVM, YOLO toàn ảnh và hybrid trên cùng manifest đã khóa. Không đánh giá mAP hộp xe bằng polygon ô đỗ. Hàm không đọc ảnh, không chọn ngưỡng hay tối ưu trên test.

## Đầu vào và ghép khóa

Ground truth nhận các dòng manifest dự án có `labels`, manifest audit có `spaces`, hoặc đối tượng Frame. Manifest phải không rỗng, mỗi sample_id duy nhất, có camera/thời tiết, mỗi ô có nhãn 0/1.

Mỗi dự đoán là một dòng JSONL hoặc dictionary:

```json
{"sample_id":"UFPR04/Sunny/2012-12-15/2012-12-15_08_00_00","space_id":"1","occupied_score":0.82}
```

Có thể thay occupied_score bằng `state`: 0 hoặc "vacant", 1 hoặc "occupied", null hoặc "unknown". Đối tượng OccupancyPrediction cũng được hỗ trợ; occupied_score=null nghĩa là chưa xác định. `slot_id` là tên thay thế cho space_id; nếu có cả hai thì phải trùng nhau.

occupied_score là điểm có xe của từng ô sau bước ghép xe–ô, không phải confidence của box xe chưa được ghép. HOG/SVM có thể xuất state từ predict; decision_function chưa chuẩn hóa không phải xác suất [0,1].

Điểm số phải hữu hạn, trong [0,1]; score >= threshold được coi là có xe. Nếu có cả state và score, state chưa xác định có quyền từ chối dự đoán dù score cao; state xác định phải nhất quán với score tại ngưỡng đang đánh giá. Khi cần thử ngưỡng trên validation, nên xuất score và không chốt state vào file trước.

Ghép bằng `(sample_id, space_id)`, không ghép theo thứ tự dòng. Dự đoán trùng khóa hoặc có khóa ngoài manifest sẽ báo lỗi. Thiếu dự đoán được tính là unknown và báo số lượng qua missing_predictions; đặt missing_policy="error" nếu muốn bắt buộc đủ dòng. Nhãn ground truth chưa xác định luôn bị từ chối, không dùng làm nhãn trống.

## Công thức

Có xe là lớp dương. TP/TN/FP/FN chỉ đếm dự đoán xác định. U+ là ô có xe bị dự đoán unknown; U− là ô trống bị dự đoán unknown. N là tổng số ô; K là tổng ô có dự đoán xác định.

| Chỉ số chính | Công thức |
|---|---|
| Coverage | K / N |
| Accuracy tổng | (TP + TN) / N; mọi unknown đều tính sai |
| Precision có xe | TP / (TP + FP) |
| Recall có xe | TP / (TP + FN + U+) |
| F1 có xe | 2TP / (2TP + FP + FN + U+) |
| false_vacancy_rate | FN / (TP + FN + U+): tỉ lệ ô thực tế có xe bị báo trống |
| reported_vacancy_error_rate | FN / (TN + FN): tỉ lệ sai trong các ô được báo trống |

Precision/recall/F1 trả 0 khi mẫu số bằng 0. Hai tỉ lệ báo trống trả null nếu không có support tương ứng. Không dùng tỉ lệ báo trống đơn độc để đánh giá phương pháp từ chối nhiều dự đoán; xem cùng coverage/accuracy/F1.

`covered_only` trả accuracy/precision/recall/F1 chuẩn nhị phân chỉ trên ô đã xác định, để chẩn đoán. Không thay accuracy tổng bằng covered_only khi so sánh các phương pháp. Khi không có unknown, các chỉ số chính khớp công thức nhị phân thông thường và hàm occupancy_metrics cũ.

Ma trận chính có 2 hàng `[vacant, occupied]` và 3 cột `[vacant, occupied, unknown]`. Ma trận 2×2 chỉ trên ô xác định cũng được xuất qua confusion_matrix_covered; không âm thầm bỏ unknown khỏi mẫu số accuracy tổng.

## Đánh giá phép đếm

Trên mỗi ảnh, đếm số ô được dự đoán trống bằng số state=0, không coi unknown là trống. `mae_vacant_count` là trung bình |số trống dự đoán − số trống thật| theo ảnh, không theo ô. Có thêm mae_occupied_count và complete_frame_coverage (tỉ lệ ảnh không còn ô unknown).

Per-frame lưu unknown_count, vacant_count_lower/upper. Đây là khoảng số ô trống có thể báo nếu giải quyết các unknown theo dự đoán: từ số đã báo trống tới số đó cộng unknown_count. Khoảng này không đảm bảo chứa số ô trống thật vì dự đoán xác định cũng có thể sai.

## Thời gian theo ảnh gốc

Mỗi dòng timing JSONL:

```json
{"sample_id":"A","preprocess_ms":10,"model_ms":20,"postprocess_ms":5,"total_ms":40,"inference_calls":1}
```

Tất cả thời gian là mili giây, hữu hạn và không âm. total_ms phải bao gồm tổng thời gian các giai đoạn; có thể lớn hơn tổng do overhead. Một sample_id chỉ có một dòng timing; không dùng một dòng cho từng tile. Với hybrid, model_ms cộng mọi lượt model và postprocess_ms gồm mapping box, NMS toàn cục, ghép ô và đếm. Image loading nằm trong preprocess_ms. Khởi tạo model, tải checkpoint, tải dataset và warm-up được làm trước phép đo.

Thống kê báo mean tiền xử lý/model/hậu xử lý/tổng trên **ảnh gốc**, p50/p95 tổng, tổng và trung bình số lượt suy luận. Không chia thời gian tổng cho số tile để báo tốc độ ảnh gốc. Nếu timing bị thiếu, thống kê chỉ dùng các ảnh đã đo và báo timing.coverage cùng danh sách ảnh thiếu; không điền thời gian 0. Chỉ so sánh tốc độ khi cùng thiết bị, điều kiện warm-up và timing coverage phù hợp; ghi các thông tin này trong metadata.

FrameTimer hỗ trợ các stage preprocess/model/postprocess, cộng các stage model lặp lại:

```python
with parking.FrameTimer(frame.sample_id, synchronize=sync_gpu_or_none) as timer:
    with timer.stage("preprocess"):
        image, tiles = prepare_image(frame)
    boxes = []
    for tile in tiles:
        with timer.stage("model"):
            boxes.extend(infer(tile))
    with timer.stage("postprocess"):
        predictions = map_merge_and_assign(boxes)
timing = timer.record()
```

Hàm đồng bộ GPU do caller truyền vào khi inference bất đồng bộ; CPU dùng None. Tạo timer mới cho mỗi ảnh, không lồng stage và không xuất record nếu pipeline lỗi hoặc thiếu giai đoạn. Nếu model xử lý nhiều tile trong một batch, truyền inference_calls thực tế vào record khi cần.

## Ví dụ kiểm chứng bằng tay

| Ảnh | Nhãn thật | Dự đoán |
|---|---|---|
| A | 1, 1, 0, 0 | 1, 0, 0, unknown |
| B | 1, 0, 1 | unknown, 1, 1 |

Ma trận 2×3: `[[1, 1, 1], [1, 2, 1]]`. Có TP=2, TN=1, FP=1, FN=1, U+=1, U−=1.

Coverage=5/7; accuracy=3/7; precision=2/3; recall=1/2; F1=4/7; false_vacancy_rate=1/4; reported_vacancy_error_rate=1/2. Ảnh A có 2 ô trống thật và báo 2; B có 1 ô trống thật và báo 0, nên MAE đếm trống=(0+1)/2=0,5. Timing tổng A=40 ms và B=120 ms thì mean=80 ms/ảnh, dù B chạy 4 lượt inference.

`parking.evaluator_self_check()` xác nhận các giá trị trên. Ví dụ là dữ liệu nhân tạo để kiểm chứng công thức, không phải kết quả của mô hình hay kết quả thực nghiệm PKLot.

## Cách dùng và đầu ra

```python
report = parking.evaluate_occupancy(
    VAL_FRAMES, predictions, timings=timings,
    threshold=0.5, missing_policy="unknown", method="hog_svm",
    metadata={"protocol": PROTOCOL, "split": "val", "device": "...", "warmup_runs": 10},
)
parking.export_evaluation(report, WORK_DIR / "outputs/evaluation/hog_svm/val")
```

Chỉ chọn threshold/quy tắc trên validation rồi giữ nguyên trên test. Với giao thức chuyển camera, dùng train/validation của chính giao thức đó. Evaluator không biết lịch sử chọn tham số, nên caller phải tuân thủ quy tắc này.

Report có overall và các nhóm by_camera, by_weather, by_camera_weather. Mọi chỉ số được cộng theo ô trong nhóm; MAE/thời gian được trung bình theo ảnh. Không lấy trung bình F1 từng ảnh để thay F1 toàn bộ.

Các file xuất:

- metrics.json: chỉ số tổng, theo nhóm, threshold, method và metadata.
- per_frame.csv: phép đếm, unknown và thời gian từng ảnh.
- per_slot.csv: nhãn thật/dự đoán/score/missing đã ghép đúng khóa; unknown để trống ở cột y_pred.
- aligned_predictions.jsonl: cùng thông tin từng ô, giữ null cho unknown để truy vết.

File aligned_predictions là đầu ra kiểm tra có y_true/y_pred, không phải schema dự đoán đầu vào. API nhận danh sách dictionary; đường dẫn file đầu vào dùng JSONL.

CLI:

```bash
python parking_module.py evaluate \
  --manifest data/manifests/pklot_v1/within_camera/val.jsonl \
  --predictions outputs/hog_svm/predictions.jsonl \
  --timings outputs/hog_svm/timings.jsonl \
  --method hog_svm --threshold 0.5 \
  --output outputs/evaluation/hog_svm/val
```

Bỏ --timings nếu chưa đo. CLI tự ghi SHA256 manifest, predictions và timing (nếu có) vào metadata. Có thể thêm --metadata metadata.json để ghi protocol/device/environment. Notebook mục 6.1 có self-check và cell gọi evaluator; để trống PREDICTIONS_PATH sẽ chỉ báo sẵn sàng, không tạo kết quả mô hình giả.
