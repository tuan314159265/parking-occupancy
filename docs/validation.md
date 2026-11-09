# Validation giới hạn và khóa cấu hình trước test

Task 12 dùng `run_validation_search` trong module độc lập. Plan tại `configs/validation_v1.json` được ghi thành `preregistered_plan.json` trước khi đọc kết quả tìm kiếm. Plan v1 dành cho giao thức `within_camera`: 1.182 ảnh train (67.704 ô), 270 validation (15.120 ô). Ảnh test không được suy luận hoặc chấm điểm; chỉ checksum của manifest test được lưu để bảo vệ bước sau.

## Review code sau khi fetch

Code đã có HOG/SVM, detector pretrained, ROI/tile, mapping và ghép ô. Runner mới giữ các phép xử lý đó và khắc phục các khoảng trống của luồng tìm kiếm cũ:

- Có grid padding/overlap thay vì chỉ thử một cấu hình tile.
- Thêm toàn ảnh độ phân giải cao, cân bằng pixel tensor thực tế với tiling, không chỉ so imgsz.
- SVM margin được chuyển thành state theo ngưỡng, không đưa margin chưa chuẩn hóa vào occupied_score [0,1].
- Khóa checkpoint bằng SHA256, split train/val/test, JSON layout, HOG model, tham chiếu hình học, source của engine và phiên bản thư viện.
- Lưu cả cấu hình không cải thiện; chọn cùng confidence hậu xử lý và cùng quy tắc ghép cho ba nhánh YOLO.
- Test cần chủ động bật RUN_FINAL_TEST; không tự chạy khi kết thúc validation.

Các cell demo cũ 4.1/4.4/4.5/4.6 và pipeline được giữ ở nhánh `VALIDATION_ONLY_MODE=False`. Chế độ mặc định True chạy runner chính thức ở 6.2 để tránh huấn luyện/inference hai lần. Cell hình học/ROI minh họa vẫn có thể chạy. Các cell độc lập không còn bắt buộc có helper đồng bộ Drive khi chạy local/unit test.

## Dải tìm kiếm đã chốt

| Thành phần | Giá trị |
|---|---|
| HOG | 64×128, cell 8×8, block 16×16, stride 8×8, 9 bins, L2-Hys; crop bao polygon và Pillow BILINEAR như baseline |
| Linear SVM | C = 0,01; 0,1; 1; dual=False, max_iter=20.000, seed 42 |
| Ngưỡng SVM margin | −0,25; 0; 0,25 (state occupied khi margin > ngưỡng) |
| YOLO | YOLO11n COCO, Ultralytics 8.4.0, FP32, không augment, rect=False, base imgsz 640 |
| Lớp xe | car, bus, truck; giống nhau ở mọi nhánh |
| Confidence đầu vào YOLO | Cố định 0,15 để cache detection ở cùng một mức nền |
| Confidence hậu xử lý | 0,15; 0,25, lọc box **sau** bước gộp/NMS hybrid, trước ghép ô |
| Padding ROI | 16; 32 pixel ảnh gốc |
| Overlap yêu cầu | 0,20; 0,40; tối đa 0,60 theo quy tắc giữ toàn bộ ô/ngữ cảnh |
| Tile | Tối đa lưới 2×2, min_tile_size 256; fallback về ít tile hơn nếu cần |
| Phần dưới box | 0,33; 0,50; 0,67 |
| Ngưỡng overlap ghép ô | 0,20; 0,30; 0,40 |
| Biên mơ hồ | 0; 0,05 |
| Cố định | min_score 0,15; occupancy NMS 0,5; global hybrid NMS 0,5; biên crop 4 px; box tối thiểu 2 px |
| Hình học yếu | unknown; tham chiếu lấy tối đa 15 ảnh train/layout; không chỉnh tham chiếu bằng validation/test |

Tổng cộng 9 cấu hình HOG và 144 tổ hợp joint detector: 4 tile request × 2 ngưỡng confidence hậu xử lý × 18 quy tắc ghép. Mỗi tổ hợp joint được chấm cho YOLO toàn ảnh, hybrid và toàn ảnh pixel-matched. Model không chạy lại cho từng ngưỡng ghép; cache pair score theo frame/fraction và giữ nguyên quyết định NMS. Overlap hiệu dụng có thể tăng để giữ trọn ô, nên hai overlap yêu cầu có thể cho cùng crop; tile_plans.json ghi rõ trường hợp này.

Confidence hậu xử lý không đồng nghĩa với thay conf trong lần gọi YOLO. Đây là chủ ý của giao thức để dùng lại cùng detection nền; đặc biệt NMS hybrid ưu tiên box không cắt biên có thể giữ box confidence thấp hơn. Quy tắc này được giữ nguyên khi benchmark và test.

## Đối chứng độ phân giải/pixel

Tất cả nhánh dùng cùng checkpoint, precision và rect=False. Mỗi tile vào model là 640×640. Số tile thực tế từ layout và cấu hình đã khóa quyết định tensor toàn ảnh đối chứng:

| Tile thực tế | Tensor toàn ảnh (H×W) | Tổng pixel mỗi nhánh |
|---:|---|---:|
| 1 | 640×640 | 409.600 |
| 2 | 640×1280 | 819.200 |
| 4 | 1280×1280 | 1.638.400 |

Forward hook trên model ghi kích thước tensor thật, kể cả letterbox; callback reset bộ đếm sau backend warm-up để GPU warm-up không bị tính thành ảnh thứ hai; runner từ chối khi kích thước không khớp. Không lấy diện tích ảnh gốc hoặc imgsz² khi model đang dùng rectangular padding để giả định ngân sách. Theo [tài liệu inference Ultralytics](https://docs.ultralytics.com/modes/predict/), quy tắc padding phụ thuộc rect và kích thước đầu vào; vì vậy v1 cố định rect=False.

Pixel ngang nhau không có nghĩa latency hoặc FLOPs hoàn toàn ngang nhau. Hybrid có nhiều lượt gọi và overhead; kết quả cuối phải xem cả thời gian ảnh gốc, số lượt inference và chất lượng. Không kết luận lợi ích do ROI nếu toàn ảnh độ phân giải cao đạt tương đương hoặc tốt hơn.

## Chọn cấu hình

- HOG: F1 có phạt unknown, coverage, MAE đếm trống thấp hơn; hòa tiếp chọn C nhỏ hơn rồi ngưỡng gần 0 hơn, còn hòa giữ thứ tự grid.
- Detector: tối đa mean F1 của ba nhánh YOLO, rồi coverage cao hơn, MAE đếm trống thấp hơn, ngân sách hybrid thấp hơn, cuối cùng thứ tự grid.
- Cùng một confidence hậu xử lý và một quy tắc ghép áp dụng cho full-frame, hybrid và đối chứng độ phân giải cao. Không chọn riêng quy tắc có lợi cho hybrid.
- So với cấu hình mặc định đã đăng ký (padding32, overlap0,20, confidence0,25, quy tắc ghép mặc định), mỗi tổ hợp có delta_mean_f1_from_default, improved_over_default và selected. Cấu hình thua/không cải thiện vẫn được lưu.
- Chọn tham số trên validation làm kết quả validation có thiên lệch chọn mô hình; đó không phải kết quả test độc lập.

## Timing và cache

Sau tìm kiếm, runner chạy lại bốn phương pháp đã chọn trên toàn validation bằng FrameTimer. Đây là một lượt pipeline thực trên từng ảnh gốc: đọc ảnh, hình học, model/tile, mapping/NMS, ghép ô và policy. Thời gian khởi tạo/warm-up, ghi file và audit XML không tính. Ultralytics preprocess/NMS nội bộ và mapping hybrid nằm trong khối model của runner; tổng thời gian bao phủ toàn pipeline. Không cộng các thời gian cache rời rạc để giả làm thời gian end-to-end.

Số lượt và pixel được báo riêng. Thiết bị và phiên bản thực tế được lưu trong plan/lock; PyTorch dùng 2 thread, OpenCV 1 thread, BLAS khi fit SVM giới hạn 2, còn BLAS khi dự đoán HOG không giới hạn riêng; không so trực tiếp số milliseconds CPU local với GPU Colab.

Cache gồm features, models ứng viên, geometry, detection theo frame và journal tìm kiếm. Chỉ dùng lại khi plan/split/source engine/môi trường khớp; sai thì phải tạo output directory mới. Nếu gián đoạn khi đang trích feature chưa xong split, phần feature đó cần trích lại. Các feature/detection cache lớn mặc định giữ local trong notebook, không đồng bộ hàng GB/hàng nghìn file lên Drive; model, tham chiếu và báo cáo vẫn đồng bộ. Colab xóa runtime trước khi hoàn tất có thể làm mất cache local. Muốn lưu mọi cache, dùng CLI/output trên ổ lưu trữ do bạn quản lý, cân nhắc chi phí I/O.

## Chạy local và Colab

Cài PyTorch phù hợp phần cứng trước, sau đó:

```bash
python -m pip install -e '.[validation]'
python parking_module.py validate \
  --dataset-root data/raw/PKLot \
  --manifest-dir data/manifests/pklot_v1/within_camera \
  --plan configs/validation_v1.json \
  --checkpoint models/yolo11n.pt \
  --output outputs/validation_v1 --device cpu
```

Checkpoint chính thức và SHA256 được chốt trong plan. Không commit checkpoint hoặc feature matrix. Muốn smoke-test có thể dùng --max-train-frames 24 --max-val-frames 3 vào **thư mục khác**; run thiếu dữ liệu không được tạo khóa test.

Trên Colab, mở notebook mới, giữ VALIDATION_ONLY_MODE=True, chạy từ môi trường/dữ liệu đến 6.2. Runner không phụ thuộc vào các lượt demo nặng trước đó. Giữ RUN_FINAL_TEST=False ở giai đoạn task 12. Cần đúng Ultralytics 8.4.0; nếu đã import một bản khác trong kernel, restart runtime sau khi cài đúng bản rồi chạy lại các cell cần thiết.

## Đầu ra và khóa

- preregistered_plan.json: dải tìm kiếm, nguồn split, engine và môi trường trước khi chấm điểm.
- hog_search.json: mọi C/ngưỡng SVM, kể cả không cải thiện.
- detector_search.json: toàn bộ 144 tổ hợp và số liệu ba nhánh, delta so với mặc định.
- tile_plans.json: số tile, overlap hiệu dụng, bounds, fallback theo layout/config.
- selected/<method>/: predictions đã ghép khóa, per-frame timing, pixel budget và báo cáo tổng/theo camera/thời tiết.
- comparison.csv, validation_summary.json: so sánh bốn phương pháp sau lựa chọn.
- locked_config.json: chỉ được tạo khi chạy đủ toàn bộ train và validation, benchmark đầy đủ và pixel đối chứng khớp.

Khóa giữ checkpoint SHA256, HOG model SHA256, train/val/test manifest và layout/lock SHA256, tham chiếu train, source fingerprint, phiên bản, ngưỡng, tiling và quy tắc. verify_validation_lock kiểm tra lại artifact trước test. run_locked_test dùng đúng khóa và không có bước chọn tham số. Notebook chỉ gọi test khi RUN_FINAL_TEST=True; task 12 không chạy test.

Kết quả nhẹ dùng chia sẻ/review sẽ nằm trong reports/validation_v1/ sau khi lượt đầy đủ hoàn tất. Bằng chứng chi tiết và cache ở outputs/validation_v1/; dữ liệu test chưa được đánh giá trong quá trình tìm kiếm này.

## Lịch sử test của notebook cũ

Commit main trước lần review này (293893d) đã lưu output của final-test-code (execution_count 25) và results-code (26), trên 270 ảnh test của cấu hình cũ. Lượt tìm kiếm mới đã chốt plan trước khi kiểm tra các output cũ và không dùng điểm test trong thuật toán chọn tham số, nhưng không được gọi test v1 là holdout chưa từng được quan sát. Giữ riêng bằng chứng cũ và báo rõ lịch sử tái sử dụng test khi viết báo cáo; không âm thầm chia lại test sau khi xem kết quả. Nếu cần một đánh giá blind mới, nhóm phải chốt giao thức/nhóm ngày độc lập trước lượt đánh giá đó.
