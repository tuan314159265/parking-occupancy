# Nhận diện trạng thái ô đỗ xe — PKLot

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tuan314159265/parking-occupancy/blob/main/notebooks/01_pklot_run_all.ipynb)

Xác định từng ô đỗ trong ảnh camera cố định là **trống / có xe / unknown**, so sánh bốn phương pháp trên cùng split đã khóa của PKLot:

| Phương pháp | Mô tả ngắn |
|---|---|
| HOG + SVM | Đặc trưng HOG trên từng ô (polygon) + Linear SVM |
| YOLO full 640 | YOLO11n trên toàn ảnh, ảnh resize 640 |
| Hybrid | Canny/Hough kiểm tra hình học → ROI/tile → YOLO11n trên tile → ghép xe–ô |
| YOLO full pixel-matched | YOLO11n toàn ảnh với số pixel đầu vào bằng đúng hybrid (đối chứng) |

- **Repository (public):** https://github.com/tuan314159265/parking-occupancy
- **Notebook Colab (Run All):** https://colab.research.google.com/github/tuan314159265/parking-occupancy/blob/main/notebooks/01_pklot_run_all.ipynb

## Kết quả validation (đã khóa)

Giao thức `within_camera`, 270 ảnh validation (15.120 ô), CPU. Đây là số liệu **dùng để chọn tham số**, không phải test độc lập. Chi tiết, toàn bộ 153 cấu hình đã thử và biểu đồ: [reports/validation_v1](reports/validation_v1/README.md).

| Phương pháp | F1 | Accuracy | Coverage | MAE số ô trống | ms/ảnh (CPU) |
|---|---:|---:|---:|---:|---:|
| HOG + SVM | 0.9888 | 0.9415 | 0.9538 | 3.093 | 267.4 |
| YOLO full 640 | 0.5691 | 0.5855 | 0.8713 | 17.970 | 242.5 |
| Hybrid 640 | 0.7439 | 0.6999 | 0.8920 | 12.674 | 504.4 |
| YOLO full pixel-matched | 0.7418 | 0.6978 | 0.8819 | 12.311 | 430.8 |

Accuracy tính *unknown* là sai; coverage là tỉ lệ ô có dự đoán. Công thức: [docs/evaluation.md](docs/evaluation.md).

## Kết quả test cuối (đã hoàn tất)

Task 13 đã hoàn tất ở lượt ngày **11/11/2026**: cùng 270 ảnh / 15.120 ô cho cả bốn phương pháp, cấu hình chọn trên validation đã khóa, CPU. Bộ bằng chứng local được tìm lại và xác minh ngày 16/11/2026; không chạy lại test để công bố bảng này.

| Phương pháp | F1 test | Accuracy | Coverage | MAE ô trống | ms/ảnh CPU |
|---|---:|---:|---:|---:|---:|
| HOG + SVM | 0.9654 | 0.9126 | 0.9353 | 4.681 | 167.8 |
| YOLO full 640 | 0.3679 | 0.6767 | 0.9015 | 15.807 | 165.3 |
| Hybrid 640 | 0.6517 | 0.7583 | 0.9026 | 11.400 | 386.5 |
| YOLO full pixel-matched | 0.6249 | 0.7487 | 0.8955 | 11.563 | 377.9 |

**Dùng cho task 17–18:** [báo cáo test và ảnh minh họa](reports/final_test_v1/README.md), [bảng tổng](reports/final_test_v1/tables/comparison.csv), [theo camera](reports/final_test_v1/tables/by_camera.csv), [theo thời tiết](reports/final_test_v1/tables/by_weather.csv), [timing từng giai đoạn](reports/final_test_v1/tables/timings_per_frame.csv). Thư mục báo cáo giữ đủ dự đoán, 88 ảnh minh họa, metadata và checksum.

Đã xác minh đủ bốn phương pháp và metrics tính lại khớp dự đoán/timing; 127 file nguồn khớp manifest gốc. Test có lịch sử đã được quan sát ở notebook cũ, nên không gọi đây là blind holdout chưa từng quan sát. Các thời gian là CPU, không so trực tiếp GPU Colab.

## Cấu trúc repository

```
parking_module.py          Toàn bộ mã xử lý (loader/audit, split, HOG, YOLO, hình học, tile, ghép, đánh giá, CLI)
final_test_module.py       Các bước test cuối: kiểm tra khóa → chạy → bảng → ảnh minh họa → hash bằng chứng
notebooks/01_pklot_run_all.ipynb   Notebook chính, chạy Run All trên Colab hoặc local
scripts/run_final_test.py         CLI chạy test cuối (giống notebook mục 6.3)
scripts/build_final_test_bundle.py  Đóng gói model/khóa để chuyển sang máy khác (không chứa dataset)
scripts/export_validation_report.py Xuất báo cáo nhẹ vào reports/ sau một lượt validation đầy đủ
configs/                   pklot_v1.json (split), validation_v1.json (plan tìm kiếm + checkpoint), default.json
data/manifests/pklot_v1/   Split train/val/test và polygon ô đỗ đã khóa (có SHA256 trong split_lock.json)
reports/validation_v1/     Kết quả validation, plan, khóa locked_config.json, hash
reports/final_test_v1/     Test cuối: bảng, dự đoán, timing, 88 ảnh minh họa và hash
src/parking_occupancy/     Lớp tương thích cho `pip install -e .` (chuyển tiếp tới parking_module)
tests/                     136 unit test (không cần dataset, model hay mạng)
docs/                      Giao thức dữ liệu, đánh giá, validation, pipeline, giao diện, nhật ký bàn giao
```

**Không có trong Git** (được `.gitignore` chặn): archive/ảnh/XML PKLot (`data/raw/`), checkpoint (`models/`, `*.pt`), model đã huấn luyện (`*.joblib`), cache/feature (`*.npy`), mọi kết quả chạy (`outputs/`) và gói bàn giao (`*.zip`). Repo không chứa thông tin đăng nhập hay đường dẫn cá nhân.

## 1. Cài đặt

Yêu cầu Python ≥ 3.10 (validation gốc chạy Python 3.14, CPU). Khoảng 10–15 GB trống cho dữ liệu.

```bash
git clone https://github.com/tuan314159265/parking-occupancy.git
cd parking-occupancy
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

[`requirements.txt`](requirements.txt) ghim **đúng** các phiên bản đã ghi trong khóa (numpy 2.5.4, opencv 4.14.0, torch 2.15.0, ultralytics 8.4.0, scikit-image 0.26.0, scikit-learn 1.9.1). Bước test cuối từ chối chạy nếu các thư viện này khác phiên bản. Chỉ cần đọc dữ liệu/audit thì `pip install -e .` (Pillow, numpy, OpenCV) là đủ.

**Windows:** đặt `PYTHONUTF8=1` trước khi chạy (PowerShell: `$env:PYTHONUTF8=1`) để console in được tiếng Việt và mọi file đọc/ghi dùng UTF-8.

## 2. Dữ liệu PKLot

| | |
|---|---|
| Trang chính thức | https://web.inf.ufpr.br/vri/databases/parking-lot-database/ |
| Archive | https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz — 4.898.276.304 byte (kiểm tra lại ngày 15/11/2026) |
| Tài liệu | https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf |
| Giấy phép | CC BY 4.0 — phải trích dẫn Almeida et al. (2015), xem cuối trang |

```bash
python parking_module.py prepare                 # tải tiếp (resume) bằng curl, giải nén ảnh toàn cảnh + XML, audit
python parking_module.py prepare --no-download   # archive đã có trong data/raw/
python parking_module.py audit --root data/raw/PKLot --output outputs/full   # dataset đã giải nén sẵn
```

Kết quả: `data/raw/PKLot/{PUCPR,UFPR04,UFPR05}/...` và `outputs/full/{valid_samples.jsonl, issues.jsonl, audit_summary.json, provenance.json}`. Chỉ giải nén ảnh toàn cảnh và XML, không lấy `PKLotSegmented`. SHA256 tính sau khi tải là dấu vết local, nhà phát hành không công bố checksum.

**Không cần chia lại dữ liệu.** Split đã khóa nằm trong `data/manifests/pklot_v1/` (chia theo ngày, seed 42):

| Giao thức | Train | Val | Test |
|---|---:|---:|---:|
| `within_camera` (PUCPR + UFPR04 + UFPR05) | 1.182 | 270 | 270 |
| `transfer_ufpr04_to_ufpr05` | 420 | 90 | 90 |

Nếu chạy lại `python parking_module.py splits --manifest outputs/full/valid_samples.jsonl`, lệnh chỉ xác minh split hiện có với audit của bạn và từ chối ghi đè. Chi tiết: [docs/dataset-protocol.md](docs/dataset-protocol.md).

## 3. Checkpoint YOLO11n

Checkpoint COCO gốc của Ultralytics, không fine-tune; chỉ giữ các lớp `car`, `bus`, `truck`.

| | |
|---|---|
| URL | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt |
| SHA256 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Kích thước | 5.613.764 byte |

```bash
mkdir -p models
curl -L -o models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
sha256sum models/yolo11n.pt          # Windows: certutil -hashfile models\yolo11n.pt SHA256
```

URL và SHA256 đã chốt trong `configs/validation_v1.json`; runner từ chối checkpoint có hash khác.

## 4. Lệnh chạy

```bash
# Kiểm thử (không cần dataset/model/mạng), gồm kiểm tra mã khóa còn khớp
python -m unittest discover -s tests -v

# Validation: tìm tham số chỉ trên train/val rồi tạo locked_config.json
python parking_module.py validate --dataset-root data/raw/PKLot \
  --manifest-dir data/manifests/pklot_v1/within_camera --plan configs/validation_v1.json \
  --checkpoint models/yolo11n.pt --output outputs/validation_v1 --device cpu

# Test cuối với cấu hình đã khóa (kiểm tra trước, rồi chạy vào thư mục MỚI)
python scripts/run_final_test.py --check-only
python scripts/run_final_test.py --output outputs/locked_test_v1 --device cpu

# Đánh giá một file dự đoán bất kỳ bằng evaluator chung
python parking_module.py evaluate --help
```

Validation đầy đủ trên CPU mất nhiều giờ; smoke-test dùng `--max-train-frames 24 --max-val-frames 3` và một `--output` **khác** (lượt thiếu dữ liệu không tạo khóa). Xem [docs/validation.md](docs/validation.md).

### Test cuối và gói khóa

Test cuối cần thêm hai model đã huấn luyện trên train ở lượt validation khóa: `outputs/validation_v1/hog_svm.joblib` và `outputs/validation_v1/geometry_references.joblib`. Chúng không nằm trong Git; SHA256 của chúng được ghi trong [`locked_config.json`](reports/validation_v1/locked_config.json). Có hai cách:

1. **Dùng gói khóa** `parking-final-test-bundle.zip` (model HOG, tham chiếu hình học, khóa, plan, checkpoint, manifest, mã; không chứa ảnh PKLot). Giải nén vào thư mục gốc repo để các đường dẫn mặc định ở trên khớp, hoặc trên Colab điền `LOCKED_BUNDLE_ZIP`. Người giữ lượt validation tạo gói bằng `python scripts/build_final_test_bundle.py`.
   Tải: https://github.com/tuan314159265/parking-occupancy/releases/download/v1.0-final/parking-final-test-bundle.zip
2. **Chạy lại validation** theo lệnh trên. Lượt mới tạo khóa mới; báo cáo số liệu theo khóa đó.

`run_final_test.py` kiểm tra lần lượt: khóa đầy đủ và được chọn trên val, SHA256 của checkpoint/manifest/model HOG/tham chiếu, mã nguồn engine, phiên bản thư viện, toàn bộ 270 ảnh test. Chỉ khi tất cả khớp mới suy luận. Kết quả ở `outputs/locked_test_v1/`: `tables/comparison.csv`, bảng theo camera/thời tiết, ma trận nhầm lẫn, timing, ảnh đúng/sai/unknown và `evidence_sha256.json`. Lệnh không ghi đè thư mục đã có kết quả.

## 5. Chạy trên Google Colab

1. Mở [notebook trên Colab](https://colab.research.google.com/github/tuan314159265/parking-occupancy/blob/main/notebooks/01_pklot_run_all.ipynb). Chọn *Runtime → Change runtime type → T4 GPU* (CPU cũng chạy được nhưng chậm).
2. Cell cấu hình mục 2 (mặc định đã đúng cho repo này):
   - `REF = "main"`. Để tái lập chính xác bản nộp, dùng tag hoặc commit SHA của bản cuối.
   - `USE_GOOGLE_DRIVE`: `False` (mặc định) thì dữ liệu chỉ nằm trong runtime và tải lại mỗi phiên; `True` để lưu dataset, audit và kết quả tại `MyDrive/parking-occupancy/` và dùng lại cho các lần sau.
   - `SHARED_DATASET_DIR`: điền nếu đã có PKLot giải nén trên Drive (thư mục chứa UFPR04/UFPR05/PUCPR). Để trống thì notebook tự tải archive 4,9 GB.
   - `RUN_VALIDATION_SEARCH=False`, `RUN_FINAL_TEST=False`: Run All chỉ đọc lại kết quả và khóa có sẵn, không tự chạy test.
3. *Runtime → Run all*. Khi Colab hỏi quyền Drive, chọn **tất cả** quyền.
4. Chạy test thật: điền `LOCKED_BUNDLE_ZIP`, đặt `RUN_FINAL_TEST=True`, chạy lại từ mục 6.3. Lưu ý: test chỉ chạy khi phiên bản torch/numpy/opencv… trên runtime khớp khóa; nếu Colab cài sẵn bản khác, cài theo `requirements.txt` rồi *Restart runtime*.

Notebook không nhúng mã xử lý: cell mục 2 tải `parking_module.py` từ GitHub theo `OWNER/REPO/REF` và ghi URL, REF và SHA256 vào `module_source.json`. Mục 3.2 tải manifest đã khóa và kiểm tra checksum. Xử lý lỗi `mount failed` / `credential propagation was unsuccessful`: chạy lại cell mount, cấp đủ quyền; nếu vẫn lỗi, *Runtime → Disconnect and delete runtime* rồi Run All (xem [googlecolab/colabtools#6089](https://github.com/googlecolab/colabtools/issues/6089)).

## 6. Tính toàn vẹn của bản khóa

- `locked_config.json` băm mã nguồn 27 hàm suy luận/đánh giá trong `parking_module.py` (`engine_sha256`). Sửa bất kỳ hàm nào trong số đó, kể cả định dạng, sẽ làm test cuối bị từ chối. `tests/test_locked_release.py` phát hiện điều này ngay khi chạy unit test.
- Không chọn tham số, ngưỡng hay model trên test. Mọi cấu hình đã thử (kể cả không cải thiện) đều được lưu trong `reports/validation_v1/`.
- **Lịch sử test:** notebook cũ (commit `293893d`) từng in output trên 270 ảnh test với cấu hình trước đó. Lượt validation mới đã chốt plan trước khi xem lại và không dùng điểm test, nhưng test v1 **không** phải holdout chưa từng quan sát. Báo cáo cần nêu rõ điều này.
- Giao thức `transfer_ufpr04_to_ufpr05` đã có split nhưng chưa chạy validation/test mới.
- Timing đo một lượt trên CPU local sau warm-up; không so trực tiếp với GPU Colab.

## Tài liệu chi tiết

- [docs/introduction-and-data.md](docs/introduction-and-data.md): bài toán, PKLot, kiểm tra chất lượng dữ liệu
- [docs/dataset-protocol.md](docs/dataset-protocol.md): quy tắc chia theo ngày, ngân sách, giao thức chuyển camera
- [docs/interfaces.md](docs/interfaces.md): định dạng Frame/ParkingSpace/VehicleDetection, hệ tọa độ
- [docs/pipeline-integration.md](docs/pipeline-integration.md): pipeline hình học → ROI/tile → YOLO → ghép xe–ô
- [docs/evaluation.md](docs/evaluation.md): công thức chỉ số, xử lý unknown, schema đầu ra
- [docs/validation.md](docs/validation.md): dải tìm kiếm, đối chứng pixel, cache, khóa
- [docs/task-status.md](docs/task-status.md): nhật ký bàn giao theo task

## Nguồn và trích dẫn

- P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., A. L. Koerich. *PKLot – A robust dataset for parking lot classification*. Expert Systems with Applications 42(11):4937–4949, 2015. Dữ liệu theo CC BY 4.0.
- Ultralytics YOLO11 (AGPL-3.0): https://github.com/ultralytics/ultralytics. Checkpoint `yolo11n.pt` từ https://github.com/ultralytics/assets/releases/tag/v8.3.0.
