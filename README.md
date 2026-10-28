# Parking occupancy — PKLot


## Chạy local

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m parking_occupancy.prepare
```

Lệnh prepare tải tiếp bằng curl, yêu cầu khoảng 10–15 GB trống; chỉ giải nén ảnh toàn cảnh và XML (không lấy PKLotSegmented), sau đó kiểm tra tất cả ảnh và nhãn. Có thể chạy lại; không chạy đồng thời hai downloader vào cùng archive. Archive chính thức có 4.898.276.304 byte khi kiểm tra ngày 28/10/2026. SHA256 tính sau tải dùng làm dấu vết local, chưa phải xác thực bằng checksum của nhà phát hành.

Nếu archive đã tải xong:

```bash
python -m parking_occupancy.prepare --no-download
```

Nếu đã giải nén sẵn, trỏ đúng thư mục chứa UFPR04, UFPR05, PUC:

```bash
parking-audit --root data/raw/PKLot --output outputs/full
```

## Colab / Run All — cách dùng chính

1. Push file **`parking_module.py`** ở thư mục gốc lên GitHub public. Đây là file duy nhất chứa mã xử lý và chỉ cần Pillow; không cần push data, outputs hoặc cài package repo trên Colab.
2. Upload **`notebooks/01_pklot_run_all.ipynb`** lên Colab.
3. Notebook đã cấu hình `OWNER=tuan314159265`, `REPO=parking-occupancy`, `REF=main`; sửa trong cell đầu nếu dùng repo khác. REF là nhánh thực tế (`main` hoặc `master`), tag hoặc commit SHA. Mặc định module nằm ở gốc repo.
4. Chọn Run All. Notebook cài Pillow, tải module từ raw.githubusercontent.com, import và gọi `parking.prepare(...)`, rồi đọc một frame hợp lệ. Lần Run All sau tải lại đúng REF cấu hình.

Notebook lưu URL, REF và SHA256 module vào `module_source.json` để ghi nhận mã đã chạy. Dùng commit SHA nếu cần tái lập chính xác. Repo private chưa được hỗ trợ trong notebook này.

Có thể dùng module độc lập ở local:

```bash
python -m pip install 'Pillow>=10,<13'
python parking_module.py prepare
python parking_module.py audit --root data/raw/PKLot --output outputs/full
```

Không cần clone hay upload toàn bộ repository lên Colab. Notebook mặc định mount Google Drive của bạn và lưu archive, ảnh/XML cùng kết quả tại `MyDrive/parking-occupancy/`. Lần sau dùng lại dữ liệu đã giải nén thành công rồi chạy lại audit; không tải/giải nén lại. Đặt `USE_GOOGLE_DRIVE=False` nếu chỉ muốn dùng bộ nhớ tạm Colab. Repo GitHub: https://github.com/tuan314159265/parking-occupancy. Notebook đã điền sẵn tài khoản, tên repo và nhánh main. Notebook chưa chạy trên dịch vụ Colab thực tế.

## Cấu trúc

- `data/raw/`: archive, ảnh toàn cảnh, XML; không commit dữ liệu lớn.
- `configs/default.json`: cấu hình và quy ước mặc định để các thành viên tham khảo. CLI hiện nhận root/output/seed trực tiếp.
- `parking_module.py`: toàn bộ contracts, loader/audit, tải/giải nén và metrics trong một file độc lập.
- `src/parking_occupancy/`: các đường import tương thích cho CLI/package local, dùng chung mã từ module gốc.
- `outputs/full/`: valid_samples.jsonl, issues.jsonl, audit_summary.json, provenance.json.
- `outputs/partial/`: nếu có, chỉ kiểm tra phần archive đã tải; không đại diện toàn bộ PKLot.
- `docs/interfaces.md`: giao diện dữ liệu, hình học, detector, đánh giá đã chốt ở mức v0.1.
- `report/main.tex`: khung báo cáo tiếng Việt; biên dịch `pdflatex main.tex` từ thư mục report, cần TeX Live với hỗ trợ tiếng Việt.

Manifest JSONL có một dòng/frame, giữ toàn bộ polygon, nhãn và metadata; rejected frame không nằm trong danh sách hợp lệ. issues ghi ảnh thiếu/XML thiếu, lỗi giải mã, XML lỗi, nhãn ngoài 0/1, trùng id, contour không hợp lệ. Audit không tự sửa hay suy đoán nhãn. Nên chỉ chạy audit trên full-frame root để tránh coi ảnh crop là orphan.

Chia train/val/test theo ngày trên tất cả camera; dùng seed 42. Chi tiết và giới hạn trong `docs/interfaces.md`.

## Nguồn và trích dẫn

- PKLot: https://web.inf.ufpr.br/vri/databases/parking-lot-database/
- Archive: https://www.inf.ufpr.br/vri/databases/PKLot.tar.gz
- Tài liệu: https://www.inf.ufpr.br/lesoliveira/download/pklot-readme.pdf
- Almeida et al., *PKLot – A robust dataset for parking lot classification*, Expert Systems with Applications 42(11):4937–4949, 2015. Trang hiện tại công bố CC BY 4.0; giữ trích dẫn khi sử dụng.
- CNRPark-EXT: https://cnrpark.it/ — nguồn tham khảo giai đoạn sau; chưa tải và chưa triển khai loader trong phạm vi hai đầu việc hiện tại.

Trạng thái bàn giao và các bước chưa xác minh: `docs/task-status.md`. Phiên tải đang chạy ghi tiến độ vào `outputs/pipeline_status.json`; khi đủ archive, tiến trình chờ tự chạy audit đầy đủ.

## Nơi lưu dữ liệu

- GitHub: module Python, notebook, cấu hình, tài liệu và tests. Không push archive hoặc ảnh/XML dataset; `.gitignore` đã loại `data/raw/` và kết quả `outputs/`.
- Local: bản tải hiện tại và ảnh/XML sau giải nén ở `data/raw/`, dùng chạy trên máy này. Không tự đồng bộ lên Drive.
- Colab: notebook mount Drive của tài khoản bạn chọn, lưu tại `MyDrive/parking-occupancy/data/raw/`; manifest và thống kê tại `MyDrive/parking-occupancy/outputs/full/`.

Bạn có thể tự upload archive local lên đúng thư mục Drive để Colab tải tiếp, hoặc upload archive hoàn chỉnh để Colab chỉ giải nén. Hai môi trường dùng bản dữ liệu riêng nếu không chủ động chuyển file.

[Mở notebook trên Colab](https://colab.research.google.com/github/tuan314159265/parking-occupancy/blob/main/notebooks/01_pklot_run_all.ipynb).
