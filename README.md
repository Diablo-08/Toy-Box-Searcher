# 🧸 Toy Box Searcher - Nhận diện hộp đồ chơi & Tra cứu thông tin

Hệ thống thị giác máy tính và ứng dụng web xây dựng bằng Python 3.11.9 (CPU-only) giúp:
1. Chuẩn hóa và tiền xử lý hình ảnh hộp đồ chơi (khử nhiễu Bilateral Filter, cân bằng độ tương phản thích nghi CLAHE trên kênh L của LAB, làm nét Unsharp Masking).
2. Nhận diện chữ quang học đa hướng bằng **PaddleOCR 3.x**.
3. Bóc tách thông minh: Lọc từ rác/cảnh báo, so khớp mờ thương hiệu (**RapidFuzz**), trích xuất mã sản phẩm/SKU/tỉ lệ bằng Regex tối ưu.
4. Tự động tìm kiếm thông tin chi tiết trên Internet qua gói **ddgs** (hoàn toàn miễn phí, không cần API key).

---

## 🚀 Hướng dẫn cài đặt & Chạy trên Windows (Python 3.11.9)

Mở **PowerShell** hoặc **Command Prompt** tại thư mục dự án và thực hiện tuần tự:

```bash
# 1. Kiểm tra phiên bản Python hệ thống (Bắt buộc 3.11.9)
py -3.11 --version

# 2. Tạo môi trường ảo venv độc lập
py -3.11 -m venv venv

# 3. Kích hoạt môi trường ảo
venv\Scripts\activate

# 4. Xác nhận môi trường ảo đang sử dụng đúng Python 3.11.9
python --version

# 5. Nâng cấp pip lên bản mới nhất
python -m pip install --upgrade pip

# 6. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 7. Khởi chạy ứng dụng Web Streamlit
python -m streamlit run app.py
```

> **Lưu ý quan trọng cho lần chạy đầu tiên**:
> Trong lần quét đầu tiên, **PaddleOCR 3.x** sẽ tự động tải các tệp trọng số mô hình (`PP-LCNet_x1_0_textline_ori`, `PP-OCRv6_medium_det`, `PP-OCRv6_medium_rec`) từ máy chủ chính thức về thư mục `~/.paddlex/official_models/`. Quá trình này diễn ra một lần duy nhất và cần kết nối Internet ổn định trong 1-2 phút đầu. Các lần khởi động sau sẽ nạp trực tiếp từ bộ nhớ đệm cục bộ cực kỳ nhanh chóng.

---

## 🧪 3 Test Case Thủ Công & Kết Quả Mong Đợi

| Test Case | Hình ảnh đầu vào | Đặc điểm chữ trên vỏ hộp | Kết quả `final_keyword` mong đợi |
| :--- | :--- | :--- | :--- |
| **Test Case 1** | **Hộp LEGO City Express Passenger Train** | Chữ `LEGO` góc trên, số hiệu `60337`, dòng chữ `CITY`, chữ cảnh báo `WARNING: CHOKING HAZARD`, `7+` | `LEGO 60337 CITY PASSENGER TRAIN` |
| **Test Case 2** | **Vỉ xe Hot Wheels Premium / Mainline** | Logo `Hot Wheels` uốn lượn, mã sản phẩm `HW123` hoặc `B-1234`, dòng chữ `FAST & FURIOUS`, tỉ lệ `1:64` | `Hot Wheels 1:64 HW123 FAST & FURIOUS` |
| **Test Case 3** | **Hộp Gunpla Bandai Gundam** | Logo `BANDAI`, chữ `GUNDAM`, cấp độ và tỉ lệ `HG 1/144`, mã người máy `RX-78-2`, chữ tiếng Nhật | `Bandai Gundam HG1/144 RX-78-2` |
