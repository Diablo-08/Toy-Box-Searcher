# 🧸 Toy Box Searcher - Nhận diện hộp đồ chơi & Tra cứu thông tin
👉 **[Truy cập ứng dụng trực tuyến tại đây](https://toy-box-searcher.streamlit.app/)**

Hệ thống thông minh nhận diện chữ trên bao bì đồ chơi (OCR), trích xuất thương hiệu & mã SKU, và tra cứu sản phẩm theo khu vực địa lý — chạy hoàn toàn **offline** trên CPU, không cần GPU, không cần API key.

---

## 📋 Yêu cầu hệ thống

| Thành phần | Yêu cầu tối thiểu |
|---|---|
| Hệ điều hành | Windows 10/11 (64-bit) |
| Python | 3.11.9 (khuyên dùng, **không dùng 3.12+**) |
| RAM | 4 GB trở lên (8 GB khuyên dùng) |
| CPU | Bất kỳ (Intel/AMD x86-64) |
| GPU | **Không cần** (CPU-only) |
| Kết nối mạng | Cần để tìm kiếm sản phẩm (DuckDuckGo) |

---

## 🚀 Cài đặt & Chạy ứng dụng (Windows)

### Bước 1 — Tạo môi trường ảo Python

```bash
python -m venv venv
venv\Scripts\activate
```

### Bước 2 — Cài đặt toàn bộ thư viện

```bash
pip install -r requirements.txt
```

> ⚠️ **Lưu ý:** `paddlepaddle` và `paddleocr` có kích thước lớn (~1.5 GB). Lần đầu tải về mô hình có thể mất 5–15 phút tùy tốc độ mạng. Các lần chạy sau sẽ dùng bộ nhớ cache cục bộ.

### Bước 3 — Chạy ứng dụng

```bash
streamlit run app.py
```

Mở trình duyệt tại địa chỉ: **http://localhost:8501**

---

## 📁 Cấu trúc thư mục

```
toy_box_searcher/
│
├── app.py               # Giao diện web Streamlit (UI chính)
├── ocr_extractor.py     # Nhận diện chữ PaddleOCR + trích xuất từ khóa
├── preprocessing.py     # Tiền xử lý ảnh (CLAHE, Unsharp Masking)
├── search_engine.py     # Tìm kiếm DuckDuckGo + lọc Wikipedia
├── requirements.txt     # Danh sách thư viện phụ thuộc
└── README.md            # Tài liệu hướng dẫn (file này)
```

---

## 🔄 Luồng xử lý (Pipeline)

```
📷 Upload ảnh (JPG/PNG)
        │
        ▼
🖼️  preprocessing.py
   ├── normalize_image()   → Sửa EXIF, chuyển RGB→BGR, resize về 960px
   └── preprocess_image()  → Bilateral Filter + CLAHE + Unsharp Masking
        │
        ▼
🔍  ocr_extractor.py
   ├── run_ocr()           → PaddleOCR 3.x predict() trên ảnh tăng cường
   ├── Fallback OCR        → Nếu ảnh tăng cường không rõ → quét thêm ảnh gốc
   ├── normalize_text()    → Làm sạch ký hiệu ©®™, khoảng trắng thừa
   ├── is_noise_text()     → Lọc cảnh báo an toàn, từ rác
   ├── match_brand()       → Fuzzy matching thương hiệu (RapidFuzz)
   ├── extract_sku_or_model() → Regex trích xuất mã SKU, số bộ, tỉ lệ
   └── extract_text_and_keywords() → Ghép từ khóa ≤8 từ / ≤80 ký tự
        │
        ▼
🌐  search_engine.py
   ├── extract_smart_query()   → Chấm điểm & lọc từ khóa thông minh
   ├── ThreadPoolExecutor      → Song song truy vấn vn-vi + wt-wt
   ├── is_wikipedia_entry()    → Lọc 100% Wikipedia/Wikimedia
   └── search_toy_info()       → 3 tầng fallback, cache 1 giờ
        │
        ▼
🖥️  app.py (Streamlit UI)
   ├── Thanh tiến trình động (tự biến mất khi xong)
   ├── Bảng OCR kèm màu trạng thái độ tin cậy
   ├── Liên kết nhanh: Shopee / MyKingdom / Google / Amazon / YouTube
   └── Thẻ kết quả tìm kiếm với hover effect
```

---

## 🛠️ Công cụ & Thư viện sử dụng

### 1. 🖥️ Streamlit — Giao diện Web App

**Phiên bản:** `>= 1.30.0`

Streamlit là framework Python để xây dựng ứng dụng web dữ liệu chỉ bằng code Python thuần túy, không cần HTML/CSS/JavaScript.

| | Chi tiết |
|---|---|
| **Vai trò** | Toàn bộ giao diện người dùng: upload ảnh, thanh tiến trình, bảng kết quả OCR, thẻ tìm kiếm |
| **Tính năng dùng** | `st.file_uploader`, `st.session_state`, `st.cache_resource`, `st.cache_data`, `st.empty()` (placeholder biến mất), `st.radio` với `format_func` và `key` |

**✅ Ưu điểm:**
- Triển khai cực nhanh — toàn bộ UI chỉ cần ~550 dòng Python
- Tích hợp sẵn `session_state` để quản lý trạng thái phức tạp (ảnh, từ khóa, kết quả)
- `@st.cache_resource` giúp load mô hình OCR **đúng 1 lần** dù server chạy lâu dài
- `@st.cache_data(ttl=3600)` cache kết quả tìm kiếm 1 giờ, tránh gọi API lặp lại
- Tự động hot-reload khi sửa code

**❌ Nhược điểm:**
- Mỗi thao tác người dùng đều khiến toàn bộ script Python chạy lại từ đầu (re-run)
- Không phù hợp cho ứng dụng cần trạng thái realtime phức tạp (chat, dashboard live)
- CSS tùy chỉnh phải dùng `unsafe_allow_html=True` — không có component UI native đủ mạnh

---

### 2. 🧠 PaddleOCR 3.x + PaddlePaddle — Nhận diện chữ (OCR)

**Phiên bản:** `paddleocr >= 3.0.0`, `paddlepaddle >= 3.0.0`

Thư viện OCR mã nguồn mở của Baidu, sử dụng mạng nơ-ron sâu (Deep Learning) để nhận diện văn bản trong hình ảnh.

#### So sánh hai mô hình được tích hợp

| Tiêu chí | ⚡ PP-OCRv5 Mobile | 🎯 PP-OCRv6 Medium |
|---|---|---|
| Mô hình phát hiện | PP-OCRv5_mobile_det | PP-OCRv6_medium_det |
| Mô hình nhận diện | PP-OCRv5_mobile_rec | PP-OCRv6_medium_rec |
| Giới hạn cạnh ảnh | 736px | 960px |
| Thời gian/ảnh (CPU) | **~0.6 – 1.3s** | ~3.5 – 5.0s |
| Độ chính xác | ≥ 95% (bao bì đồ chơi) | ≥ 97% |
| Ngưỡng từ khóa | **85%** | **95%** |
| RAM cần thiết | ~400 MB | ~700 MB |
| Khuyên dùng | ✅ Mặc định | Khi cần độ chính xác tối đa |

**Kỹ thuật tích hợp đặc biệt:**
- **Monkey-patch PaddlePaddle 3.x Windows bug:** Vô hiệu hóa MKLDNN oneDNN và PIR (New IR) để tránh lỗi C++ `ConvertPirAttribute2RuntimeAttribute not support` trên Windows CPU
- **Warmup dummy image:** Chạy 1 ảnh giả lập nhỏ ngay khi khởi tạo để biên dịch JIT đồ thị tính toán, loại bỏ độ trễ 30–40s ở lần predict đầu tiên
- **API 3.x `ocr.predict()`:** Dùng API mới nhất thay cho `ocr.ocr(..., cls=True)` đã deprecated

**✅ Ưu điểm:**
- Chạy hoàn toàn **offline** trên CPU — không cần GPU, không cần API key, không cần internet
- Hỗ trợ tiếng Anh, tiếng Việt và nhiều ngôn ngữ khác
- Tự động phát hiện hướng chữ bị xoay (textline orientation) trên bao bì đồ chơi
- Mô hình Mobile đạt tốc độ cực nhanh (~0.7s) mà vẫn giữ độ chính xác thực tế cao

**❌ Nhược điểm:**
- **GIL (Global Interpreter Lock) của Python:** OCR trên CPU giữ GIL, không thể tăng tốc bằng `threading` hay `ThreadPoolExecutor` — đa luồng không có tác dụng cho bước OCR
- Kích thước download lần đầu lớn (~1.5 GB tổng cộng các model file)
- Trên Windows có bug PaddlePaddle 3.x với MKLDNN và New IR phải vá thủ công
- Không nhận diện được chữ viết tay (chỉ hoạt động với chữ in trên bao bì)

---

### 3. 🖼️ OpenCV (`opencv-python`) — Xử lý ảnh

**Phiên bản:** `>= 4.8.0`

Thư viện xử lý ảnh và thị giác máy tính phổ biến nhất thế giới, dùng để tiền xử lý hình ảnh trước khi đưa vào OCR.

**Kỹ thuật được sử dụng:**

| Kỹ thuật | Hàm OpenCV | Mục đích |
|---|---|---|
| Bilateral Filter | `cv2.bilateralFilter(d=5, sigmaColor=40, sigmaSpace=40)` | Khử nhiễu bề mặt bóng hộp nhựa, **giữ nguyên** viền cạnh sắc nét của chữ |
| CLAHE | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` | Cân bằng độ tương phản cục bộ, kéo nổi chữ bị chìm do bóng đèn/ánh sáng không đều |
| Unsharp Masking | `cv2.addWeighted(1.3, -0.3)` + `cv2.GaussianBlur` | Làm nét viền chữ thêm 30% mà không gây nhiễu hạt hay ringing artifacts |
| Resize thông minh | `cv2.resize()` với `INTER_AREA` / `INTER_CUBIC` | Thu nhỏ về 960px (tăng tốc 40%), phóng lên 800px nếu ảnh quá nhỏ |
| Color convert | `cv2.cvtColor()` | Chuyển RGB↔BGR↔LAB để xử lý từng kênh màu riêng biệt |

**✅ Ưu điểm:**
- Cực nhanh (C++ backend) — toàn bộ tiền xử lý ảnh chỉ mất < 0.1s
- Bilateral Filter nhanh hơn gấp nhiều lần so với `fastNlMeansDenoising`
- Xử lý trên kênh L của không gian màu LAB giúp tăng độ tương phản mà **không làm đổi màu** bao bì

**❌ Nhược điểm:**
- API dạng procedural (thủ tục), không hướng đối tượng — code dài hơn so với Pillow
- Quy ước kênh màu BGR (thay vì RGB chuẩn) dễ gây nhầm lẫn khi tích hợp với Pillow và Streamlit
- `opencv-python` và `opencv-python-headless` xung đột nếu cài cả hai — cần cài `headless` cho môi trường server

---

### 4. 🖼️ Pillow — Đọc & xử lý ảnh ban đầu

**Phiên bản:** `>= 10.0.0`

Thư viện ảnh Python (Python Imaging Library fork), dùng để đọc file ảnh từ Streamlit uploader và xử lý metadata EXIF.

**Tính năng sử dụng:**
- `Image.open()` — đọc ảnh JPG/PNG từ Streamlit `UploadedFile`
- `ImageOps.exif_transpose()` — tự động sửa hướng ảnh chụp từ điện thoại bị xoay 90°/180°/270° theo EXIF metadata
- `image.convert("RGB")` — loại bỏ kênh Alpha (PNG RGBA) và nâng cấp ảnh 1 kênh (Grayscale) thành RGB

**✅ Ưu điểm:**
- Đơn giản, dễ dùng — xử lý được mọi định dạng phổ biến (JPG, PNG, WEBP, BMP, TIFF…)
- `exif_transpose()` giải quyết 100% vấn đề ảnh điện thoại bị nghiêng mà không cần viết tay

**❌ Nhược điểm:**
- Chậm hơn OpenCV đáng kể với các phép tính số học trên mảng pixel
- Không hỗ trợ các thuật toán xử lý ảnh nâng cao (filter, morphology, feature detection)
- Phải chuyển đổi sang NumPy array để dùng với OpenCV

---

### 5. 🔢 NumPy — Mảng số học hiệu năng cao

**Phiên bản:** `>= 1.24.0, < 2.0.0`

Thư viện tính toán mảng N chiều (ndarray) là nền tảng của toàn bộ hệ sinh thái khoa học dữ liệu Python.

> ⚠️ **Giới hạn phiên bản `< 2.0.0`:** PaddlePaddle 3.x và OpenCV 4.x tương thích tối ưu với NumPy 1.x. NumPy 2.x thay đổi API C có thể gây lỗi runtime.

**✅ Ưu điểm:**
- Cầu nối chuẩn giữa Pillow (PIL Image) → NumPy array → OpenCV → PaddleOCR
- Phép tính vector hóa cực nhanh (C/Fortran backend), không cần vòng lặp Python

**❌ Nhược điểm:**
- Giới hạn phiên bản `< 2.0.0` tạo ràng buộc phụ thuộc phức tạp trong `requirements.txt`

---

### 6. 🔍 ddgs (DuckDuckGo Search) — Tìm kiếm web miễn phí

**Phiên bản:** `>= 9.0.0`

Thư viện Python để gọi API tìm kiếm DuckDuckGo không chính thức, **hoàn toàn miễn phí và không cần API key**.

**Chiến lược 3 tầng Fallback:**
1. **Tầng 1:** `extract_smart_query(max_words=4)` → Truy vấn song song `vn-vi` + `wt-wt`
2. **Tầng 2:** `extract_smart_query(max_words=2)` → Nới lỏng truy vấn nếu Tầng 1 trả rỗng
3. **Tầng 3:** Từ khóa gốc đã làm sạch → Truy vấn toàn cầu

**✅ Ưu điểm:**
- **Miễn phí hoàn toàn** — không cần đăng ký tài khoản hay API key (so với Google Search API ~\$5/1000 truy vấn)
- Hỗ trợ tham số `region` để ưu tiên kết quả theo khu vực (`vn-vi` cho Việt Nam, `wt-wt` toàn cầu)
- Tích hợp sẵn nhiều engine tìm kiếm phụ (Yahoo, Brave, Startpage) làm fallback tự động
- `@st.cache_data(ttl=3600)` giảm số lần gọi thực tế bằng cache 1 giờ

**❌ Nhược điểm:**
- **Rate limiting:** DuckDuckGo có thể trả về 429 (Too Many Requests) khi gọi quá nhiều trong thời gian ngắn
- **Không ổn định:** Kết quả thay đổi theo ngày, engine phụ có thể bị chặn (Brave 429, Mojeek 403)
- Không có SLA (Service Level Agreement) — không phù hợp cho môi trường sản xuất doanh nghiệp
- **DNS timeout trên Windows:** `primp` (HTTP client của ddgs) dùng Rust Hickory DNS qua UDP 53 bị timeout trên một số mạng Windows — cần vá monkey-patch (xem mục Lỗi đã biết)

---

### 7. ⚡ primp — HTTP Client hiệu năng cao

**Phiên bản:** Tự động cài theo `ddgs`

HTTP client Rust-based được ddgs dùng làm backend, có khả năng giả lập fingerprint trình duyệt (browser impersonation) để tránh bị chặn bởi các trang web chống bot.

**✅ Ưu điểm:**
- Rất nhanh (Rust backend)
- Browser impersonation giúp các request trông giống Chrome/Firefox thực sự

**❌ Nhược điểm:**
- **Lỗi DNS trên Windows:** Dùng Rust Hickory DNS resolver qua UDP 53 — bị timeout trên một số cấu hình mạng Windows (firewall, VPN, proxy nội bộ)
- **Cách khắc phục:** Monkey-patch `ddgs.http_client.HttpClient.__init__` để ép dùng `dns_resolver="system"` (Win32 OS resolver)

---

### 8. 🔤 RapidFuzz — So khớp chuỗi mờ (Fuzzy String Matching)

**Phiên bản:** `>= 3.0.0`

Thư viện so khớp chuỗi xấp xỉ siêu nhanh, dùng để nhận diện thương hiệu đồ chơi dù OCR đọc sai 1–2 ký tự.

**Kỹ thuật được dùng:**

| Thuật toán | Hàm | Mục đích |
|---|---|---|
| Levenshtein Distance | `Levenshtein.distance()` | Đo khoảng cách edit — chịu được sai sót 1 ký tự (LECO→LEGO) |
| Fuzz Ratio | `fuzz.ratio()` | So khớp toàn chuỗi theo tỉ lệ phần trăm giống nhau |
| Partial Ratio | `fuzz.partial_ratio()` | Khớp bộ phận — tìm thương hiệu nằm trong chuỗi dài hơn (LEGO City) |

**✅ Ưu điểm:**
- Nhanh hơn thư viện `fuzzywuzzy` gốc tới 10–100x (C++ backend)
- Cho phép nhận diện thương hiệu dù OCR đọc nhầm: `'LECO'→'LEGO'`, `'HOTWHEELS'→'Hot Wheels'`
- API đa dạng: exact match, Levenshtein, partial ratio, token set ratio

**❌ Nhược điểm:**
- Chỉ so khớp chuỗi đơn giản — không hiểu ngữ nghĩa (semantic)
- Danh sách thương hiệu `KNOWN_BRANDS` phải cập nhật thủ công khi có thương hiệu mới
- Có thể khớp nhầm với chuỗi ngắn (ví dụ: "NE" khớp với "NERF" nếu ngưỡng quá thấp)

---

### 9. 📊 pandas — Hiển thị bảng dữ liệu OCR

**Phiên bản:** Tự động cài theo Streamlit

Thư viện phân tích dữ liệu, dùng để tạo DataFrame hiển thị bảng kết quả OCR trong Streamlit.

**✅ Ưu điểm:**
- `st.dataframe(df, use_container_width=True)` hiển thị bảng đẹp với filter, sort tự động
- Dễ dàng thêm cột tính toán (cột "Trạng thái" từ confidence)

**❌ Nhược điểm:**
- Overhead khá lớn khi chỉ dùng để hiển thị bảng nhỏ (< 50 dòng) — có thể dùng `st.table()` thay thế

---

## ⚠️ Các vấn đề đã biết & Giải pháp

### 1. PaddlePaddle 3.x Windows — Lỗi oneDNN / PIR

**Lỗi:** `C++ exception: ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`

**Nguyên nhân:** PaddlePaddle >= 3.0 trên Windows CPU kết hợp cơ chế PIR executor (New IR) với oneDNN MKLDNN tạo ra xung đột thuộc tính khi biên dịch đồ thị tính toán.

**Giải pháp đã tích hợp** (`ocr_extractor.py`):
```python
import paddle.inference as paddle_inf
_original = paddle_inf.create_predictor
def _safe_create_predictor(config):
    config.disable_mkldnn()      # Tắt Intel MKLDNN / oneDNN
    config.enable_new_ir(False)  # Tắt PIR (New IR executor)
    return _original(config)
paddle_inf.create_predictor = _safe_create_predictor
```

---

### 2. ddgs DNS Timeout trên Windows

**Lỗi:** `TimeoutException` hoặc `ConnectionError` khi gọi `DDGS().text()`

**Nguyên nhân:** `primp` HTTP client dùng Rust Hickory DNS resolver qua UDP port 53 — bị block trên một số mạng Windows (firewall doanh nghiệp, VPN, proxy nội bộ chặn UDP).

**Giải pháp đã tích hợp** (`search_engine.py`):
```python
import primp, ddgs.http_client
_orig = ddgs.http_client.HttpClient.__init__
def _patched(self, proxy=None, timeout=6, *, verify=True):
    self.client = primp.Client(
        dns_resolver="system",  # Dùng Win32 OS DNS thay vì Rust Hickory
        impersonate="random",
        ...
    )
ddgs.http_client.HttpClient.__init__ = _patched
```

---

### 3. GIL Python và OCR Đa luồng

**Vấn đề:** `ThreadPoolExecutor` **không** tăng tốc bước OCR (PaddlePaddle).

**Lý do:** PaddlePaddle chạy suy luận trên CPU giữ GIL (Global Interpreter Lock) của Python. Các thread Python không thể chạy song song thực sự — chỉ luân phiên nhau.

**Áp dụng đa luồng đúng chỗ:** `ThreadPoolExecutor(max_workers=2)` **chỉ** được dùng cho bước tìm kiếm mạng (I/O-bound) vì network I/O tự động nhả GIL, cho phép 2 truy vấn `vn-vi` và `wt-wt` chạy thực sự song song.

---

## 🔭 Hướng phát triển

| Tính năng | Mức độ ưu tiên | Ghi chú |
|---|---|---|
| Hỗ trợ GPU (CUDA) | Cao | Thêm `paddlepaddle-gpu` cho máy có NVIDIA GPU → tốc độ x5–x10 |
| OCR tiếng Việt | Trung bình | Thêm `lang="vi"` hoặc multilingual model cho vỏ hộp đồ chơi Việt Nam |
| Cache kết quả tìm kiếm xuống DB | Trung bình | Dùng SQLite để lưu từ khóa → kết quả, giảm phụ thuộc mạng |
| Hỗ trợ ảnh chụp từ webcam | Thấp | Tích hợp `st.camera_input()` của Streamlit |
| Thêm nguồn tìm kiếm | Thấp | Tích hợp Shopee API, Tiki API (khi có) |
| Docker container | Thấp | Đóng gói toàn bộ môi trường để deploy dễ dàng hơn |
