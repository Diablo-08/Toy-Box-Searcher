"""
Module: app.py
Mục đích: Ứng dụng web giao diện người dùng (Streamlit) nhận diện chữ trên ảnh hộp đồ chơi,
          trích xuất từ khóa thương hiệu/SKU và tra cứu thông tin sản phẩm trên Internet theo khu vực.
          Đảm bảo: Xóa sạch dữ liệu cũ khi tải ảnh mới, cập nhật ô tìm kiếm tự động,
          và lọc từ khóa độ tin cậy trên 95% (100% không lấy Wikipedia).
Tác giả: Senior Computer Vision Engineer & Python Developer
Môi trường: Python 3.11.9, Streamlit 1.x
"""

import time
import urllib.parse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Import các module chức năng tự xây dựng
from preprocessing import normalize_image, preprocess_image
from ocr_extractor import extract_text_and_keywords
from search_engine import search_toy_info, extract_domain, get_last_search_error

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Toy Box Searcher - Nhận diện hộp đồ chơi",
    page_icon="🧸",
    layout="wide",
    initial_sidebar_state="expanded",
)
import streamlit as st

st.markdown("""
    <style>
        /* Ẩn toàn bộ header mặc định của Streamlit (chứa nút Fork, GitHub, menu 3 chấm) */
        header[data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Ẩn triệt để cụm nút badge / avatar / vương miện ở góc dưới bên phải */
        div[class*="viewerBadge"], 
        div[data-testid="stStatusWidget"],
        footer {
            display: none !important;
            visibility: hidden !important;
        }
        
        /* Cân đối lại khoảng trắng phía trên cho giao diện khít đẹp */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS để giao diện trực quan, hiện đại và chuẩn công nghiệp
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .speed-badge {
        display: inline-block;
        background-color: #ECFDF5;
        color: #059669;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #A7F3D0;
        margin-bottom: 10px;
    }
    .region-badge {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #FCD34D;
        margin-bottom: 8px;
    }
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    .result-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .domain-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .result-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1D4ED8;
        text-decoration: none;
    }
    .result-title:hover {
        text-decoration: underline;
    }
    .result-body {
        font-size: 0.95rem;
        color: #374151;
        margin-top: 6px;
        line-height: 1.45;
    }
    .direct-links-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_app_data() -> None:
    """Xóa sạch toàn bộ dữ liệu, hình ảnh, từ khóa và kết quả của ảnh cũ."""
    st.session_state.current_file_id = None
    st.session_state.image_original_bgr = None
    st.session_state.image_enhanced_bgr = None
    st.session_state.raw_texts = []
    st.session_state.final_keyword = ""
    st.session_state.search_results = []
    st.session_state.search_error = None
    st.session_state.keyword_box = ""
    st.session_state.ocr_time_seconds = 0.0
    st.session_state.ocr_success = False


def init_session_state() -> None:
    """Khởi tạo các biến trong st.session_state để quản lý trạng thái ứng dụng."""
    if "current_file_id" not in st.session_state:
        st.session_state.current_file_id = None
    if "image_original_bgr" not in st.session_state:
        st.session_state.image_original_bgr = None
    if "image_enhanced_bgr" not in st.session_state:
        st.session_state.image_enhanced_bgr = None
    if "raw_texts" not in st.session_state:
        st.session_state.raw_texts = []
    if "final_keyword" not in st.session_state:
        st.session_state.final_keyword = ""
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "search_error" not in st.session_state:
        st.session_state.search_error = None
    if "keyword_box" not in st.session_state:
        st.session_state.keyword_box = ""
    if "ocr_time_seconds" not in st.session_state:
        st.session_state.ocr_time_seconds = 0.0
    if "ocr_success" not in st.session_state:
        st.session_state.ocr_success = False
    if "search_region" not in st.session_state:
        st.session_state.search_region = "vn-vi"
    if "ocr_model_type" not in st.session_state:
        st.session_state.ocr_model_type = "mobile"


init_session_state()

def on_ocr_model_change() -> None:
    """Khi đổi mô hình OCR, xóa file_id để hệ thống tự động quét lại ảnh với mô hình mới."""
    st.session_state.current_file_id = None


OCR_MODEL_LABELS = {
    "mobile": "⚡ Flash (~0.7s - Khuyên dùng)",
    "medium": "🎯 Standard (Chính xác cao)",
}

# ==============================================================================
# THANH BÊN (SIDEBAR) CẤU HÌNH & TĂNG TỐC HỆ THỐNG
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình hệ thống")
    st.radio(
        "Mô hình OCR nhận diện chữ:",
        options=["mobile", "medium"],
        format_func=lambda m: OCR_MODEL_LABELS.get(m, m),
        key="ocr_model_type",
        on_change=on_ocr_model_change,
        help="Flash dùng PP-OCRv5 Mobile nhẹ và nhanh gấp 7 lần trên CPU. Standard dùng mô hình tiêu chuẩn.",
    )

    st.markdown("---")
    st.markdown(
        """
        **💡 Tối ưu hóa hiệu năng CPU:**
        - **Flash:** Tối ưu hóa tốc độ, thời gian quét chỉ ~0.6s - 1.2s.
        - **Đa luồng tìm kiếm:** Truy vấn song song khu vực và toàn cầu qua ThreadPoolExecutor.
        - **Lọc thông minh:** Tự động loại bỏ 100% Wikipedia, chỉ giữ các trang sản phẩm & mua sắm.
        """
    )

# ==============================================================================
# PHẦN HEADER
# ==============================================================================
st.markdown('<div class="main-title">🧸 Toy Box Searcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    "Hệ thống thông minh nhận diện chữ trên bao bì đồ chơi (PaddleOCR 3.x), "
    "trích xuất thương hiệu & mã SKU, và tra cứu sản phẩm theo khu vực địa lý."
    "</div>",
    unsafe_allow_html=True,
)

# ==============================================================================
# KHUNG TẢI ẢNH (FILE UPLOADER) & THANH TIẾN TRÌNH BIẾN MẤT TỰ ĐỘNG
# ==============================================================================
uploaded_file = st.file_uploader(
    "Tải lên hình ảnh hộp đồ chơi của bạn (hỗ trợ JPG, JPEG, PNG):",
    type=["jpg", "jpeg", "png"],
    help="Hãy chọn ảnh chụp rõ nét bề mặt hộp đồ chơi hoặc mặt trước chứa logo và mã sản phẩm.",
)

# Vùng chứa thanh tiến trình động: hiển thị trong lúc xử lý và BIẾN MẤT HOÀN TOÀN khi hoàn thành
progress_placeholder = st.empty()

# 1. Trường hợp người dùng xóa ảnh đã tải (ấn dấu X hoặc hủy upload)
if uploaded_file is None:
    if st.session_state.current_file_id is not None:
        reset_app_data()
        st.rerun()

# 2. Trường hợp có ảnh được tải lên
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    # Khi bắt đầu up ảnh MỚI: XÓA NGAY TOÀN BỘ DỮ LIỆU VÀ KẾT QUẢ CŨ
    if st.session_state.current_file_id != file_id:
        # Xóa sạch trạng thái cũ để không lưu vết ảnh trước
        st.session_state.image_original_bgr = None
        st.session_state.image_enhanced_bgr = None
        st.session_state.raw_texts = []
        st.session_state.final_keyword = ""
        st.session_state.search_results = []
        st.session_state.search_error = None
        st.session_state.keyword_box = ""

        try:
            start_time = time.time()
            progress_placeholder.progress(
                15, text="🖼️ Bước 1/4: Đang chuẩn hóa kích thước tối ưu & sửa góc xoay..."
            )

            # 1. Đọc ảnh từ Streamlit UploadedFile vào PIL
            image_pil = Image.open(uploaded_file)

            # 2. Tiền xử lý: Chuẩn hóa kích thước thông minh (max 960px) & sửa góc xoay
            image_orig_bgr = normalize_image(image_pil)

            progress_placeholder.progress(
                35, text="✨ Bước 2/4: Đang cân bằng tương phản cục bộ CLAHE & khử nhiễu..."
            )

            # 3. Tiền xử lý: Tăng cường tương phản cục bộ CLAHE & làm nét nhẹ
            image_enh_bgr = preprocess_image(image_orig_bgr)

            # 4. Trích xuất chữ OCR & từ khóa tìm kiếm (kèm callback cập nhật tiến độ)
            def update_ocr_progress(pct: int, msg: str) -> None:
                progress_placeholder.progress(pct, text=msg)

            # Xác định ngưỡng độ tin cậy: Flash lấy trên 85% (0.85), Standard lấy trên 95% (0.95)
            active_conf_thresh = (
                0.85 if st.session_state.ocr_model_type == "mobile" else 0.95
            )

            raw_texts, final_keyword = extract_text_and_keywords(
                image_orig_bgr,
                image_enh_bgr,
                min_keyword_conf=active_conf_thresh,
                progress_callback=update_ocr_progress,
                model_type=st.session_state.ocr_model_type,
            )

            elapsed_time = time.time() - start_time

            # 5. Lưu kết quả xử lý ảnh & OCR vào session_state
            st.session_state.current_file_id = file_id
            st.session_state.image_original_bgr = image_orig_bgr
            st.session_state.image_enhanced_bgr = image_enh_bgr
            st.session_state.raw_texts = raw_texts
            st.session_state.final_keyword = final_keyword
            # ĐỒNG BỘ TRỰC TIẾP VÀO KEY CỦA WIDGET để ô tìm kiếm tự động đổi sang từ khóa mới
            st.session_state.keyword_box = final_keyword
            st.session_state.ocr_time_seconds = elapsed_time
            st.session_state.ocr_success = bool(raw_texts)

            # 6. Tự động tìm kiếm nếu đã trích xuất được từ khóa hợp lệ
            if final_keyword:
                progress_placeholder.progress(
                    95, text=f"🌐 Đang tra cứu '{final_keyword}' trên Internet..."
                )
                st.session_state.search_results = search_toy_info(
                    final_keyword,
                    max_results=5,
                    region=st.session_state.search_region,
                )
                st.session_state.search_error = get_last_search_error()
            else:
                st.session_state.search_results = []
                st.session_state.search_error = None

        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình xử lý ảnh: {e}")
        finally:
            # QUAN TRỌNG: Thanh tiến độ biến mất hoàn toàn ngay sau khi hoàn thành
            progress_placeholder.empty()

# ==============================================================================
# HIỂN THỊ KẾT QUẢ KHI ĐÃ CÓ DỮ LIỆU TRONG SESSION STATE
# ==============================================================================
if st.session_state.image_original_bgr is not None:
    col_left, col_right = st.columns([1, 1], gap="medium")

    # --------------------------------------------------------------------------
    # CỘT TRÁI: HIỂN THỊ HÌNH ẢNH SO SÁNH
    # --------------------------------------------------------------------------
    with col_left:
        st.subheader("🖼️ Hình ảnh phân tích")

        # Sử dụng st.tabs để người dùng so sánh ảnh gốc và ảnh sau khi tăng cường CLAHE
        tab_original, tab_enhanced = st.tabs(
            ["Ảnh gốc chuẩn hóa", "Ảnh tăng cường"]
        )

        with tab_original:
            # Chuyển BGR -> RGB để hiển thị đúng màu trong Streamlit
            rgb_orig = cv2.cvtColor(
                st.session_state.image_original_bgr, cv2.COLOR_BGR2RGB
            )
            h, w = rgb_orig.shape[:2]
            st.image(
                rgb_orig,
                caption=f"Ảnh chuẩn hóa (Kích thước: {w}x{h}px)",
                use_container_width=True,
            )

        with tab_enhanced:
            rgb_enh = cv2.cvtColor(
                st.session_state.image_enhanced_bgr, cv2.COLOR_BGR2RGB
            )
            st.image(
                rgb_enh,
                caption="Ảnh tăng cường đã khử nhiễu",
                use_container_width=True,
            )

    # --------------------------------------------------------------------------
    # CỘT PHẢI: TỪ KHÓA, BẢNG OCR & KẾT QUẢ TÌM KIẾM
    # --------------------------------------------------------------------------
    with col_right:
        st.subheader("🔍 Trích xuất từ khóa & Tra cứu")

        # Ngưỡng độ tin cậy hiển thị theo mô hình đang chọn (Mobile: 85%, Medium: 95%)
        conf_threshold = (
            0.85 if st.session_state.ocr_model_type == "mobile" else 0.95
        )
        conf_label = "85%" if st.session_state.ocr_model_type == "mobile" else "95%"

        # Hiển thị thông báo trạng thái độ tin cậy và kết quả quét
        if not st.session_state.ocr_success:
            st.error(
                "❌ **Không quét được kết quả!** Không phát hiện thấy chữ nào trên hình ảnh này.\n\n"
                "*Nguyên nhân thường gặp:* Ảnh quá mờ, góc chụp bị khuất, lóa sáng bề mặt vỏ hộp hoặc không có chữ.\n\n"
                "👉 **Bạn có thể tự nhập từ khóa tên đồ chơi vào ô dưới đây để tra cứu ngay lập tức!**"
            )
        elif not st.session_state.final_keyword:
            st.warning(
                f"⚠️ Đã quét được {len(st.session_state.raw_texts)} dòng chữ, nhưng **không có từ khóa nào đạt độ tin cậy trên {conf_label}** "
                "(chữ bị mờ hoặc nét không đủ rõ ràng để đảm bảo chính xác).\n\n"
                "👉 Bạn hãy nhập từ khóa tên đồ chơi vào ô dưới đây để tra cứu!"
            )
        else:
            time_str = f"{st.session_state.ocr_time_seconds:.2f}s"
            mode_tag = "Flash ⚡" if st.session_state.ocr_model_type == "mobile" else "Standard 🎯"
            st.markdown(
                f'<div class="speed-badge">⚡ Thời gian quét: {time_str} ({mode_tag}) | Trích xuất từ khóa đạt độ tin cậy > {conf_label}</div>',
                unsafe_allow_html=True,
            )

        # Ô nhập từ khóa: Gắn trực tiếp với key='keyword_box' để tự động cập nhật khi đổi ảnh
        keyword_input = st.text_input(
            "Từ khóa tìm kiếm (bạn có thể tự do chỉnh sửa nếu cần):",
            key="keyword_box",
            help=f"Từ khóa tự động tổng hợp từ Thương hiệu, Mã SKU và Dòng sản phẩm (độ tin cậy > {conf_label}).",
        )

        # Chọn khu vực tra cứu theo vị trí địa lý
        col_reg, col_btn = st.columns([1, 1])
        with col_reg:
            region_map = {
                "Việt Nam": "vn-vi",
                "Toàn cầu": "wt-wt",
            }
            selected_region_label = st.selectbox(
                "Khu vực tìm kiếm:",
                options=list(region_map.keys()),
                index=0,
                help="Chọn khu vực ưu tiên tìm kiếm.",
            )
            chosen_region = region_map[selected_region_label]
            st.session_state.search_region = chosen_region

        with col_btn:
            st.markdown(
                "<div style='height: 27px;'></div>", unsafe_allow_html=True
            )
            search_button = st.button("🔎 Tìm kiếm lại", use_container_width=True)

        # Thanh tiến độ tìm kiếm (hiển thị khi đang tìm và biến mất khi hoàn thành)
        search_progress_placeholder = st.empty()

        if search_button:
            current_kw = st.session_state.keyword_box.strip()
            if current_kw:
                try:
                    search_progress_placeholder.progress(
                        30, text=f"🌐 Đang kết nối máy chủ tra cứu '{current_kw}'..."
                    )
                    time.sleep(0.04)
                    search_progress_placeholder.progress(
                        70, text="🔍 Đang trích xuất liên kết..."
                    )
                    # Gọi hàm tìm kiếm từ search_engine theo khu vực đã chọn
                    st.session_state.search_results = search_toy_info(
                        current_kw,
                        max_results=5,
                        region=chosen_region,
                    )
                    st.session_state.search_error = get_last_search_error()
                finally:
                    # QUAN TRỌNG: Thanh tiến trình biến mất hoàn toàn khi tìm kiếm xong
                    search_progress_placeholder.empty()
            else:
                st.warning("Vui lòng nhập từ khóa trước khi bấm tìm kiếm.")

        st.markdown("---")

        # Expander hiển thị toàn bộ văn bản thô OCR kèm độ tin cậy
        with st.expander(
            f"📄 Xem toàn bộ văn bản thô OCR quét được ({len(st.session_state.raw_texts)} dòng)"
        ):
            if st.session_state.raw_texts:
                table_data = [
                    {
                        "STT": idx + 1,
                        "Văn bản nhận diện": text,
                        "Độ tin cậy (Confidence)": f"{conf * 100:.1f}%",
                        "Trạng thái": (
                            f"✅ Đạt chuẩn (>{conf_label})"
                            if conf >= conf_threshold
                            else f"⚠️ Dưới {conf_label} (Bỏ qua)"
                        ),
                    }
                    for idx, (text, conf) in enumerate(st.session_state.raw_texts)
                ]
                df_texts = pd.DataFrame(table_data)
                st.dataframe(df_texts, use_container_width=True, hide_index=True)
            else:
                st.info("Không có dòng chữ nào được nhận diện từ ảnh này.")

        st.markdown("### 🌐 Kết quả tra cứu thông tin")

        # Hiển thị huy hiệu khu vực đang tra cứu
        st.markdown(
            f'<div class="region-badge">📍 Khu vực tìm kiếm: {selected_region_label}</div>',
            unsafe_allow_html=True,
        )

        current_active_kw = st.session_state.keyword_box.strip()

        # Thanh công cụ tra cứu trực tiếp 1-chạm
        if current_active_kw:
            encoded_q = urllib.parse.quote_plus(current_active_kw)
            if chosen_region == "vn-vi":
                shopee_url = f"https://shopee.vn/search?keyword={encoded_q}"
                mykingdom_url = f"https://www.mykingdom.com.vn/search?q={encoded_q}"
                google_url = f"https://www.google.com/search?q={encoded_q}&cr=countryVN"
                youtube_url = f"https://www.youtube.com/results?search_query={encoded_q}+do+choi+review"
                st.markdown(
                    f"""
                    <div class="direct-links-box">
                        <b style="color: black;">🚀 Tra cứu trực tiếp tại Việt Nam:</b>&nbsp;&nbsp;
                        <a href="{shopee_url}" target="_blank" rel="noopener noreferrer" style="margin-right: 12px; font-weight: 600; text-decoration: none; color: #EA580C;">🛒 Mua trên Shopee</a>
                        <a href="{mykingdom_url}" target="_blank" rel="noopener noreferrer" style="margin-right: 12px; font-weight: 600; text-decoration: none; color: #E11D48;">🧸 Xem trên MyKingdom</a>
                        <a href="{google_url}" target="_blank" rel="noopener noreferrer" style="margin-right: 12px; font-weight: 600; text-decoration: none; color: #2563EB;">🔍 Google VN</a>
                        <a href="{youtube_url}" target="_blank" rel="noopener noreferrer" style="font-weight: 600; text-decoration: none; color: #DC2626;">▶️ Video Mở Hộp</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                google_url = f"https://www.google.com/search?q={encoded_q}"
                amazon_url = f"https://www.amazon.com/s?k={encoded_q}"
                youtube_url = f"https://www.youtube.com/results?search_query={encoded_q}+toy+review"
                st.markdown(
                    f"""
                    <div class="direct-links-box">
                        <b style="color: black;">🚀 Tra cứu trực tiếp toàn cầu:</b>&nbsp;&nbsp;
                        <a href="{google_url}" target="_blank" rel="noopener noreferrer" style="margin-right: 12px; font-weight: 600; text-decoration: none; color: #2563EB;">🔍 Google Global</a>
                        <a href="{amazon_url}" target="_blank" rel="noopener noreferrer" style="margin-right: 12px; font-weight: 600; text-decoration: none; color: #EA580C;">📦 Amazon</a>
                        <a href="{youtube_url}" target="_blank" rel="noopener noreferrer" style="font-weight: 600; text-decoration: none; color: #DC2626;">▶️ YouTube Review</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Hiển thị thông báo cảnh báo nếu có lỗi kết nối mạng
        if st.session_state.search_error:
            st.warning(f"⚠️ {st.session_state.search_error}")

        # Hiển thị danh sách kết quả tìm kiếm dạng thẻ
        results = st.session_state.search_results
        if results:
            for idx, res in enumerate(results, start=1):
                title = res.get("title", "Không có tiêu đề")
                href = res.get("href", "#")
                body = res.get("body", "Không có mô tả.")
                domain = extract_domain(href)

                card_html = f"""
                <div class="result-card">
                    <span class="domain-badge">{domain}</span><br/>
                    <a class="result-title" href="{href}" target="_blank" rel="noopener noreferrer">
                        {idx}. {title} ↗
                    </a>
                    <div class="result-body">{body}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            if current_active_kw:
                st.info(
                    "💡 Chưa tìm thấy bài viết cụ thể qua bộ quét tự động. "
                    "Bạn hãy bấm vào các nút liên kết nhanh bên trên (Shopee / MyKingdom / Google) "
                    "để xem ngay giá bán và thông tin chi tiết!"
                )
            else:
                st.info("Hãy nhập từ khóa và nhấn 'Tìm kiếm lại' để tra cứu.")
else:
    # Giao diện chào mừng khi chưa upload ảnh
    st.info(
        "👋 Chào mừng bạn! Hãy tải lên một bức ảnh chụp hộp đồ chơi ở khung trên để bắt đầu."
    )
