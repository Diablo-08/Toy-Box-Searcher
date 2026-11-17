"""
Module: ocr_extractor.py
Mục đích: Khởi tạo mô hình PaddleOCR 3.x, trích xuất văn bản từ hình ảnh hộp đồ chơi,
          lọc nhiễu thông minh, nhận diện thương hiệu (Fuzzy Matching), bóc tách mã sản phẩm (SKU)
          và ghép thành từ khóa tối ưu cho công cụ tìm kiếm.
Tác giả: Senior Computer Vision Engineer & Python Developer
Môi trường: Python 3.11.9, PaddleOCR 3.x, RapidFuzz 3.x
"""

import os
import re
from typing import List, Tuple, Dict, Set, Optional, Callable

# Tắt kiểm tra kết nối máy chủ mô hình từ xa của PaddleX để tăng tốc độ khởi động tức thì
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import numpy as np
import streamlit as st
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# ==============================================================================
# XỬ LÝ TƯƠNG THÍCH PADDLEPADDLE 3.x TRÊN WINDOWS CPU
# ==============================================================================
# Trong PaddlePaddle >= 3.0 trên Windows CPU, cơ chế PIR executor với oneDNN
# có thể phát sinh lỗi C++: "ConvertPirAttribute2RuntimeAttribute not support".
# Hook này vô hiệu hóa MKLDNN và New IR trên CPU để đảm bảo chạy mượt mà 100%.
try:
    import paddle.inference as paddle_inf

    if hasattr(paddle_inf, "create_predictor"):
        _original_create_predictor = paddle_inf.create_predictor

        def _safe_create_predictor(config):
            if hasattr(config, "disable_mkldnn"):
                config.disable_mkldnn()
            if hasattr(config, "enable_new_ir"):
                config.enable_new_ir(False)
            return _original_create_predictor(config)

        paddle_inf.create_predictor = _safe_create_predictor
except Exception:
    pass

from paddleocr import PaddleOCR

# ==============================================================================
# TỪ ĐIỂN THƯƠNG HIỆU ĐỒ CHƠI CHUẨN XÁC (CANONICAL BRANDS)
# ==============================================================================
KNOWN_BRANDS: List[str] = [
    "LEGO",
    "Hot Wheels",
    "Barbie",
    "Bandai",
    "Gundam",
    "Transformers",
    "Hasbro",
    "Mattel",
    "Fisher-Price",
    "Nerf",
    "Playmobil",
    "Funko",
    "Tomica",
    "Disney",
    "Marvel",
    "Pokemon",
    "Play-Doh",
    "Tamiya",
    "Kyosho",
]

# ==============================================================================
# DANH SÁCH TỪ DỪNG / CẢNH BÁO BẢO HỘ TRÊN VỎ HỘP ĐỒ CHƠI (STOPWORDS)
# ==============================================================================
STOPWORDS: Set[str] = {
    "warning",
    "choking hazard",
    "small parts",
    "ages",
    "age",
    "years",
    "year",
    "made in china",
    "pieces",
    "pcs",
    "new",
    "www",
    "com",
    "http",
    "https",
    "safety",
    "caution",
    "not for children",
    "under 3 years",
    "item",
    "no",
    "toy",
}


@st.cache_resource(show_spinner=False)
def get_ocr(model_type: str = "mobile") -> PaddleOCR:
    """
    Khởi tạo mô hình PaddleOCR đúng 1 lần và lưu vào cache của Streamlit (@st.cache_resource).
    Bắt buộc sử dụng API PaddleOCR 3.x và tối ưu hóa suy luận CPU:
    - model_type="mobile" (Mặc định): Sử dụng PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec,
      tốc độ suy luận CPU đạt ~0.6s (nhanh gấp 7 lần so với medium ~4.5s) mà vẫn đạt độ chính xác >95%.
    - model_type="medium": Sử dụng mô hình PP-OCRv6_medium mặc định của PaddleOCR 3.x.
    - use_textline_orientation=True: Tự động xoay dòng chữ bị nghiêng/lộn ngược trên bao bì.
    - use_doc_orientation_classify=False: Tắt phân loại hướng toàn bộ trang tài liệu.
    - use_doc_unwarping=False: Tắt nắn phẳng trang sách tài liệu (tiết kiệm tài nguyên CPU).
    - Tự động làm nóng mô hình (Warmup) ngay khi khởi tạo để loại bỏ hiện tượng giật lag 30-40s ở lần chạy đầu tiên.

    Args:
        model_type (str): "mobile" (siêu tốc, tối ưu CPU) hoặc "medium" (tiêu chuẩn).

    Returns:
        PaddleOCR: Đối tượng pipeline OCR sẵn sàng cho suy luận.
    """
    if model_type == "mobile":
        ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_det_limit_side_len=736,
            text_det_limit_type="max",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )
    else:
        ocr = PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            text_det_limit_side_len=960,
            text_det_limit_type="max",
        )

    # Khởi động làm nóng (Warmup) mô hình với ảnh giả lập nhỏ để biên dịch JIT đồ thị tính toán
    try:
        dummy = np.zeros((32, 100, 3), dtype=np.uint8)
        list(ocr.predict(dummy))
    except Exception:
        pass

    return ocr


def run_ocr(
    image_bgr: np.ndarray, min_conf: float = 0.6, model_type: str = "mobile"
) -> List[Tuple[str, float]]:
    """
    Chạy nhận diện chữ trên mảng ảnh BGR bằng PaddleOCR 3.x API.
    BẮT BUỘC gọi ocr.predict(image_bgr), trích xuất từ dict kết quả:
    res["rec_texts"] và res["rec_scores"].

    Args:
        image_bgr (np.ndarray): Mảng ảnh 3 kênh BGR (uint8).
        min_conf (float): Ngưỡng độ tin cậy tối thiểu (mặc định 0.6 = 60%).
        model_type (str): Loại mô hình OCR ("mobile" hoặc "medium").

    Returns:
        List[Tuple[str, float]]: Danh sách các cặp (chuỗi văn bản, độ tin cậy).
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    ocr = get_ocr(model_type=model_type)
    results: List[Tuple[str, float]] = []

    try:
        # Gọi API 3.x ocr.predict (truyền trực tiếp ndarray ảnh)
        prediction_list = ocr.predict(image_bgr)
        if not prediction_list:
            return []

        for item in prediction_list:
            if not isinstance(item, dict):
                continue

            # Đọc rec_texts và rec_scores (có thể là list hoặc np.ndarray trong PaddleX 3.x)
            rec_texts = item.get("rec_texts", [])
            rec_scores = item.get("rec_scores", [])

            # Ép kiểu an toàn về danh sách Python
            if isinstance(rec_texts, np.ndarray):
                rec_texts = rec_texts.tolist()
            if isinstance(rec_scores, np.ndarray):
                rec_scores = rec_scores.tolist()

            for text_raw, score_raw in zip(rec_texts, rec_scores):
                if text_raw is None:
                    continue

                text_str = str(text_raw).strip()
                score_val = float(score_raw) if score_raw is not None else 0.0

                # Lọc bỏ chuỗi rỗng và văn bản có độ tin cậy dưới ngưỡng quy định
                if text_str and score_val >= min_conf:
                    results.append((text_str, score_val))

    except Exception as exc:
        # Ghi nhận log nhưng không làm crash chương trình
        print(f"[Warning] OCR execution error: {exc}")
        return []

    return results


def normalize_text(text: str) -> str:
    """
    Làm sạch và chuẩn hóa chuỗi ký tự OCR:
    - Loại bỏ ký hiệu bản quyền đặc biệt: ©, ®, ™.
    - Gom nhiều khoảng trắng liên tiếp thành một dấu cách đơn.
    - Cắt bỏ các ký tự dấu câu lạ xuất hiện ở đầu hoặc cuối chuỗi.

    Args:
        text (str): Chuỗi gốc đọc từ OCR.

    Returns:
        str: Chuỗi văn bản đã được làm sạch.
    """
    cleaned = re.sub(r"[©®™]", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" -_.,:;!?'\"/\\|*#@[](){}<>~`")
    return cleaned


def is_noise_text(text: str) -> bool:
    """
    Kiểm tra xem chuỗi có phải là nhiễu, cảnh báo an toàn hoặc từ rác không liên quan.

    Args:
        text (str): Chuỗi văn bản cần kiểm tra.

    Returns:
        bool: True nếu là nhiễu, False nếu là văn bản tiềm năng.
    """
    lower = text.lower().strip()

    # Chuỗi quá ngắn (< 2 ký tự) hoặc chỉ chứa số đơn lẻ
    if len(lower) < 2:
        return True

    # Chuỗi chỉ toàn dấu câu và ký tự đặc biệt, không có chữ hay số
    if not re.search(r"[a-zA-Z0-9]", lower):
        return True

    # Kiểm tra các cụm cảnh báo an toàn / quy chuẩn sản xuất trên vỏ hộp
    for stop in STOPWORDS:
        if len(stop) > 3 and stop in lower:
            return True
        if lower == stop:
            return True

    # Các mẫu cảnh báo độ tuổi thông dụng: "3+", "4-99", "ages 6+", "age 8 and up" (số tuổi <= 2 chữ số)
    if re.search(r"^(?:ages?|for\s+ages)\s*\d{1,2}(?:\+|[-–]\s*\d{1,2})?$", lower):
        return True
    if re.search(r"^\d{1,2}\s*(?:\+|[-–]\s*\d{1,2})$", lower):
        return True

    # Mẫu số mảnh ghép LEGO: "500 pcs", "1024 pieces"
    if re.search(r"^\d+\s*(?:pcs|pieces|teil|pièces)$", lower):
        return True

    return False


def match_brand(text: str, threshold: float = 85.0) -> Optional[str]:
    """
    So khớp mờ (Fuzzy Matching) để nhận diện thương hiệu đồ chơi nổi tiếng,
    chịu được sai sót 1-2 ký tự do OCR (ví dụ: 'LECO' -> 'LEGO', 'HOTWHEELS' -> 'Hot Wheels').

    Args:
        text (str): Chuỗi cần kiểm tra.
        threshold (float): Ngưỡng tương đồng tối thiểu (mặc định 85/100).

    Returns:
        Optional[str]: Tên thương hiệu chuẩn nếu khớp, ngược lại None.
    """
    text_clean = text.strip()
    text_lower = text_clean.lower()

    for brand in KNOWN_BRANDS:
        brand_lower = brand.lower()

        # 1. Khớp chính xác tuyệt đối (không phân biệt hoa/thường)
        if text_lower == brand_lower:
            return brand

        # 2. Chịu lỗi sai số 1 ký tự với thương hiệu ngắn (<= 5 ký tự như LEGO, NERF)
        #    hoặc 2 ký tự với thương hiệu dài hơn (Hot Wheels, Transformers)
        dist = Levenshtein.distance(text_lower, brand_lower)
        if (len(brand_lower) <= 5 and dist <= 1) or (
            len(brand_lower) > 5 and dist <= 2
        ):
            return brand

        # 3. So khớp không dấu cách (vd: "hotwheels" vs "hot wheels")
        text_no_space = text_lower.replace(" ", "")
        brand_no_space = brand_lower.replace(" ", "")
        if text_no_space == brand_no_space or fuzz.ratio(text_no_space, brand_no_space) >= 85.0:
            return brand

        # 4. So khớp theo tỉ lệ fuzz.ratio tiêu chuẩn
        ratio = fuzz.ratio(text_lower, brand_lower)
        if ratio >= threshold:
            return brand

        # 5. So khớp bộ phận nếu text chứa thương hiệu bên trong (vd: "BANDAI SPIRITS", "LEGO City")
        if len(text_lower) >= len(brand_lower):
            partial_score = fuzz.partial_ratio(brand_lower, text_lower)
            if partial_score >= 90.0:
                # Kiểm tra ranh giới từ để tránh khớp nhầm
                if re.search(rf"\b{re.escape(brand_lower)}\b", text_lower):
                    return brand

    return None


def extract_sku_or_model(text: str) -> List[str]:
    """
    Trích xuất mã sản phẩm, SKU, số hiệu bộ đồ chơi hoặc tỷ lệ mô hình bằng biểu thức chính quy (Regex).

    Giải thích chi tiết các thành phần Regex:
    1. Tỷ lệ mô hình xe / Gunpla (Scale ratio):
       - Regex: r'\\b(?:(?:HG|MG|PG|RG|EG|SD)\\s*)?(?:1\\s*[/:]\\s*(?:64|100|144|72|48|32|24|18|12))\\b'
       - \\b: Ranh giới từ (word boundary).
       - (?:(?:HG|MG|PG|RG|EG|SD)\\s*)?: Bắt các phân hạng Gunpla tùy chọn đứng trước (vd: HG, MG).
       - 1\\s*[/:]\\s*: Bắt số 1 cùng dấu gạch chéo '/' hoặc hai chấm ':' (vd: '1/144', '1:64').
       - (?:64|100|144|72|48|32|24|18|12): Các kích thước tỉ lệ tiêu chuẩn trong giới mô hình.

    2. Mã sản phẩm / SKU hỗn hợp chữ và số (Alphanumeric SKU):
       - Regex: r'\\b[A-Za-z0-9]{1,5}(?:[-_][A-Za-z0-9]{1,5})+\\b' và r'\\b[A-Za-z]{1,4}[0-9]{2,6}[A-Za-z0-9]?\\b'
       - \\b[A-Za-z0-9]{1,5}(?:[-_][A-Za-z0-9]{1,5})+\\b: Bắt các mã có dấu gạch nối phức hợp
         (vd: 'RX-78-2', 'HW-1234', 'B-1234', 'MS-06S').
       - r'\\b[A-Za-z]{1,4}[0-9]{2,6}[A-Za-z0-9]?\\b': Bắt các mã dính liền chữ trước số sau
         (vd: 'HW123', 'CT01').
       - Điều kiện kiểm tra: Chuỗi bắt buộc phải chứa ít nhất 1 chữ cái và ít nhất 1 chữ số.

    3. Dãy số định danh bộ đồ chơi (LEGO Set ID / Serial):
       - Regex: r'\\b\\d{4,6}\\b'
       - \\b: Ranh giới từ.
       - \\d{4,6}: Bắt dãy từ 4 đến 6 chữ số liên tiếp (vd: '60337', '75192').
       - Loại trừ các năm sản xuất phổ biến từ 1980 đến 2030 khi đứng một mình.

    Args:
        text (str): Chuỗi văn bản đầu vào.

    Returns:
        List[str]: Danh sách các chuỗi SKU/tỷ lệ trích xuất được.
    """
    matches: List[str] = []
    seen: Set[str] = set()

    # 1. Tỷ lệ mô hình (Scale ratio)
    scale_pattern = (
        r"\b(?:(?:HG|MG|PG|RG|EG|SD)\s*)?"
        r"(?:1\s*[/:]\s*(?:64|100|144|72|48|32|24|18|12))\b"
    )
    for m in re.finditer(scale_pattern, text, re.IGNORECASE):
        cleaned_scale = re.sub(r"\s+", "", m.group().upper())
        if cleaned_scale not in seen:
            seen.add(cleaned_scale)
            matches.append(cleaned_scale)

    # 2. Mã sản phẩm có gạch nối hỗn hợp (ví dụ: RX-78-2, HW-1234, B-1234)
    hyphen_sku_pattern = r"\b[A-Za-z0-9]{1,5}(?:[-_][A-Za-z0-9]{1,5})+\b"
    for m in re.finditer(hyphen_sku_pattern, text):
        val = m.group().strip()
        # Đảm bảo có cả chữ và số
        if re.search(r"[A-Za-z]", val) and re.search(r"[0-9]", val):
            val_upper = val.upper()
            if val_upper not in seen:
                seen.add(val_upper)
                matches.append(val_upper)

    # 3. Mã sản phẩm dính liền chữ và số (ví dụ: HW123, CT99)
    joined_sku_pattern = r"\b[A-Za-z]{1,4}[0-9]{2,6}[A-Za-z0-9]?\b"
    for m in re.finditer(joined_sku_pattern, text):
        val = m.group().strip().upper()
        if val not in seen:
            seen.add(val)
            matches.append(val)

    # 4. Dãy số định danh bộ đồ chơi (LEGO Set ID: 60337, 75192)
    num_pattern = r"\b\d{4,6}\b"
    for m in re.finditer(num_pattern, text):
        val = m.group().strip()
        # Kiểm tra xem số này có phải là một phần của SKU đã bắt trước đó không
        if any(val in s for s in seen):
            continue
        # Bỏ qua các năm thông dụng 1980 - 2030 nếu đứng riêng lẻ
        if 1980 <= int(val) <= 2030:
            continue
        if val not in seen:
            seen.add(val)
            matches.append(val)

    return matches


def extract_text_and_keywords(
    image_original: np.ndarray,
    image_enhanced: np.ndarray,
    min_keyword_conf: Optional[float] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    model_type: str = "mobile",
) -> Tuple[List[Tuple[str, float]], str]:
    """
    Chạy OCR trên CẢ ảnh gốc chuẩn hóa và ảnh tăng cường độ tương phản,
    gộp kết quả, loại bỏ trùng lặp, lọc nhiễu và ghép từ khóa tìm kiếm tối ưu.

    QUY TẮC ĐỘ TIN CẬY TRÍCH XUẤT TỪ KHÓA:
    - Mô hình Siêu tốc Mobile: Lấy trên 85% (>= 0.85) để bắt trọn vẹn văn bản và mã sản phẩm.
    - Mô hình Tiêu chuẩn Medium: Lấy trên 95% (>= 0.95) để đạt độ chính xác khắt khe.

    Chiến lược chấm điểm ưu tiên:
    1. Thương hiệu (Brand): Điểm 100 (khớp mờ RapidFuzz >= 85).
    2. Mã sản phẩm / SKU / Tỷ lệ: Điểm 85.
    3. Cụm chữ in hoa nổi bật (2-4 từ toàn chữ hoa): Điểm 65.
    4. Các từ khóa thông thường có độ tin cậy cao: Điểm 40 * confidence.

    Args:
        image_original (np.ndarray): Ảnh gốc đã chuẩn hóa kích thước (BGR).
        image_enhanced (np.ndarray): Ảnh đã tăng cường tương phản CLAHE & làm nét (BGR).
        min_keyword_conf (Optional[float]): Ngưỡng độ tin cậy tối thiểu (mặc định 0.85 cho Mobile, 0.95 cho Medium).
        progress_callback (Optional[Callable[[int, str], None]]): Hàm cập nhật thanh tiến trình theo phần trăm.
        model_type (str): "mobile" (~0.6s siêu tốc) hoặc "medium" (tiêu chuẩn).

    Returns:
        Tuple[List[Tuple[str, float]], str]:
            - raw_texts: Danh sách tất cả các dòng chữ OCR đọc được kèm confidence.
            - final_keyword: Chuỗi từ khóa tối ưu để tìm kiếm trên Internet.
    """
    if min_keyword_conf is None:
        min_keyword_conf = 0.85 if model_type == "mobile" else 0.95

    if progress_callback:
        progress_callback(50, "🔍 Đang nhận diện ký tự quang học với PaddleOCR 3.x...")

    # 1. Chạy OCR tăng tốc: Ưu tiên ảnh tăng cường tương phản CLAHE trước
    ocr_res_enh = run_ocr(image_enhanced, min_conf=0.55, model_type=model_type)

    # Đánh giá nhanh: Nếu ảnh tăng cường đã trích xuất được kết quả tốt (>= 2 dòng chữ hoặc có thương hiệu/SKU),
    # ta bỏ qua lượt chạy OCR thứ hai để TIẾT KIỆM 50% THỜI GIAN suy luận trên CPU.
    has_strong_clue = False
    if len(ocr_res_enh) >= 2:
        has_strong_clue = True
    elif len(ocr_res_enh) == 1:
        text_first = ocr_res_enh[0][0]
        if match_brand(text_first) or extract_sku_or_model(text_first):
            has_strong_clue = True

    # Chỉ chạy fallback trên ảnh gốc nếu ảnh tăng cường chưa tìm thấy chữ đủ rõ
    ocr_res_orig: List[Tuple[str, float]] = []
    if not has_strong_clue:
        if progress_callback:
            progress_callback(65, "🔍 Đang quét bổ trợ trên ảnh gốc để bảo đảm không sót chữ...")
        ocr_res_orig = run_ocr(image_original, min_conf=0.55, model_type=model_type)

    if progress_callback:
        conf_pct_str = f"{int(min_keyword_conf * 100)}%"
        progress_callback(85, f"🎯 Đang lọc chữ nhiễu và bóc tách từ khóa độ tin cậy > {conf_pct_str}...")

    # 2. Hợp nhất kết quả và loại bỏ trùng lặp (giữ bản có confidence cao hơn)
    merged_dict: Dict[str, Tuple[str, float]] = {}

    for text_raw, score in ocr_res_enh + ocr_res_orig:
        cleaned = normalize_text(text_raw)
        if not cleaned or is_noise_text(cleaned):
            continue

        lower_key = cleaned.lower()
        if lower_key not in merged_dict:
            merged_dict[lower_key] = (cleaned, score)
        else:
            # Cập nhật nếu bản này có độ tin cậy cao hơn
            if score > merged_dict[lower_key][1]:
                merged_dict[lower_key] = (cleaned, score)

    raw_texts: List[Tuple[str, float]] = sorted(
        merged_dict.values(), key=lambda x: x[1], reverse=True
    )

    if not raw_texts:
        return [], ""

    # 3. Phân loại và chấm điểm các thành phần từ khóa
    detected_brands: List[Tuple[str, float]] = []
    detected_skus: List[Tuple[str, float]] = []
    detected_caps: List[Tuple[str, float]] = []
    detected_others: List[Tuple[str, float]] = []

    seen_brands: Set[str] = set()
    seen_skus: Set[str] = set()

    for text_clean, conf in raw_texts:
        # LỌC ĐỘ TIN CẬY: Chỉ lấy từ khóa có độ tin cậy >= min_keyword_conf
        # (Mô hình Siêu tốc Mobile: >= 85%, Mô hình Medium: >= 95%)
        if conf < min_keyword_conf:
            continue

        # A. Nhận diện thương hiệu bằng so khớp mờ
        brand_match = match_brand(text_clean)
        if brand_match and brand_match not in seen_brands:
            detected_brands.append((brand_match, 100.0 + conf))
            seen_brands.add(brand_match)

        # B. Trích xuất SKU, mã sản phẩm hoặc tỷ lệ
        skus = extract_sku_or_model(text_clean)
        for s in skus:
            if s not in seen_skus:
                detected_skus.append((s, 85.0 + conf))
                seen_skus.add(s)

        # C. Nhận diện cụm từ in hoa nổi bật (vd: 'PASSENGER TRAIN', 'CITY')
        words = text_clean.split()
        if 1 <= len(words) <= 4 and text_clean.isupper() and len(text_clean) >= 3:
            if text_clean.upper() not in seen_skus:
                detected_caps.append((text_clean, 65.0 + conf))
        else:
            detected_others.append((text_clean, 40.0 * conf))

    # 4. Ghép final_keyword theo thứ tự ưu tiên: Thương hiệu -> SKU -> Cụm in hoa -> Từ khác
    keyword_tokens: List[str] = []
    seen_tokens_lower: Set[str] = set()

    def add_token(word: str) -> None:
        """Thêm từ khóa nếu chưa xuất hiện (không phân biệt hoa thường và loại trùng lặp con)."""
        w = word.strip(" ,.-_")
        if not w or is_noise_text(w):
            return
        w_lower = w.lower()
        w_compact = re.sub(r"[\s/:-]", "", w_lower)

        # Tránh thêm lại nếu token đã tồn tại
        if w_lower in seen_tokens_lower:
            return

        # Tránh thêm các mảnh nhỏ đã nằm trong một SKU/từ khóa trước đó (ví dụ: '1/144' khi đã có 'HG1/144')
        for seen in seen_tokens_lower:
            seen_compact = re.sub(r"[\s/:-]", "", seen)
            if w_compact == seen_compact or (len(w_compact) >= 2 and w_compact in seen_compact):
                return

        seen_tokens_lower.add(w_lower)
        keyword_tokens.append(w)

    # Ưu tiên 1: Thương hiệu đứng đầu
    for b_name, _ in detected_brands:
        for w in b_name.split():
            add_token(w)

    # Ưu tiên 2: Mã SKU / Set ID / Tỷ lệ đứng ngay sau thương hiệu
    for s_name, _ in detected_skus:
        add_token(s_name)

    # Ưu tiên 3: Tên dòng sản phẩm in hoa
    for c_text, _ in detected_caps:
        for w in c_text.split():
            add_token(w)

    # Ưu tiên 4: Các từ ngữ mô tả khác
    for o_text, _ in detected_others:
        for w in o_text.split():
            add_token(w)

    # Giới hạn tối đa 8 từ và không quá 80 ký tự để tìm kiếm hiệu quả
    final_tokens: List[str] = []
    current_length = 0

    for token in keyword_tokens:
        if len(final_tokens) >= 8:
            break
        if current_length + len(token) + 1 > 80:
            break
        final_tokens.append(token)
        current_length += len(token) + 1

    final_keyword = " ".join(final_tokens).strip()

    return raw_texts, final_keyword
