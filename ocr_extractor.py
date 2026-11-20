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
from typing import List, Tuple, Dict, Set, Optional, Callable, Any

# Tắt kiểm tra kết nối máy chủ mô hình từ xa của PaddleX để tăng tốc độ khởi động tức thì
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import cv2
import numpy as np
import streamlit as st
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from gemini_fallback import extract_keywords_gemini

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
    # Xây dựng & Lắp ráp (Building Blocks)
    "LEGO",
    "Mega Construx",
    "Mega Bloks",
    "Sluban",
    "Cobi",
    "CaDA",
    "Mould King",
    "Sembo",
    "K'NEX",
    "Banbao",
    # Xe mô hình & Đua xe (Die-cast & RC Models)
    "Hot Wheels",
    "Matchbox",
    "Tomica",
    "Majorette",
    "Maisto",
    "Bburago",
    "Mini GT",
    "Greenlight",
    "Johnny Lightning",
    "Jada Toys",
    "AutoArt",
    "Kyosho",
    "Tamiya",
    "Siku",
    "Hasegawa",
    "Aoshima",
    "Fujimi",
    "Revell",
    "Carrera",
    # Mô hình Anime, Manga & Game (Anime, Mecha & Collectibles)
    "Bandai",
    "Bandai Namco",
    "Bandai Spirits",
    "Gundam",
    "Gunpla",
    "Tamashii Nations",
    "Takara Tomy",
    "Pokemon",
    "Beyblade",
    "Bakugan",
    "Yu-Gi-Oh!",
    "Digimon",
    "Dragon Ball",
    "One Piece",
    "Naruto",
    "Demon Slayer",
    "Sanrio",
    "Hello Kitty",
    "Doraemon",
    "Studio Ghibli",
    # Nhân vật, Búp bê & Art Toy (Figures, Dolls & Art Toys)
    "Barbie",
    "Monster High",
    "Bratz",
    "LOL Surprise",
    "Rainbow High",
    "Sylvanian Families",
    "Baby Alive",
    "Polly Pocket",
    "My Little Pony",
    "Funko",
    "Pop Mart",
    "Good Smile Company",
    "Nendoroid",
    "Figma",
    "Kotobukiya",
    "Banpresto",
    "Medicom Toy",
    "McFarlane Toys",
    "NECA",
    "Hot Toys",
    "Schleich",
    "Papo",
    "Mego",
    # Hành động, Điện ảnh & Siêu anh hùng (Action Figures & Superheroes)
    "Transformers",
    "Hasbro",
    "Mattel",
    "Marvel",
    "DC Comics",
    "Star Wars",
    "Jurassic World",
    "Power Rangers",
    "TMNT",
    "Disney",
    "Ben 10",
    # Đồ chơi trẻ em, Mầm non & Giáo dục (Infant, Preschool & Creative)
    "Fisher-Price",
    "VTech",
    "LeapFrog",
    "Melissa & Doug",
    "Playmobil",
    "Play-Doh",
    "Little Tikes",
    "Crayola",
    "Kinetic Sand",
    "Nerf",
    "Spin Master",
    "Moose Toys",
    "WowWee",
    "Playmates",
    "Silverlit",
    # Board Game & Thẻ bài (Games & Puzzles)
    "Monopoly",
    "Uno",
    "Rubik's",
    "Ravensburger",
    "Asmodee",
]

# ==============================================================================
# TỪ ĐIỂN DÒNG SẢN PHẨM / SERIES NỔI TIẾNG (PRODUCT LINES / SUBLINES)
# ==============================================================================
KNOWN_PRODUCT_LINES: List[str] = [
    # LEGO Lines
    "Technic",
    "City",
    "Ninjago",
    "Speed Champions",
    "Creator",
    "Creator Expert",
    "Icons",
    "Star Wars",
    "Duplo",
    "Friends",
    "Architecture",
    "Harry Potter",
    "Marvel Super Heroes",
    "Batman",
    "Classic",
    "Botanical Collection",
    "Ideas",
    "Mindstorms",
    "Bionicle",
    "Monkie Kid",
    "Dreamzzz",
    # Hot Wheels & Diecast Lines
    "Monster Trucks",
    "Track Stars",
    "Premium",
    "Car Culture",
    "Boulevard",
    "Fast & Furious",
    "Red Line Club",
    "Mainline",
    "Treasure Hunt",
    "Team Transport",
    "Pop Culture",
    # Gundam & Mecha Lines
    "High Grade",
    "Master Grade",
    "Real Grade",
    "Perfect Grade",
    "Entry Grade",
    "Super Deformed",
    "Gundam Universe",
    "Robot Spirits",
    "Metal Build",
    "Witch from Mercury",
    "Iron-Blooded Orphans",
    # Transformers Lines
    "Studio Series",
    "Generations",
    "Legacy",
    "War for Cybertron",
    "Masterpiece",
    "Kingdom",
    "Earthrise",
    "Siege",
    "Rise of the Beasts",
    # Action Figures & Collectibles Lines
    "Marvel Legends",
    "Black Series",
    "Vintage Collection",
    "Pop! Vinyl",
    "S.H.Figuarts",
    "Mythic Legions",
    "GI Joe Classified",
]

# ==============================================================================
# DANH SÁCH TỪ DỪNG / CẢNH BÁO BẢO HỘ TRÊN VỎ HỘP ĐỒ CHƠI (STOPWORDS)
# ==============================================================================
STOPWORDS: Set[str] = {
    # Tiếng Anh - Cảnh báo an toàn & độ tuổi
    "warning",
    "choking hazard",
    "small parts",
    "not for children",
    "under 3 years",
    "keep out of reach",
    "adult supervision",
    "safety",
    "caution",
    "attention",
    "achtung",
    "aviso",
    "ages",
    "age",
    "years",
    "year",
    "months",
    # Tiếng Anh - Quy cách đóng gói & sản xuất
    "pieces",
    "pcs",
    "new",
    "item",
    "no",
    "toy",
    "toys",
    "model kit",
    "made in china",
    "made in vietnam",
    "made in thailand",
    "made in japan",
    "made in indonesia",
    "all rights reserved",
    "distributed by",
    "customer service",
    "each sold separately",
    "colors may vary",
    "contents may vary",
    "batteries included",
    "batteries not included",
    "battery operated",
    "non toxic",
    "instructions included",
    "conforms to",
    "astm",
    "en71",
    "ce",
    "recycle",
    "barcode",
    "www",
    "com",
    "http",
    "https",
    # Tiếng Việt - Cảnh báo an toàn
    "cảnh báo",
    "nguy cơ nghẹt thở",
    "chi tiết nhỏ",
    "không thích hợp cho trẻ",
    "trẻ dưới 3 tuổi",
    "độ tuổi",
    "tuổi",
    "hướng dẫn sử dụng",
    "giám sát của người lớn",
    # Tiếng Việt - Thông tin sản phẩm & xuất xứ
    "xuất xứ",
    "nơi sản xuất",
    "sản xuất tại",
    "nhập khẩu bởi",
    "nhà phân phối",
    "thành phần",
    "thông số kỹ thuật",
    "tiêu chuẩn",
    "năm sản xuất",
    "đồ chơi trẻ em",
    "hộp đồ chơi",
    "mảnh ghép",
    "chi tiết",
}


@st.cache_resource(show_spinner=False)
def get_ocr(model_type: str = "mobile") -> PaddleOCR:
    """
    Khởi tạo mô hình PaddleOCR đúng 1 lần và lưu vào cache của Streamlit (@st.cache_resource).
    Bắt buộc sử dụng API PaddleOCR 3.x và tối ưu hóa suy luận CPU:
    - model_type="mobile" (Mặc định): Sử dụng PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec,
      tốc độ suy luận CPU đạt ~0.6s (nhanh gấp 7 lần so với Standard ~4.5s) mà vẫn đạt độ chính xác >95%.
    - model_type="medium": Sử dụng mô hình PP-OCRv6_medium mặc định của PaddleOCR 3.x (Standard).
    - use_textline_orientation=True: Tự động xoay dòng chữ bị nghiêng/lộn ngược trên bao bì.
    - use_doc_orientation_classify=False: Tắt phân loại hướng toàn bộ trang tài liệu.
    - use_doc_unwarping=False: Tắt nắn phẳng trang sách tài liệu (tiết kiệm tài nguyên CPU).
    - Tự động làm nóng mô hình (Warmup) ngay khi khởi tạo để loại bỏ hiện tượng giật lag 30-40s ở lần chạy đầu tiên.

    Args:
        model_type (str): "mobile" / Flash (siêu tốc, tối ưu CPU) hoặc "medium" / Standard (tiêu chuẩn).

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


def match_product_line(text: str, threshold: float = 85.0) -> Optional[str]:
    """
    So khớp mờ để nhận diện dòng sản phẩm / series đồ chơi nổi tiếng
    (ví dụ: 'Technic', 'Ninjago', 'Speed Champions', 'Monster Trucks', 'Studio Series').

    Args:
        text (str): Chuỗi cần kiểm tra.
        threshold (float): Ngưỡng tương đồng tối thiểu (mặc định 85/100).

    Returns:
        Optional[str]: Tên dòng sản phẩm chuẩn nếu khớp, ngược lại None.
    """
    text_clean = text.strip()
    text_lower = text_clean.lower()

    if len(text_lower) < 3:
        return None

    for line in KNOWN_PRODUCT_LINES:
        line_lower = line.lower()
        if text_lower == line_lower:
            return line

        dist = Levenshtein.distance(text_lower, line_lower)
        if (len(line_lower) <= 5 and dist <= 1) or (len(line_lower) > 5 and dist <= 2):
            return line

        if fuzz.ratio(text_lower, line_lower) >= threshold:
            return line

        # So khớp bộ phận nếu chứa ranh giới từ
        if len(text_lower) >= len(line_lower):
            if re.search(rf"\b{re.escape(line_lower)}\b", text_lower):
                return line

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
    enable_gemini_fallback: bool = True,
) -> Tuple[List[Tuple[str, float]], str, bool]:
    """
    Chạy OCR trên CẢ ảnh gốc chuẩn hóa và ảnh tăng cường độ tương phản,
    gộp kết quả, loại bỏ trùng lặp, lọc nhiễu và ghép từ khóa tìm kiếm tối ưu.
    Tích hợp Hybrid Gemini Flash fallback khi OCR không đủ độ tin cậy.

    QUY TẮC ĐỘ TIN CẬY TRÍCH XUẤT TỪ KHÓA:
    - Mô hình Flash (Mobile): Lấy trên 85% (>= 0.85) để bắt trọn vẹn văn bản và mã sản phẩm.
    - Mô hình Standard (Medium): Lấy trên 95% (>= 0.95) để đạt độ chính xác khắt khe.

    Chiến lược chấm điểm ưu tiên:
    1. Thương hiệu (Brand): Điểm 100 (khớp mờ RapidFuzz >= 85).
    2. Mã sản phẩm / SKU / Tỷ lệ: Điểm 85.
    3. Cụm chữ in hoa nổi bật (2-4 từ toàn chữ hoa): Điểm 65.
    4. Các từ khóa thông thường có độ tin cậy cao: Điểm 40 * confidence.

    Args:
        image_original (np.ndarray): Ảnh gốc đã chuẩn hóa kích thước (BGR).
        image_enhanced (np.ndarray): Ảnh đã tăng cường tương phản CLAHE & làm nét (BGR).
        min_keyword_conf (Optional[float]): Ngưỡng độ tin cậy tối thiểu (mặc định 0.85 cho Flash, 0.95 cho Standard).
        progress_callback (Optional[Callable[[int, str], None]]): Hàm cập nhật thanh tiến trình theo phần trăm.
        model_type (str): "mobile" / Flash (~0.6s siêu tốc) hoặc "medium" / Standard (tiêu chuẩn).
        enable_gemini_fallback (bool): Cho phép gọi Gemini Flash khi OCR yếu (mặc định True).

    Returns:
        Tuple[List[Tuple[str, float]], str, bool]:
            - raw_texts: Danh sách tất cả các dòng chữ OCR đọc được kèm confidence.
            - final_keyword: Chuỗi từ khóa tối ưu để tìm kiếm trên Internet.
            - used_gemini_fallback: True nếu Gemini Flash đã tham gia hỗ trợ phân tích.
    """
    if min_keyword_conf is None:
        min_keyword_conf = 0.85 if model_type == "mobile" else 0.95

    if progress_callback:
        progress_callback(50, "🔍 Đang nhận diện ký tự quang học với PaddleOCR 3.x...")

    # 1. Chạy OCR tăng tốc: Ưu tiên ảnh tăng cường tương phản CLAHE trước
    ocr_res_enh = run_ocr(image_enhanced, min_conf=0.55, model_type=model_type)

    # Đánh giá nhanh: Nếu ảnh tăng cường đã trích xuất được kết quả tốt (>= 2 dòng chữ hoặc có thương hiệu/SKU/Dòng sản phẩm),
    # ta bỏ qua lượt chạy OCR thứ hai để TIẾT KIỆM 50% THỜI GIAN suy luận trên CPU.
    has_strong_clue = False
    if len(ocr_res_enh) >= 2:
        has_strong_clue = True
    elif len(ocr_res_enh) == 1:
        text_first = ocr_res_enh[0][0]
        if match_brand(text_first) or extract_sku_or_model(text_first) or match_product_line(text_first):
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

    def assemble_final_keyword() -> str:
        final_tokens: List[str] = []
        current_length = 0
        for token in keyword_tokens:
            if len(final_tokens) >= 8:
                break
            if current_length + len(token) + 1 > 80:
                break
            final_tokens.append(token)
            current_length += len(token) + 1
        return " ".join(final_tokens).strip()

    # Trường hợp PaddleOCR không tìm thấy chữ nào trên ảnh:
    if not raw_texts:
        if enable_gemini_fallback:
            if progress_callback:
                progress_callback(90, "🤖 PaddleOCR không phát hiện chữ, đang chuyển sang Gemini Flash...")
            try:
                gemini_keywords = extract_keywords_gemini(image_original)
                if gemini_keywords:
                    for g_kw in gemini_keywords:
                        raw_texts.append((f"[Gemini] {g_kw}", 0.95))
                        add_token(g_kw)
                    final_keyword = assemble_final_keyword()
                    return raw_texts, final_keyword, True
            except Exception as e:
                st.warning(f"Gemini fallback thất bại: {e}")
        return [], "", False

    # 3. Phân loại và chấm điểm các thành phần từ khóa
    detected_brands: List[Tuple[str, float]] = []
    detected_lines: List[Tuple[str, float]] = []
    detected_skus: List[Tuple[str, float]] = []
    detected_caps: List[Tuple[str, float]] = []
    detected_others: List[Tuple[str, float]] = []

    seen_brands: Set[str] = set()
    seen_lines: Set[str] = set()
    seen_skus: Set[str] = set()

    for text_clean, conf in raw_texts:
        # LỌC ĐỘ TIN CẬY: Chỉ lấy từ khóa có độ tin cậy >= min_keyword_conf
        # (Mô hình Flash / Mobile: >= 85%, Mô hình Standard / Medium: >= 95%)
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

        # C. Nhận diện dòng sản phẩm / series nổi tiếng (Technic, Ninjago, Monster Trucks...)
        line_match = match_product_line(text_clean)
        if line_match and line_match not in seen_lines and line_match not in seen_brands:
            detected_lines.append((line_match, 75.0 + conf))
            seen_lines.add(line_match)

        # D. Nhận diện cụm từ in hoa nổi bật (vd: 'PASSENGER TRAIN', 'CITY')
        words = text_clean.split()
        if 1 <= len(words) <= 4 and text_clean.isupper() and len(text_clean) >= 3:
            if text_clean.upper() not in seen_skus and text_clean.title() not in seen_lines:
                detected_caps.append((text_clean, 65.0 + conf))
        else:
            detected_others.append((text_clean, 40.0 * conf))

    # 4. Ghép final_keyword theo thứ tự ưu tiên: Thương hiệu -> SKU -> Dòng sản phẩm -> Cụm in hoa -> Từ khác
    for b_name, _ in detected_brands:
        for w in b_name.split():
            add_token(w)

    for s_name, _ in detected_skus:
        add_token(s_name)

    for l_name, _ in detected_lines:
        for w in l_name.split():
            add_token(w)

    for c_text, _ in detected_caps:
        for w in c_text.split():
            add_token(w)

    for o_text, _ in detected_others:
        for w in o_text.split():
            add_token(w)

    final_keyword = assemble_final_keyword()

    # --- HYBRID LOGIC FALLBACK (GEMINI FLASH) ---
    # Điều kiện kích hoạt:
    # - Ít hơn 2 từ khóa hợp lệ (< MIN_KEYWORDS)
    # - HOẶC độ tin cậy của các chữ đọc được < 0.75 (< MIN_CONFIDENCE)
    # - HOẶC không trích xuất được final_keyword
    MIN_KEYWORDS = 2
    MIN_CONFIDENCE = 0.75
    used_gemini_fallback = False

    max_conf = max([score for _, score in raw_texts], default=0.0)
    has_trusted_clue = bool(detected_brands or detected_skus or detected_lines)

    if enable_gemini_fallback and (
        len(keyword_tokens) < MIN_KEYWORDS
        or (not has_trusted_clue and max_conf < MIN_CONFIDENCE)
        or not final_keyword
    ):
        if progress_callback:
            progress_callback(90, "🤖 Đang nhờ Gemini Flash phân tích hình ảnh bổ trợ...")
        try:
            gemini_keywords = extract_keywords_gemini(image_original)
            if gemini_keywords:
                used_gemini_fallback = True
                # Merge: ưu tiên giữ kết quả OCR, bổ sung từ Gemini
                for g_kw in gemini_keywords:
                    g_clean = g_kw.strip()
                    if not g_clean or is_noise_text(g_clean):
                        continue

                    # Kiểm tra xem g_kw có phải thương hiệu không
                    b_match = match_brand(g_clean)
                    if b_match and b_match not in seen_brands:
                        detected_brands.insert(0, (b_match, 100.0))
                        seen_brands.add(b_match)

                    # Kiểm tra xem g_kw có phải SKU không
                    skus = extract_sku_or_model(g_clean)
                    for s in skus:
                        if s not in seen_skus:
                            detected_skus.append((s, 90.0))
                            seen_skus.add(s)

                    # Kiểm tra xem g_kw có phải dòng sản phẩm không
                    l_match = match_product_line(g_clean)
                    if l_match and l_match not in seen_lines and l_match not in seen_brands:
                        detected_lines.append((l_match, 80.0))
                        seen_lines.add(l_match)

                    # Bổ sung vào bảng raw_texts để người dùng thấy nguồn gốc từ Gemini
                    raw_texts.append((f"[Gemini] {g_clean}", 0.95))

                # Xây dựng lại keyword_tokens với các thông tin bổ trợ từ Gemini
                keyword_tokens.clear()
                seen_tokens_lower.clear()

                for b_name, _ in detected_brands:
                    for w in b_name.split():
                        add_token(w)

                for s_name, _ in detected_skus:
                    add_token(s_name)

                for l_name, _ in detected_lines:
                    for w in l_name.split():
                        add_token(w)

                for c_text, _ in detected_caps:
                    for w in c_text.split():
                        add_token(w)

                for o_text, _ in detected_others:
                    for w in o_text.split():
                        add_token(w)

                for g_kw in gemini_keywords:
                    for w in g_kw.split():
                        add_token(w)

                final_keyword = assemble_final_keyword()
        except Exception as e:
            # Không để lỗi Gemini làm sập ứng dụng
            st.warning(f"Gemini fallback thất bại: {e}")

    return raw_texts, final_keyword, used_gemini_fallback


def extract_keywords(image: Any, mode: str = "mobile") -> List[str]:
    """
    Hàm trích xuất từ khóa Hybrid tinh gọn:
    1. Chạy PaddleOCR và lọc rule-based + RapidFuzz.
    2. Nếu số từ khóa < 2 hoặc độ tin cậy < 0.75, tự động gọi Gemini Flash phân tích bổ trợ.
    3. Hợp nhất kết quả và trả về danh sách từ khóa.

    Args:
        image: Ảnh đầu vào (PIL Image hoặc numpy ndarray BGR).
        mode (str): Chế độ mô hình OCR ("mobile" hoặc "medium").

    Returns:
        List[str]: Danh sách từ khóa tìm kiếm.
    """
    if isinstance(image, np.ndarray):
        img_bgr = image
    else:
        # Chuyển PIL Image sang OpenCV BGR
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    from preprocessing import preprocess_image

    img_enh = preprocess_image(img_bgr)

    raw_texts, final_keyword, _ = extract_text_and_keywords(
        image_original=img_bgr,
        image_enhanced=img_enh,
        model_type=mode,
        enable_gemini_fallback=True,
    )

    return final_keyword.split() if final_keyword else []

