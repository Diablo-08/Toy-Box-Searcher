# ==============================================================================
# gemini_fallback.py - Hybrid Gemini Flash Fallback cho Toy Box Searcher
# ==============================================================================
"""
Module dự phòng thông minh sử dụng Gemini Flash để phân tích hộp đồ chơi
khi ảnh đầu vào bị mờ, góc khuất, hoặc PaddleOCR không trích xuất đủ từ khóa tin cậy.
"""

import os
import re
from typing import Any, List, Optional
import numpy as np
from PIL import Image
import streamlit as st
import google.generativeai as genai

# Danh sách mô hình Flash ưu tiên theo thứ tự tính tương thích
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]

_cached_model: Optional[genai.GenerativeModel] = None
_configured_key: Optional[str] = None


def get_api_key() -> Optional[str]:
    """
    Lấy Google Gemini API Key từ Streamlit secrets hoặc biến môi trường hệ thống.
    """
    # 1. Thử lấy từ Streamlit secrets
    try:
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            key = st.secrets["GOOGLE_API_KEY"]
            if key and isinstance(key, str) and key.strip():
                return key.strip()
    except Exception:
        pass

    # 2. Thử lấy từ biến môi trường
    env_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key

    return None


def get_gemini_model() -> Optional[genai.GenerativeModel]:
    """
    Khởi tạo và cấu hình mô hình Gemini Flash tối ưu.
    """
    global _cached_model, _configured_key

    api_key = get_api_key()
    if not api_key:
        return None

    if _cached_model is not None and _configured_key == api_key:
        return _cached_model

    try:
        genai.configure(api_key=api_key)
        _configured_key = api_key

        # Thử tìm mô hình khả dụng trong danh sách ưu tiên
        for model_name in MODEL_CANDIDATES:
            try:
                model = genai.GenerativeModel(model_name)
                _cached_model = model
                return model
            except Exception:
                continue

        # Mặc định sử dụng gemini-flash-latest
        model = genai.GenerativeModel("gemini-flash-latest")
        _cached_model = model
        return model
    except Exception as e:
        st.warning(f"Không thể khởi tạo Gemini client: {e}")
        return None


def extract_keywords_gemini(image: Any) -> List[str]:
    """
    Sử dụng Gemini Flash phân tích hình ảnh hộp đồ chơi để trích xuất:
    - Thương hiệu (Brand)
    - Tên sản phẩm / Dòng sản phẩm (Product line)
    - Mã SKU / Set number (nếu có)

    Args:
        image: PIL Image hoặc np.ndarray (ảnh OpenCV BGR hoặc RGB).

    Returns:
        List[str]: Danh sách từ khóa đã làm sạch, ví dụ: ['LEGO', 'Technic', '42151', 'Bugatti']
    """
    # Chuyển đổi định dạng ảnh sang PIL Image nếu cần
    pil_image: Optional[Image.Image] = None

    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, np.ndarray):
        try:
            # Nếu là mảng OpenCV BGR 3 kênh
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Đổi BGR -> RGB
                rgb_array = image[:, :, ::-1]
                pil_image = Image.fromarray(rgb_array)
            elif len(image.shape) == 2:
                # Ảnh xám
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image)
        except Exception:
            return []
    else:
        return []

    if pil_image is None:
        return []

    # Thu nhỏ ảnh nếu quá lớn để tối ưu tốc độ truyền tải
    max_dim = 1024
    if max(pil_image.size) > max_dim:
        pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    model = get_gemini_model()
    if model is None:
        return []

    prompt = """Phân tích hộp đồ chơi này. Trích xuất:
- Tên thương hiệu
- Tên sản phẩm / dòng sản phẩm  
- Mã số (SKU/set number nếu có)
Chỉ trả về danh sách từ khóa, cách nhau bằng dấu phẩy.
Ví dụ: LEGO, Technic, 42151, Bugatti"""

    try:
        response = None
        # Thử gọi mô hình hiện tại, nếu gặp lỗi không tìm thấy mô hình thì thử các candidate khác
        last_error = None
        for model_name in MODEL_CANDIDATES:
            try:
                active_model = genai.GenerativeModel(model_name)
                response = active_model.generate_content([prompt, pil_image])
                if response and response.text:
                    break
            except Exception as err:
                last_error = err
                continue

        if not response or not response.text:
            if last_error:
                st.warning(f"Gemini fallback thất bại: {last_error}")
            return []

        raw_text = response.text.strip()

        # Kiểm tra xem Gemini có trả lời kiểu phủ định (không có thông tin / ảnh mờ không đọc được)
        negation_markers = [
            "không có thông tin",
            "không có",
            "không tìm thấy",
            "không thể xác định",
            "không xác định",
            "không rõ",
            "chưa rõ",
            "unclear",
            "cannot identify",
            "no information",
            "none",
            "n/a",
        ]

        # Tách từ khóa theo dấu phẩy hoặc dòng mới
        parts = re.split(r"[,;\n\r]+", raw_text)
        keywords: List[str] = []
        seen = set()

        for p in parts:
            clean = p.strip(" \t\n\r*-\"':`•")
            if not clean or len(clean) < 2:
                continue
            clean_low = clean.lower()
            # Bỏ các dòng chú thích thừa hoặc cụm phủ định
            if clean_low.startswith("ví dụ") or clean_low.startswith("lưu ý"):
                continue
            if any(marker in clean_low for marker in negation_markers) and len(clean_low.split()) <= 4:
                continue
            if clean_low not in seen:
                seen.add(clean_low)
                keywords.append(clean)

        return keywords

    except Exception as e:
        st.warning(f"Gemini fallback thất bại: {e}")
        return []
