# ==============================================================================
# gemini_fallback.py - Hybrid Gemini Flash Fallback cho Toy Box Searcher
# ==============================================================================
"""
Module dự phòng thông minh sử dụng Gemini Flash để phân tích hộp đồ chơi khi ảnh đầu vào bị mờ, góc khuất, hoặc PaddleOCR không trích xuất đủ từ khóa tin cậy.
"""

import os
import re
from typing import Any, List, Optional
import numpy as np
from PIL import Image
import streamlit as st
# Sử dụng SDK mới google-genai (client‑based)
from google import genai

# Danh sách mô hình Flash ưu tiên
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]

def get_api_key() -> Optional[str]:
    """Lấy API key từ Streamlit secrets hoặc biến môi trường."""
    try:
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            key = st.secrets["GOOGLE_API_KEY"]
            if key and isinstance(key, str) and key.strip():
                return key.strip()
    except Exception:
        pass
    env_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key
    return None

def _get_client() -> Optional[genai.Client]:
    """Khởi tạo client một lần, dùng API key đã cấu hình."""
    api_key = get_api_key()
    if not api_key:
        st.warning("Chưa cấu hình GOOGLE_API_KEY cho Gemini.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.warning(f"Không thể khởi tạo Gemini client: {e}")
        return None

def extract_keywords_gemini(image: Any) -> List[str]:
    """Sử dụng Gemini Flash để trích xuất từ khóa từ ảnh.
    Trả về danh sách các từ khóa đã làm sạch.
    """
    # Chuyển đổi ảnh sang PIL nếu cần
    pil_image: Optional[Image.Image] = None
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, np.ndarray):
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB
            elif len(image.shape) == 2:
                pil_image = Image.fromarray(image)  # Grayscale
            else:
                pil_image = Image.fromarray(image)
        except Exception:
            return []
    else:
        return []
    if pil_image is None:
        return []

    # Thu nhỏ ảnh nếu quá lớn để giảm latency
    max_dim = 1024
    if max(pil_image.size) > max_dim:
        pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    client = _get_client()
    if client is None:
        return []

    prompt = """Phân tích hộp đồ chơi này. Trích xuất:
    - Tên thương hiệu
    - Tên sản phẩm / dòng sản phẩm
    - Mã số (SKU/set number nếu có)
    Chỉ trả về danh sách từ khóa, cách nhau bằng dấu phẩy.
    Ví dụ: LEGO, Technic, 42151, Bugatti"""

    for model_name in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, pil_image],
            )
            if response and getattr(response, "text", None):
                raw_text = response.text.strip()
                # Các marker để loại bỏ câu phủ định ngắn gọn
                negation_markers = [
                    "không có thông tin", "không có", "không tìm thấy", "không thể xác định",
                    "không xác định", "không rõ", "chưa rõ", "unclear", "cannot identify",
                    "no information", "none", "n/a",
                ]
                parts = re.split(r"[,;\n\r]+", raw_text)
                keywords: List[str] = []
                seen = set()
                for p in parts:
                    clean = p.strip(" \t\n\r*-\"':`•")
                    if not clean or len(clean) < 2:
                        continue
                    low = clean.lower()
                    if any(m in low for m in negation_markers) and len(low.split()) <= 4:
                        continue
                    if low not in seen:
                        seen.add(low)
                        keywords.append(clean)
                return keywords
        except Exception:
            # Nếu model này lỗi, thử model tiếp theo
            continue
    st.warning("Gemini fallback thất bại: không có mô hình nào trả về kết quả.")
    return []
