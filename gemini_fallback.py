# ==============================================================================
# gemini_fallback.py - Hybrid Gemini Flash Fallback cho Toy Box Searcher
# ==============================================================================
"""
Module dự phòng thông minh sử dụng Gemini Flash để phân tích hộp đồ chơi
khi ảnh đầu vào bị mờ, góc khuất, hoặc PaddleOCR không trích xuất đủ thương hiệu/từ khóa tin cậy.
Hỗ trợ cả Google GenAI SDK mới (google-genai) và thư viện cũ (google-generativeai) với cơ chế tự động thích ứng.
"""

import os
import re
from typing import Any, List, Optional
import numpy as np
from PIL import Image
import streamlit as st

# Thử import Google GenAI SDK mới nhất (google-genai), fallback sang google-generativeai nếu cần
USE_NEW_SDK = False
try:
    from google import genai
    USE_NEW_SDK = True
except Exception:
    try:
        import google.genai as genai
        USE_NEW_SDK = True
    except Exception:
        try:
            import google.generativeai as legacy_genai
            USE_NEW_SDK = False
        except Exception:
            genai = None
            legacy_genai = None

# Danh sách mô hình Flash ưu tiên
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-flash-latest",
]


def get_api_key() -> Optional[str]:
    """Lấy API key từ Streamlit secrets hoặc biến môi trường hệ thống."""
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


def extract_keywords_gemini(image: Any) -> List[str]:
    """
    Sử dụng Gemini Flash để phân tích trực quan hình ảnh hộp đồ chơi:
    - Nhận diện chính xác thương hiệu (kể cả logo uốn lượn như Hot Wheels, Bandai, LEGO, v.v.)
    - Tên dòng sản phẩm / series
    - Tên mẫu xe / nhân vật / sản phẩm
    - Mã số sản phẩm (SKU/Set number)
    """
    api_key = get_api_key()
    if not api_key:
        return []

    # Chuẩn hóa ảnh sang PIL Image
    pil_image: Optional[Image.Image] = None
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, np.ndarray):
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(image[:, :, ::-1])  # BGR sang RGB
            elif len(image.shape) == 2:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image)
        except Exception:
            return []
    else:
        return []

    if pil_image is None:
        return []

    # Thu nhỏ ảnh nếu quá lớn để tối ưu độ trễ
    max_dim = 1024
    if max(pil_image.size) > max_dim:
        pil_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

    prompt = """Bạn là chuyên gia thẩm định và tra cứu đồ chơi (Toy Specialist).
Hãy quan sát kỹ ảnh bao bì/hộp đồ chơi này (đặc biệt lưu ý logo thương hiệu, dòng sản phẩm, tên xe/nhân vật, mã SKU):
1. Tên thương hiệu (ví dụ: Hot Wheels, LEGO, Bandai, Tomica, Matchbox, Hasbro, Mattel, v.v.)
2. Tên dòng sản phẩm (ví dụ: Monster Trucks, Car Culture, Mainline, Technic, Speed Champions, v.v.)
3. Tên cụ thể của sản phẩm / mẫu xe / nhân vật (ví dụ: Bone Shaker, Bugatti Chiron, RX-78-2, v.v.)
4. Mã số sản phẩm / SKU nếu nhìn thấy trên vỏ hộp.

Chỉ trả về danh sách các từ khóa quan trọng nhất, cách nhau bằng dấu phẩy.
Ví dụ: Hot Wheels, Monster Trucks, Bone Shaker
KHÔNG viết câu dài dòng, KHÔNG giải thích."""

    negation_markers = [
        "không có thông tin", "không có", "không tìm thấy", "không thể xác định",
        "không xác định", "không rõ", "chưa rõ", "unclear", "cannot identify",
        "no information", "none", "n/a",
    ]

    # --- NHÁNH 1: SỬ DỤNG GOOGLE-GENAI MỚI (CLIENT-BASED) ---
    if USE_NEW_SDK and genai is not None:
        try:
            client = genai.Client(api_key=api_key)
            for model_name in MODEL_CANDIDATES:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, pil_image],
                    )
                    if response and getattr(response, "text", None):
                        raw_text = response.text.strip()
                        return _parse_gemini_keywords(raw_text, negation_markers)
                except Exception:
                    continue
        except Exception as err:
            st.warning(f"Gemini client lỗi: {err}")

    # --- NHÁNH 2: FALLBACK SANG GOOGLE-GENERATIVEAI CŨ NẾU CÓ ---
    elif legacy_genai is not None:
        try:
            legacy_genai.configure(api_key=api_key)
            for model_name in MODEL_CANDIDATES:
                try:
                    model = legacy_genai.GenerativeModel(model_name)
                    response = model.generate_content([prompt, pil_image])
                    if response and getattr(response, "text", None):
                        raw_text = response.text.strip()
                        return _parse_gemini_keywords(raw_text, negation_markers)
                except Exception:
                    continue
        except Exception as err:
            st.warning(f"Legacy Gemini client lỗi: {err}")

    return []


def _parse_gemini_keywords(raw_text: str, negation_markers: List[str]) -> List[str]:
    """Tách và chuẩn hóa danh sách từ khóa từ text trả về của Gemini."""
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
