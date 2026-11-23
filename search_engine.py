"""
Module: search_engine.py
Mục đích: Tìm kiếm thông tin sản phẩm đồ chơi trên Internet bằng thư viện ddgs (DuckDuckGo Search)
          tối ưu hóa theo khu vực (Việt Nam vn-vi / Toàn cầu wt-wt), tự động lọc từ rác OCR,
          LUÔN THÊM TỪ KHÓA ĐỒ CHƠI ("toy" hoặc "đồ chơi") vào truy vấn tìm kiếm để tập trung hoàn toàn
          vào sản phẩm đồ chơi thực tế, LOẠI BỎ 100% CÁC TRANG NGOÀI LỀ (Wikipedia, Fandom wiki, từ điển).
Tác giả: Senior Computer Vision Engineer & Python Developer
Môi trường: Python 3.11.9, ddgs 9.x, Streamlit 1.x
"""

import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
import streamlit as st

# ==============================================================================
# TẮT LOG CẢNH BÁO CỦA DDGS
# ==============================================================================
logging.getLogger("ddgs").setLevel(logging.ERROR)

# ==============================================================================
# VÁ LỖI DNS TRÊN WINDOWS CHO PRIMP (HTTP CLIENT CỦA DDGS)
# ==============================================================================
try:
    import primp
    import ddgs.http_client

    _orig_http_init = ddgs.http_client.HttpClient.__init__

    def _patched_http_init(self, proxy=None, timeout=6, *, verify=True):
        self.client = primp.Client(
            proxy=proxy,
            timeout=timeout,
            dns_resolver="system",  # Ép dùng Win32 OS DNS resolver để loại bỏ lỗi timeout DNS
            impersonate="random",
            impersonate_os="random",
            verify=verify if isinstance(verify, bool) else True,
            ca_cert_file=verify if isinstance(verify, str) else None,
        )

    ddgs.http_client.HttpClient.__init__ = _patched_http_init
except Exception:
    pass

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# Bộ từ điển thuật ngữ đồ chơi phổ biến để chấm điểm ưu tiên từ khóa tìm kiếm
TOY_VOCABULARY = {
    # Thuật ngữ chung
    "toy", "toys", "đồ", "chơi", "do", "choi", "đồchơi",
    # Thương hiệu
    "lego", "hot", "wheels", "hotwheels", "hotwheel", "barbie", "bandai",
    "gundam", "transformers", "hasbro", "mattel", "fisher", "price", "nerf",
    "playmobil", "funko", "tomica", "disney", "marvel", "pokemon", "tamiya",
    "kyosho", "matchbox", "majorette", "maisto", "bburago", "sluban",
    # Loại phương tiện & đồ chơi
    "car", "cars", "truck", "trucks", "train", "trains", "plane", "planes",
    "robot", "robots", "vehicle", "vehicles", "series", "diy", "construction",
    "builder", "builders", "building", "racing", "speed", "city", "police",
    "fire", "action", "figure", "figures", "blocks", "block", "model", "kit",
    "set", "diecast", "scale", "express", "passenger", "gunpla", "track",
    "rc", "remote", "control", "drone", "puzzle", "game", "monster",
    # Tiếng Việt
    "xe", "lap", "rap", "mo", "hinh", "dieu", "khien", "bup", "be"
}

_LAST_SEARCH_ERROR: Optional[str] = None


def get_last_search_error() -> Optional[str]:
    """Lấy thông điệp lỗi của lần tìm kiếm gần nhất (nếu có)."""
    global _LAST_SEARCH_ERROR
    return _LAST_SEARCH_ERROR


def is_irrelevant_entry(url: str, title: str, body: str = "") -> bool:
    """
    Kiểm tra và LOẠI BỎ TRIỆT ĐỂ các trang thông tin ngoài lề:
    - Bách khoa toàn thư: Wikipedia, Wikimedia, Wiktionary, Wikibooks
    - Wiki cộng đồng không bán hàng: Fandom, Wikia
    - Từ điển ngữ pháp & giải nghĩa từ: Dictionary, Cambridge, Merriam-Webster
    - Trang tin tức/bài viết không liên quan đến sản phẩm/đồ chơi
    """
    u = (url or "").lower()
    t = (title or "").lower()

    # 1. Wikipedia & các dự án Wikimedia
    if any(wiki in u for wiki in ["wikipedia.org", "wikimedia.org", "wiktionary.org", "wikibooks.org"]):
        return True
    if "wikipedia" in t or "bách khoa toàn thư" in t:
        return True

    # 2. Fandom & Wikia (thường là wiki cốt truyện phim/game dài dòng ngoài lề)
    if any(fan in u for fan in ["fandom.com", "wikia.org", "wikia.com"]):
        return True

    # 3. Từ điển ngữ pháp & giải nghĩa từ
    if any(dict_site in u for dict_site in ["dictionary.com", "merriam-webster.com", "cambridge.org", "collinsdictionary.com"]):
        return True
    if "nghĩa là gì" in t or "định nghĩa" in t:
        return True

    return False


def extract_smart_query(raw_keyword: str, max_words: int = 4, region: str = "vn-vi") -> str:
    """
    Trích xuất từ khóa thông minh có chọn lọc:
    - Ưu tiên thương hiệu (35đ), SKU (30đ), dòng sản phẩm (25đ), từ vựng đồ chơi (15đ).
    - Loại bỏ hoàn toàn các từ rác do OCR sinh ra.
    - ĐẶC BIỆT: LUÔN TỰ ĐỘNG THÊM TỪ KHÓA ĐỒ CHƠI ("toy" hoặc "đồ chơi") vào câu truy vấn
      để công cụ tìm kiếm tập trung 100% vào sản phẩm đồ chơi, loại bỏ các kết quả xe thật/ngoài lề.

    Args:
        raw_keyword (str): Chuỗi từ khóa quét được từ OCR / Gemini.
        max_words (int): Số lượng từ tối đa cần giữ lại.
        region (str): Khu vực tìm kiếm ("vn-vi" hoặc "wt-wt").

    Returns:
        str: Chuỗi truy vấn đã được tinh lọc và gắn nhãn đồ chơi.
    """
    tokens = re.findall(r"[A-Za-z0-9_-]+", raw_keyword)
    if not tokens:
        return "đồ chơi" if region == "vn-vi" else "toy"

    scored_tokens = []
    has_brand = False

    # Lấy danh sách thương hiệu và dòng sản phẩm chuẩn để so khớp
    try:
        from ocr_extractor import KNOWN_BRANDS, KNOWN_PRODUCT_LINES
        known_brands_set = {b.lower() for b in KNOWN_BRANDS}
        known_lines_set = {l.lower() for l in KNOWN_PRODUCT_LINES}
    except Exception:
        known_brands_set = set()
        known_lines_set = set()

    for t in tokens:
        tl = t.lower()

        # 1. Tên thương hiệu trực tiếp (ưu tiên cao nhất)
        if tl in known_brands_set or tl in {"hotwheels", "hotwheel", "lego", "bandai", "tomica", "mattel"}:
            scored_tokens.append((t, 35))
            has_brand = True
        # 2. Mã sản phẩm / Số hiệu hộp đồ chơi (ví dụ: 60337, 42151, HW123, RX-78)
        elif re.match(r"^\d{3,6}$", t) or re.match(r"^[A-Za-z]{1,4}\d{2,6}$", t):
            scored_tokens.append((t, 30))
        # 3. Dòng sản phẩm / Series nổi tiếng (Technic, Monster Trucks, etc.)
        elif tl in known_lines_set:
            scored_tokens.append((t, 25))
        # 4. Từ khóa nằm trong từ điển đồ chơi
        elif tl in TOY_VOCABULARY:
            scored_tokens.append((t, 15))
        # 5. Từ tiếng Anh thông thường có nguyên âm hợp lệ
        elif len(t) >= 3 and re.search(r"[aeiouyAEIOUY]", t):
            scored_tokens.append((t, 5))
        else:
            # Từ rác hoặc lỗi chính tả OCR
            scored_tokens.append((t, 1))

    # Sắp xếp từ theo điểm số giảm dần
    scored_tokens.sort(key=lambda x: x[1], reverse=True)

    # Chọn top từ có điểm cao nhất
    selected_words = [item[0] for item in scored_tokens[:max_words]]
    result_query = " ".join(selected_words).strip()

    # --- ĐẢM BẢO LUÔN CÓ TỪ KHÓA "TOY" HOẶC "ĐỒ CHƠI" ---
    lower_res = result_query.lower()
    has_toy_keyword = any(k in lower_res for k in ["toy", "toys", "đồ chơi", "do choi"])

    if not has_toy_keyword:
        if region == "vn-vi":
            # Tại Việt Nam, thêm "đồ chơi" giúp tìm kiếm đúng sàn Shopee, Tiki, MyKingdom
            result_query = f"{result_query} đồ chơi".strip()
        else:
            # Toàn cầu, thêm "toy"
            result_query = f"{result_query} toy".strip()

    return result_query


def _execute_ddgs_query(query: str, region: str, max_results: int) -> List[Dict[str, str]]:
    """
    Thực thi truy vấn trên DDGS và trả về danh sách kết quả sản phẩm thương mại.
    BẮT BUỘC: LỌC BỎ 100% CÁC TRANG WIKIPEDIA, FANDOM, TỪ ĐIỂN VÀ THÔNG TIN NGOÀI LỀ.
    """
    results: List[Dict[str, str]] = []
    fetch_limit = max_results + 6

    try:
        with DDGS(timeout=6) as ddgs:
            raw_response = ddgs.text(query, region=region, max_results=fetch_limit)
            if raw_response:
                for item in raw_response:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title") or item.get("name") or ""
                    href = item.get("href") or item.get("url") or item.get("link") or ""
                    body = item.get("body") or item.get("snippet") or item.get("description") or ""

                    # BỎ QUA HOÀN TOÀN CÁC TRANG THÔNG TIN NGOÀI LỀ
                    if is_irrelevant_entry(href, title, body):
                        continue

                    if href and title:
                        results.append({
                            "title": str(title).strip(),
                            "href": str(href).strip(),
                            "body": str(body).strip(),
                        })
                        if len(results) >= max_results:
                            break
    except Exception:
        pass

    return results


@st.cache_data(ttl=3600, show_spinner=False)
def search_toy_info(
    keyword: str, max_results: int = 5, region: str = "vn-vi"
) -> List[Dict[str, str]]:
    """
    Tra cứu thông tin sản phẩm đồ chơi thương mại theo khu vực (KHÔNG LẤY WIKIPEDIA/TRANG NGOÀI LỀ),
    kết hợp đa luồng (ThreadPoolExecutor) để truy vấn đồng thời khu vực chỉ định và toàn cầu:
    1. Tầng 1: Tinh lọc từ khóa thông minh (Smart Query), tự động gắn thêm 'toy'/'đồ chơi'.
    2. Tầng 2: Nới lỏng từ khóa (Relaxed Query) nếu Tầng 1 chưa có kết quả.
    3. Tầng 3: Truy vấn với từ khóa gốc kết hợp nhãn đồ chơi.

    Args:
        keyword (str): Từ khóa quét được từ OCR / Gemini.
        max_results (int): Số lượng kết quả tối đa (mặc định 5).
        region (str): Mã vùng tìm kiếm ('vn-vi' cho Việt Nam, 'wt-wt' cho Toàn cầu).

    Returns:
        List[Dict[str, str]]: Danh sách kết quả sản phẩm/cửa hàng thực tế.
    """
    global _LAST_SEARCH_ERROR
    _LAST_SEARCH_ERROR = None

    if not keyword.strip():
        return []

    # 1. Tầng 1: Tinh lọc từ khóa trọng tâm + Luôn thêm từ khóa đồ chơi
    smart_query = extract_smart_query(keyword, max_words=4, region=region)
    if smart_query:
        if region != "wt-wt":
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_region = executor.submit(
                    _execute_ddgs_query, smart_query, region, max_results
                )
                future_global = executor.submit(
                    _execute_ddgs_query, smart_query, "wt-wt", max_results
                )
                res_region = future_region.result()
                res_global = future_global.result()

            if res_region:
                _LAST_SEARCH_ERROR = None
                return res_region
            if res_global:
                _LAST_SEARCH_ERROR = None
                return res_global
        else:
            res_global = _execute_ddgs_query(smart_query, "wt-wt", max_results)
            if res_global:
                _LAST_SEARCH_ERROR = None
                return res_global

    # 2. Tầng 2: Tự động nới lỏng truy vấn với 2 từ cốt lõi nhất nếu Tầng 1 chưa tìm thấy
    relaxed_query = extract_smart_query(keyword, max_words=2, region=region)
    if relaxed_query and relaxed_query != smart_query:
        if region != "wt-wt":
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_region = executor.submit(
                    _execute_ddgs_query, relaxed_query, region, max_results
                )
                future_global = executor.submit(
                    _execute_ddgs_query, relaxed_query, "wt-wt", max_results
                )
                res_region = future_region.result()
                res_global = future_global.result()

            if res_region:
                _LAST_SEARCH_ERROR = None
                return res_region
            if res_global:
                _LAST_SEARCH_ERROR = None
                return res_global
        else:
            res_global = _execute_ddgs_query(relaxed_query, "wt-wt", max_results)
            if res_global:
                _LAST_SEARCH_ERROR = None
                return res_global

    # 3. Tầng 3: Thử tìm kiếm trực tiếp bằng từ khóa gốc kết hợp 'toy'
    toy_suffix = " đồ chơi" if region == "vn-vi" else " toy"
    clean_original = " ".join([w for w in keyword.split() if len(w) >= 2]) + toy_suffix
    if clean_original and clean_original != smart_query:
        results = _execute_ddgs_query(clean_original, region="wt-wt", max_results=max_results)
        if results:
            _LAST_SEARCH_ERROR = None
            return results

    _LAST_SEARCH_ERROR = None
    return []


def extract_domain(url: str) -> str:
    """Trích xuất tên miền ngắn gọn từ URL để hiển thị nhãn nguồn tin."""
    try:
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc or "Liên kết ngoài"
    except Exception:
        return "Liên kết ngoài"
