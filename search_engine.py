"""
Module: search_engine.py
Mục đích: Tìm kiếm thông tin sản phẩm đồ chơi trên Internet bằng thư viện ddgs (DuckDuckGo Search)
          tối ưu hóa theo khu vực (Việt Nam vn-vi / Toàn cầu wt-wt), tự động lọc từ rác OCR,
          LOẠI BỎ 100% CÁC TRANG WIKIPEDIA/WIKIMEDIA để tập trung hoàn toàn vào các website sản phẩm,
          sàn thương mại điện tử, bài đánh giá và cửa hàng đồ chơi thực tế.
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
TOY_VOCABULARY = {\n    # Additional generic toy terms\n    "toy", "đồ", "chơi", "đồ chơi", "đồchơi",\n    # Existing terms ...
    # Thương hiệu
    "lego", "hot", "wheels", "barbie", "bandai", "gundam", "transformers",
    "hasbro", "mattel", "fisher", "price", "nerf", "playmobil", "funko",
    "tomica", "disney", "marvel", "pokemon", "tamiya", "kyosho",
    # Loại phương tiện & đồ chơi
    "car", "cars", "truck", "trucks", "train", "trains", "plane", "planes",
    "robot", "robots", "vehicle", "vehicles", "series", "diy", "construction",
    "builder", "builders", "building", "racing", "speed", "city", "police",
    "fire", "action", "figure", "figures", "blocks", "block", "model", "kit",
    "set", "diecast", "scale", "express", "passenger", "gunpla", "track",
    "rc", "remote", "control", "drone", "puzzle", "game",
    # Tiếng Việt
    "xe", "do", "choi", "lap", "rap", "mo", "hinh", "dieu", "khien"
}

_LAST_SEARCH_ERROR: Optional[str] = None


def get_last_search_error() -> Optional[str]:
    """Lấy thông điệp lỗi của lần tìm kiếm gần nhất (nếu có)."""
    global _LAST_SEARCH_ERROR
    return _LAST_SEARCH_ERROR


def is_wikipedia_entry(url: str, title: str) -> bool:
    """
    Kiểm tra xem liên kết hoặc tiêu đề có thuộc Wikipedia / Wikimedia hay không
    để LOẠI BỎ HOÀN TOÀN (theo yêu cầu người dùng không lấy trang bách khoa).
    """
    u = (url or "").lower()
    t = (title or "").lower()
    return (
        "wikipedia.org" in u
        or "wikimedia.org" in u
        or "wiktionary.org" in u
        or "wikipedia" in t
        or "bách khoa toàn thư" in t
    )


def extract_smart_query(raw_keyword: str, max_words: int = 4) -> str:
    """
    Trích xuất từ khóa thông minh có chọn lọc:
    - Chấm điểm từng từ: Mã số/SKU (25đ), Từ vựng đồ chơi (15đ), Từ tiếng Anh có nghĩa (5đ), Từ rác/lỗi OCR (1đ).
    - Loại bỏ hoàn toàn các từ rác do OCR sinh ra (như 'JOKNGDI', 'SCREWEIT').
    - Lấy top N từ có điểm số cao nhất để tạo truy vấn có xác suất tìm thấy cao nhất.

    Args:
        raw_keyword (str): Chuỗi từ khóa quét được từ OCR.
        max_words (int): Số lượng từ tối đa cần giữ lại.

    Returns:
        str: Chuỗi truy vấn đã được tinh lọc.
    """
    tokens = re.findall(r"[A-Za-z0-9_-]+", raw_keyword)
    if not tokens:
        return ""

    scored_tokens = []
    has_brand_or_sku = False
    try:
        from ocr_extractor import KNOWN_BRANDS
        known_brands_set = {b.lower() for b in KNOWN_BRANDS}
    except Exception:
        known_brands_set = set()

    for t in tokens:
        tl = t.lower()

        # 1. Mã sản phẩm / Số hiệu hộp đồ chơi (ví dụ: 60337, HW123, RX-78)
        if re.match(r"^\d{3,6}$", t) or re.match(r"^[A-Za-z]{1,4}\d{2,6}$", t):
            scored_tokens.append((t, 25))
            has_brand_or_sku = True
        # 2. Từ khóa nằm trong từ điển đồ chơi chính xác
        # 2. Direct brand match (higher priority)
        elif tl in known_brands_set:
            scored_tokens.append((t, 30))
            has_brand_or_sku = True
        # 3. Token in toy vocabulary
        elif tl in TOY_VOCABULARY:
            scored_tokens.append((t, 15))
            if tl in {"lego", "hot", "wheels", "barbie", "bandai", "gundam", "transformers"}:
                has_brand_or_sku = True
        # 3. Từ tiếng Anh thông thường có nguyên âm hợp lệ
        elif len(t) >= 3 and re.search(r"[aeiouyAEIOUY]", t):
            scored_tokens.append((t, 5))
        else:
            # Từ rác hoặc lỗi chính tả OCR
            scored_tokens.append((t, 1))

    # Sắp xếp từ theo điểm số giảm dần
    scored_tokens.sort(key=lambda x: x[1], reverse=True)

    # Chọn top từ có điểm cao nhất
    selected_words = [item[0] for item in scored_tokens[:max_words]]
    result_query = " ".join(selected_words)

    # Chỉ bổ sung 'toy' nếu chưa có thương hiệu/mã SKU và chưa có từ chỉ đồ chơi
    lower_res = result_query.lower()
    if (
        not has_brand_or_sku
        and "toy" not in lower_res
        and "xe" not in lower_res
        and "choi" not in lower_res
    ):
        result_query = f"{result_query} toy".strip()

    return result_query


def _execute_ddgs_query(query: str, region: str, max_results: int) -> List[Dict[str, str]]:
    """
    Thực thi truy vấn trên DDGS và trả về danh sách kết quả sản phẩm thương mại.
    BẮT BUỘC: LỌC BỎ 100% CÁC TRANG WIKIPEDIA / WIKIMEDIA.
    """
    results: List[Dict[str, str]] = []
    # Lấy dư thêm kết quả để sau khi lọc Wikipedia vẫn đủ số lượng cần thiết
    fetch_limit = max_results + 5

    try:
        with DDGS(timeout=5) as ddgs:
            raw_response = ddgs.text(query, region=region, max_results=fetch_limit)
            if raw_response:
                for item in raw_response:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title") or item.get("name") or ""
                    href = item.get("href") or item.get("url") or item.get("link") or ""
                    body = item.get("body") or item.get("snippet") or item.get("description") or ""

                    # BỎ QUA HOÀN TOÀN NẾU LÀ TRANG WIKIPEDIA
                    if is_wikipedia_entry(href, title):
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
    Tra cứu thông tin sản phẩm đồ chơi thương mại theo khu vực (KHÔNG LẤY WIKIPEDIA),
    kết hợp đa luồng (ThreadPoolExecutor) để truy vấn đồng thời khu vực chỉ định và toàn cầu,
    giúp giảm độ trễ phản hồi tối đa:
    1. Tầng 1: Tinh lọc từ khóa thông minh (Smart Query), truy vấn song song (Region + Global).
    2. Tầng 2: Nới lỏng từ khóa (Relaxed Query) nếu Tầng 1 chưa có kết quả.
    3. Tầng 3: Truy vấn với từ khóa gốc đã làm sạch.
    4. BẢO ĐẢM: Không bao giờ trả về bất kỳ liên kết Wikipedia hay bách khoa toàn thư nào.

    Args:
        keyword (str): Từ khóa thô từ OCR.
        max_results (int): Số lượng kết quả tối đa (mặc định 5).
        region (str): Mã vùng tìm kiếm ('vn-vi' cho Việt Nam, 'wt-wt' cho Toàn cầu).

    Returns:
        List[Dict[str, str]]: Danh sách kết quả sản phẩm/cửa hàng thực tế.
    """
    global _LAST_SEARCH_ERROR
    _LAST_SEARCH_ERROR = None

    if not keyword.strip():
        return []

    # 1. Trích xuất từ khóa trọng tâm (ưu tiên thương hiệu, mã SKU và từ vựng đồ chơi)
    smart_query = extract_smart_query(keyword, max_words=4)
    if smart_query:
        if region != "wt-wt":
            # ĐA LUỒNG: Chạy song song truy vấn khu vực (vn-vi) và toàn cầu (wt-wt)
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

    # 2. Tự động nới lỏng truy vấn với 2 từ cốt lõi nhất nếu Tầng 1 chưa tìm thấy
    relaxed_query = extract_smart_query(keyword, max_words=2)
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

    # 3. Thử tìm kiếm trực tiếp bằng chính từ khóa gốc (loại trừ từ rác đơn giản)
    clean_original = " ".join([w for w in keyword.split() if len(w) >= 2])
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
