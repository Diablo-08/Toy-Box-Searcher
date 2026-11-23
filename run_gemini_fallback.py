import sys
from pathlib import Path
from PIL import Image
from gemini_fallback import extract_keywords_gemini

# Đường dẫn tới ảnh c.jpg
image_path = Path(r"c:\Users\USER\Downloads\toy_box_searcher\c.jpg")
if not image_path.is_file():
    print('Không tìm thấy ảnh:', image_path)
    sys.exit(1)

# Mở ảnh và gọi Gemini để trích xuất từ khóa
image = Image.open(image_path)
keywords = extract_keywords_gemini(image)
print('Gemini trả về các từ khóa:')
print(', '.join(keywords) if keywords else '(không có từ khóa)')
