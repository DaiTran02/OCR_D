import sys
sys.path.insert(0, "src")

from pathlib import Path
from utils import load_settings, setup_logging
from main import _process_page

setup_logging()
settings = load_settings()

POPPLER_PATH = settings["POPPLER_PATH"]
PADDLE_API_URL = settings["PADDLE_API_URL"]
MODEL_NAME = settings["MODEL_NAME"]
PDF_PATH = "input/A61.33-020-01-0001-008.pdf"

from pdf2image import convert_from_path
pages = convert_from_path(PDF_PATH, first_page=4, last_page=4, poppler_path=POPPLER_PATH)

page_num, text = _process_page(
    page_number=4,
    total_pages=12,
    image=pages[0],
    api_base_url=PADDLE_API_URL,
    model_name=MODEL_NAME
)

Path("artifacts/debug_page4.txt").write_text(text, encoding="utf-8")
print("Saved to artifacts/debug_page4.txt")
print("First 300 chars:", text[:300].encode("ascii", errors="replace").decode("ascii"))
