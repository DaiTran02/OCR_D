from pdf2image import convert_from_path
from pathlib import Path

POPPLER_PATH = r"D:\NGNs\ai_ngn\oursource\poppler\Library\bin"
PDF_PATH = "input/A61.33-020-01-0001-008.pdf"
OUTPUT_DIR = Path("artifacts")

images = convert_from_path(PDF_PATH, first_page=4, last_page=4, poppler_path=POPPLER_PATH)
img = images[0]
height = img.height
slice1 = img.crop((0, 0, img.width, int(height/7 + height/35)))
slice1.save(OUTPUT_DIR / "debug_slice1.png")
print("Saved debug_slice1.png")
