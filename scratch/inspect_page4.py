import os
from pdf2image import convert_from_path
from pathlib import Path

# Adjust paths as needed
POPPLER_PATH = r"D:\NGNs\ai_ngn\oursource\poppler\Library\bin"
PDF_PATH = r"input/A61.33-020-01-0001-008.pdf"
OUTPUT_DIR = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Converting page 4 of {PDF_PATH}...")
images = convert_from_path(PDF_PATH, first_page=4, last_page=4, poppler_path=POPPLER_PATH)
if images:
    images[0].save(OUTPUT_DIR / "page_4_original.png")
    print("Saved page_4_original.png")
else:
    print("Failed to convert page 4")
