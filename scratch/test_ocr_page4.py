import os
from pathlib import Path
from pdf2image import convert_from_path
import requests
import base64
from io import BytesIO
from PIL import Image

# Setup paths
POPPLER_PATH = r"D:\NGNs\ai_ngn\oursource\poppler\Library\bin"
PADDLE_API_URL = "http://192.168.10.247:8089/v1"
MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"
PDF_PATH = "input/A61.33-020-01-0001-008.pdf"

def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def call_ocr(image):
    b64 = image_to_base64(image)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the entire text from this image VERBATIM. "
                            "Do not omit any visible text, including page numbers, stamps, or signatures. "
                            "Retain original formatting and abbreviations."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]
            }
        ],
        "temperature": 0
    }
    response = requests.post(f"{PADDLE_API_URL}/chat/completions", json=payload, timeout=180)
    response.raise_for_status()
    raw = response.json()
    import json
    Path("artifacts/raw_response.json").write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    return raw["choices"][0]["message"]["content"]

print(f"Processing page 4...")
images = convert_from_path(PDF_PATH, first_page=4, last_page=4, poppler_path=POPPLER_PATH)
img = images[0]

# Full page OCR
print("\n--- Full Page OCR ---")
full_text = call_ocr(img)
Path("artifacts/test_full_text.txt").write_text(full_text, encoding="utf-8")
print("Saved to artifacts/test_full_text.txt")

# Sliced OCR (top slice)
print("\n--- Slice 1 (Top 1/7) OCR ---")
height = img.height
slice1 = img.crop((0, 0, img.width, int(height/7 + height/35)))
slice1_text = call_ocr(slice1)
Path("artifacts/test_slice1_text.txt").write_text(slice1_text, encoding="utf-8")
print("Saved to artifacts/test_slice1_text.txt")
