import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import cv2
import numpy as np

# Copy utilities from utils.py
def pil_image_to_cv2_gray(image):
    rgb_image = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)

def cv2_gray_to_pil(image):
    return Image.fromarray(image)

def preprocess_image_opencv_document(image):
    processed = pil_image_to_cv2_gray(image)
    processed = cv2.resize(processed, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    processed = cv2.fastNlMeansDenoising(processed, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    processed = cv2.filter2D(processed, -1, sharpen_kernel)
    processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
    return cv2_gray_to_pil(processed)

def preprocess_image_opencv_adaptive(image):
    processed = pil_image_to_cv2_gray(image)
    processed = cv2.resize(processed, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
    processed = cv2.GaussianBlur(processed, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    processed = clahe.apply(processed)
    processed = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return cv2_gray_to_pil(processed)

POPPLER_PATH = r"D:\NGNs\ai_ngn\oursource\poppler\Library\bin"
PDF_PATH = "input/A61.33-020-01-0001-008.pdf"
OUTPUT_DIR = Path("artifacts/variants")
OUTPUT_DIR.mkdir(exist_ok=True)

images = convert_from_path(PDF_PATH, first_page=4, last_page=4, poppler_path=POPPLER_PATH)
img = images[0]

preprocess_image_opencv_document(img).save(OUTPUT_DIR / "page_4_opencv_document.png")
preprocess_image_opencv_adaptive(img).save(OUTPUT_DIR / "page_4_opencv_adaptive.png")
print("Saved variants to artifacts/variants/")
