from __future__ import annotations

import base64
import logging
import os
import re
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps


LOGGER = logging.getLogger(__name__)
LOCATION_TAG_PATTERN = re.compile(r"<\|LOC_\d+\|>")
MULTISPACE_PATTERN = re.compile(r"\s+")
MOJIBAKE_PATTERN = re.compile(r"(?:Ã.|Ä.|Å.|Æ.|Ç.|Ð.|Ñ.|Õ.|Ø.|á»|áº)")
LIKELY_ABBREVIATIONS = ("V/v", "Số:", "UBND", "BCH", "BTV", "LHPN", "%")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_settings() -> dict[str, str]:
    load_dotenv()

    settings = {
        "POPPLER_PATH": os.getenv("POPPLER_PATH", "").strip(),
        "PADDLE_API_URL": os.getenv("PADDLE_API_URL", "").strip().rstrip("/"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "").strip(),
    }

    missing = [key for key, value in settings.items() if not value]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required environment variables: {missing_text}")

    return settings


def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def image_to_base64(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def upscale_grayscale_image(image, upscale_ratio: float = 1.5):
    processed = image.convert("L")
    new_size = (
        int(processed.width * upscale_ratio),
        int(processed.height * upscale_ratio),
    )
    resampling = getattr(Image, "Resampling", Image)
    return processed.resize(new_size, resample=resampling.LANCZOS)


def preprocess_image_for_ocr(image):
    processed = upscale_grayscale_image(image, upscale_ratio=1.5)

    # Improve local contrast and edge definition without forcing harsh binarization.
    processed = ImageOps.autocontrast(processed, cutoff=1)
    processed = processed.filter(ImageFilter.MedianFilter(size=3))
    processed = processed.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))

    return processed


def preprocess_image_threshold(image):
    processed = upscale_grayscale_image(image, upscale_ratio=1.6)
    processed = ImageOps.autocontrast(processed, cutoff=2)
    processed = processed.filter(ImageFilter.MedianFilter(size=3))
    processed = processed.point(lambda pixel: 255 if pixel > 168 else 0, mode="1").convert("L")
    processed = processed.filter(ImageFilter.UnsharpMask(radius=1.0, percent=140, threshold=1))
    return processed


def preprocess_image_soft_contrast(image):
    processed = upscale_grayscale_image(image, upscale_ratio=1.3)
    processed = ImageOps.autocontrast(processed, cutoff=0)
    processed = processed.filter(ImageFilter.SMOOTH_MORE)
    processed = processed.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=2))
    return processed


def pil_image_to_cv2_gray(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)


def cv2_gray_to_pil(image: np.ndarray) -> Image.Image:
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
    processed = cv2.adaptiveThreshold(
        processed,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return cv2_gray_to_pil(processed)


def split_image_vertically(image, segment_count: int = 4, overlap_ratio: float = 0.18) -> list[object]:
    if segment_count <= 1:
        return [image]

    height = image.height
    segment_height = max(1, int(height / segment_count))
    overlap = max(1, int(segment_height * overlap_ratio))
    segments: list[object] = []

    for index in range(segment_count):
        start_y = max(0, index * segment_height - overlap)
        end_y = min(height, (index + 1) * segment_height + overlap)
        if index == 0:
            start_y = 0
        if index == segment_count - 1:
            end_y = height
        segments.append(image.crop((0, start_y, image.width, end_y)))

    return segments


def build_ocr_image_variants(image) -> list[tuple[str, object]]:
    return [
        ("full_page", image.convert("RGB")),
        ("opencv_document", preprocess_image_opencv_document(image)),
        ("opencv_adaptive", preprocess_image_opencv_adaptive(image)),
        ("sharpened", preprocess_image_for_ocr(image)),
        ("threshold", preprocess_image_threshold(image)),
        ("soft_contrast", preprocess_image_soft_contrast(image)),
    ]


def clean_ocr_text(text: str) -> str:
    cleaned = LOCATION_TAG_PATTERN.sub("", text)
    cleaned = remove_repeated_lines(cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_line_for_dedup(line: str) -> str:
    normalized = MULTISPACE_PATTERN.sub(" ", line.strip())
    return normalized.casefold()


def remove_repeated_lines(text: str, max_consecutive_duplicates: int = 0) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    cleaned_lines: list[str] = []
    previous_normalized = None
    duplicate_count = 0

    for line in lines:
        normalized = normalize_line_for_dedup(line)

        if normalized and normalized == previous_normalized:
            duplicate_count += 1
            if duplicate_count > max_consecutive_duplicates:
                continue
        else:
            previous_normalized = normalized
            duplicate_count = 0

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _fuzzy_line_similarity(line_a: str, line_b: str) -> float:
    """Return similarity ratio [0.0, 1.0] between two normalized lines."""
    return SequenceMatcher(None, line_a, line_b).ratio()


def _is_duplicate_line(normalized_line: str, recent_normalized: list[str], threshold: float = 0.92) -> bool:
    """Return True if normalized_line is highly similar to any recent line."""
    if not normalized_line:
        return False
    for recent in recent_normalized:
        if _fuzzy_line_similarity(normalized_line, recent) >= threshold:
            return True
    return False


def _find_best_overlap(merged_lines: list[str], chunk_lines: list[str], fuzzy_threshold: float = 0.88) -> int:
    """Find the largest overlap between the tail of merged_lines and the head of chunk_lines.

    1. Try exact multi-line overlap (sliding window from largest to smallest).
    2. Fall back to fuzzy multi-line overlap using the same sliding window.
    3. Finally try single-line fuzzy match scanning the last few merged lines.

    Returns the number of lines at the start of chunk_lines that should be skipped.
    """
    # ── Exact multi-line overlap ──────────────────────────────────────────────
    max_check = min(len(merged_lines), len(chunk_lines))
    for size in range(max_check, 0, -1):
        merged_tail = [normalize_line_for_dedup(ln) for ln in merged_lines[-size:]]
        chunk_head = [normalize_line_for_dedup(ln) for ln in chunk_lines[:size]]
        if merged_tail == chunk_head:
            return size

    # ── Fuzzy multi-line overlap ─────────────────────────────────────────────
    # For each candidate window size (from large to small) check if every
    # paired line exceeds the fuzzy threshold.
    for size in range(min(max_check, 8), 1, -1):
        merged_tail = [normalize_line_for_dedup(ln) for ln in merged_lines[-size:]]
        chunk_head = [normalize_line_for_dedup(ln) for ln in chunk_lines[:size]]
        if all(
            _fuzzy_line_similarity(a, b) >= fuzzy_threshold
            for a, b in zip(merged_tail, chunk_head)
        ):
            return size

    # ── Single-line fuzzy scan ────────────────────────────────────────────────
    # Check if the first line of the new chunk matches any of the recent lines
    # in the merged result.  If it does, skip exactly one line.
    if chunk_lines:
        first_norm = normalize_line_for_dedup(chunk_lines[0])
        look_back_limit = min(8, len(merged_lines))
        for look_back in range(1, look_back_limit + 1):
            tail_norm = normalize_line_for_dedup(merged_lines[-look_back])
            if _fuzzy_line_similarity(first_norm, tail_norm) >= 0.92:
                return 1

    return 0


def merge_ocr_text_chunks(text_chunks: list[str]) -> str:
    """Merge OCR text chunks produced from vertical page slices.

    Uses a sliding-window overlap algorithm to remove duplicated content
    introduced by the slice overlap, then applies a final fuzzy-dedup pass
    to suppress any remaining near-duplicate lines.
    """
    merged_lines: list[str] = []

    for chunk in text_chunks:
        chunk_lines = [line for line in chunk.splitlines() if line.strip()]
        if not chunk_lines:
            continue

        if not merged_lines:
            merged_lines.extend(chunk_lines)
            continue

        overlap_size = _find_best_overlap(merged_lines, chunk_lines)

        # Append non-overlapping lines with duplicate guard.
        recent_normalized = [normalize_line_for_dedup(item) for item in merged_lines[-20:]]
        for line in chunk_lines[overlap_size:]:
            normalized_line = normalize_line_for_dedup(line)
            if normalized_line and _is_duplicate_line(normalized_line, recent_normalized):
                continue
            merged_lines.append(line)
            recent_normalized.append(normalized_line)
            if len(recent_normalized) > 20:
                recent_normalized.pop(0)

    return "\n".join(merged_lines)


def score_ocr_text(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return -10_000.0

    score = 0.0
    lines = [line for line in stripped.splitlines() if line.strip()]
    words = stripped.split()

    score += len(words) * 0.8
    score += len(lines) * 0.5

    # Density bonus: reward results with more content (encourages complete extraction).
    char_count = len(stripped)
    if char_count >= 2000:
        score += 60.0
    elif char_count >= 1000:
        score += 30.0
    elif char_count >= 500:
        score += 10.0

    score -= len(LOCATION_TAG_PATTERN.findall(text)) * 10
    score -= len(MOJIBAKE_PATTERN.findall(text)) * 2.5

    previous_line = None
    repeated_line_count = 0
    for line in lines:
        normalized_line = normalize_line_for_dedup(line)
        if normalized_line and normalized_line == previous_line:
            repeated_line_count += 1
        previous_line = normalized_line
    score -= repeated_line_count * 10

    previous_word = None
    repeated_word_count = 0
    for word in words:
        normalized_word = normalize_line_for_dedup(word)
        if normalized_word and normalized_word == previous_word:
            repeated_word_count += 1
        previous_word = normalized_word
    score -= repeated_word_count * 2

    for abbreviation in LIKELY_ABBREVIATIONS:
        if abbreviation in text:
            score += 4

    preferred_patterns = (
        "V/v",
        "được mời",
        "Số:",
    )
    penalized_patterns = (
        "Vị",
        "được mỗi",
    )

    for pattern in preferred_patterns:
        if pattern in text:
            score += 8

    for pattern in penalized_patterns:
        if pattern in text:
            score -= 8

    return score


def build_chat_payload(model_name: str, image_b64: str) -> dict:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe ALL text from this image exactly as it appears. "
                            "Include every word, number, and line in reading order from top to bottom. "
                            "Keep original abbreviations (e.g. V/v, UBND, BCH). "
                            "Do not skip, summarize, or rewrite any text. "
                            "Do not repeat characters or words endlessly."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        "temperature": 0,
    }


def extract_text_from_response(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected API response format: {payload}") from exc

    if isinstance(content, str):
        return clean_ocr_text(content)

    if isinstance(content, list):
        text_chunks: list[str] = []
        for item in content:
            if item.get("type") == "text" and item.get("text"):
                text_chunks.append(item["text"])
        return clean_ocr_text("\n".join(text_chunks))

    raise ValueError(f"Unsupported response content type: {type(content)!r}")


def call_paddle_ocr(api_base_url: str, model_name: str, image_b64: str, timeout: int = 180) -> str:
    endpoint = f"{api_base_url}/chat/completions"
    payload = build_chat_payload(model_name=model_name, image_b64=image_b64)

    LOGGER.info("Calling OCR API: %s", endpoint)
    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()

    return extract_text_from_response(response.json())
