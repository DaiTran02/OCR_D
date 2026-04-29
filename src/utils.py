from __future__ import annotations

import base64
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)
LOCATION_TAG_PATTERN = re.compile(r"<\|LOC_\d+\|>")
MULTISPACE_PATTERN = re.compile(r"\s+")


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
                            "Extract all text from this image precisely. "
                            "If you encounter stamps, seals, or signatures at the end of the document, "
                            "identify them as [Stamp] or [Signature] and STOP generating immediately. "
                            "Do not repeat words or characters endlessly."
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
