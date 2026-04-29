from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch
from ftfy import fix_text
from transformers import pipeline


CORRECTION_MODEL_NAME = "bmd1905/vietnamese-correction-v2"
MAX_OUTPUT_LENGTH = 512
MAX_CHUNK_CHARS = 400
MARKDOWN_HEADER_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s")
LIST_PREFIX_PATTERN = re.compile(r"^(\s*(?:[-+*]|\d+\.)\s+)(.*)$")


def get_inference_device() -> int:
    return 0 if torch.cuda.is_available() else -1


@lru_cache(maxsize=1)
def load_corrector():
    return pipeline(
        "text2text-generation",
        model=CORRECTION_MODEL_NAME,
        device=get_inference_device(),
    )


def split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    if len(stripped) <= max_chars:
        return [stripped]

    words = stripped.split()
    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate = " ".join(current_words + [word]).strip()
        if current_words and len(candidate) > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def normalize_corrected_text(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def repair_text_encoding(text: str) -> str:
    return fix_text(text, normalization="NFC").strip()


def correct_text_chunks(chunks: list[str]) -> str:
    if not chunks:
        return ""

    corrector = load_corrector()
    predictions = corrector(chunks, max_length=MAX_OUTPUT_LENGTH)
    corrected_chunks = [prediction["generated_text"].strip() for prediction in predictions]
    return "\n".join(chunk for chunk in corrected_chunks if chunk).strip()


def should_skip_line_correction(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True

    # Preserve short labels, page markers, and mostly-uppercase headings as-is.
    if len(stripped) <= 3:
        return True
    if stripped.isupper():
        return True

    return False


def split_line_prefix(line: str) -> tuple[str, str]:
    match = LIST_PREFIX_PATTERN.match(line)
    if match:
        return match.group(1), match.group(2)

    leading_spaces = line[: len(line) - len(line.lstrip())]
    return leading_spaces, line.lstrip()


def correct_line_text(line: str) -> str:
    prefix, content = split_line_prefix(line.rstrip())
    if should_skip_line_correction(content):
        return line.rstrip()

    repaired = repair_text_encoding(content)
    corrected = correct_text_chunks(split_long_text(repaired))
    corrected = normalize_corrected_text(corrected)
    if not corrected:
        return line.rstrip()

    return f"{prefix}{corrected}"


def correct_markdown_text(markdown_text: str) -> str:
    output_lines: list[str] = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            output_lines.append("")
            continue

        if MARKDOWN_HEADER_PATTERN.match(line):
            output_lines.append(line.strip())
            continue

        output_lines.append(correct_line_text(line))

    return "\n".join(output_lines).strip() + "\n"


def generate_clean_output(input_path: Path, output_path: Path) -> None:
    markdown_text = input_path.read_text(encoding="utf-8")
    corrected_markdown = correct_markdown_text(markdown_text)
    output_path.write_text(corrected_markdown, encoding="utf-8")
