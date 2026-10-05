from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from pathlib import Path

import torch
from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt
from ftfy import fix_text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from utils import is_runaway_ocr_text


CORRECTION_MODEL_NAME = "bmd1905/vietnamese-correction-v2"
MAX_OUTPUT_LENGTH = 512
MAX_CHUNK_CHARS = 400
MARKDOWN_HEADER_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s")
LIST_PREFIX_PATTERN = re.compile(r"^(\s*(?:[-+*]|\d+\.)\s+)(.*)$")

# Batch inference configuration.
# Larger batches = better GPU utilisation but more VRAM usage.
# 16 is a good default for a 6 GB GPU with this ~600 M-param MBart model.
CORRECTION_BATCH_SIZE = 16


def get_inference_device() -> int:
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[postprocess] Using GPU: {gpu_name}")
        return 0
    print("[postprocess] CUDA not available - using quantized CPU inference")
    return -1


def _quantize_cpu_model(model):
    """Int8 dynamic quantization. About 6x faster than fp32 on this CPU."""
    try:
        return torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
    except Exception as exc:
        print(f"[postprocess] CPU quantization skipped: {exc}")
        return model


@lru_cache(maxsize=1)
def load_corrector():
    device = get_inference_device()
    if device < 0:
        torch.set_num_threads(os.cpu_count() or 4)

    dtype = torch.float16 if device >= 0 else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(CORRECTION_MODEL_NAME, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            CORRECTION_MODEL_NAME,
            local_files_only=True,
            torch_dtype=dtype,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(CORRECTION_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(CORRECTION_MODEL_NAME, torch_dtype=dtype)

    corrector = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    if device < 0:
        corrector.model = _quantize_cpu_model(corrector.model)
        print("[postprocess] Using int8 CPU weights")
    return corrector


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


def correct_markdown_text(markdown_text: str) -> str:
    """Correct OCR text using batched model inference for speed.

    Instead of calling the model per-line, this collects all correctable lines
    into a single batch (split into sub-batches of CORRECTION_BATCH_SIZE) and
    runs inference in bulk.  On GPU this is 5-10x faster than line-by-line.
    """
    corrector = load_corrector()
    lines = markdown_text.splitlines()

    # ---- Phase 1: classify lines and collect correctable content ----------
    # Each entry: (line_index, prefix, [text_chunks])
    correctable_entries: list[tuple[int, str, list[str]]] = []
    # All chunks across all lines, flattened for batch inference.
    all_chunks: list[str] = []
    # Mapping: (entry_index_in_correctable_entries, chunk_position_in_entry)
    chunk_to_entry: list[tuple[int, int]] = []

    output_lines: list[str | None] = [None] * len(lines)

    for idx, raw_line in enumerate(lines):
        line = raw_line.rstrip()

        if not line.strip():
            output_lines[idx] = ""
            continue

        if is_runaway_ocr_text(line):
            continue

        if MARKDOWN_HEADER_PATTERN.match(line):
            output_lines[idx] = line.strip()
            continue

        prefix, content = split_line_prefix(line)
        if should_skip_line_correction(content):
            output_lines[idx] = line
            continue

        repaired = repair_text_encoding(content)
        chunks = split_long_text(repaired)
        if not chunks:
            output_lines[idx] = line
            continue

        entry_index = len(correctable_entries)
        correctable_entries.append((idx, prefix, chunks))
        for chunk_pos, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_to_entry.append((entry_index, chunk_pos))

    # ---- Phase 2: batched model inference ---------------------------------
    if all_chunks:
        # Identical lines are common after slice merging. Correct each once.
        unique_chunks: list[str] = []
        remap: list[int] = []
        seen: dict[str, int] = {}
        for chunk in all_chunks:
            index = seen.get(chunk)
            if index is None:
                index = len(unique_chunks)
                seen[chunk] = index
                unique_chunks.append(chunk)
            remap.append(index)

        unique_corrected: list[str] = [""] * len(unique_chunks)
        for batch_start in range(0, len(unique_chunks), CORRECTION_BATCH_SIZE):
            batch = unique_chunks[batch_start : batch_start + CORRECTION_BATCH_SIZE]
            predictions = corrector(batch, max_length=MAX_OUTPUT_LENGTH, num_beams=1)
            for i, pred in enumerate(predictions):
                unique_corrected[batch_start + i] = pred["generated_text"].strip()

        all_corrected = [unique_corrected[index] for index in remap]

        # ---- Phase 3: reassemble lines -----------------------------------
        # Group corrected chunks back per entry.
        entry_chunks: dict[int, list[str]] = {}
        for global_idx, (entry_idx, _chunk_pos) in enumerate(chunk_to_entry):
            entry_chunks.setdefault(entry_idx, []).append(all_corrected[global_idx])

        for entry_idx, (line_idx, prefix, _original_chunks) in enumerate(correctable_entries):
            corrected_parts = entry_chunks.get(entry_idx, [])
            corrected_text = normalize_corrected_text(
                "\n".join(c for c in corrected_parts if c).strip()
            )
            if corrected_text:
                output_lines[line_idx] = f"{prefix}{corrected_text}"
            else:
                # Fall back to the original line if correction produced nothing.
                output_lines[line_idx] = lines[line_idx].rstrip()

    # Any remaining None entries (shouldn't happen, but safety net).
    # Runaway lines stay dropped so the correction model cannot expand them.
    kept_lines: list[str] = []
    for idx, line in enumerate(output_lines):
        if line is None:
            original = lines[idx].rstrip()
            if is_runaway_ocr_text(original):
                continue
            kept_lines.append(original)
        else:
            kept_lines.append(line)

    return "\n".join(kept_lines).strip() + "\n"


def _apply_run_font(run, size_pt: int, bold: bool = False) -> None:
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(size_pt)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def write_docx(markdown_path: Path, docx_path: Path) -> None:
    """Write the cleaned Markdown as a Word document (.docx)."""
    document = Document()
    section = document.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.left_margin = Cm(2)
    section.right_margin = Cm(2)
    section.top_margin = Cm(2)
    section.bottom_margin = Cm(2)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(13)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.startswith("# "):
            paragraph = document.add_heading(level=1)
            _apply_run_font(paragraph.add_run(line[2:].strip()), 16, bold=True)
            continue

        if line.startswith("## "):
            paragraph = document.add_heading(level=2)
            _apply_run_font(paragraph.add_run(line[3:].strip()), 14, bold=True)
            continue

        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        _apply_run_font(paragraph.add_run(line), 13)

    document.save(docx_path)
    print(f"[postprocess] Word document saved to: {docx_path}")


def generate_clean_output(input_path: Path, output_path: Path) -> None:
    print(f"[postprocess] Starting text correction for: {input_path.name}")
    start = time.time()

    markdown_text = input_path.read_text(encoding="utf-8")
    corrected_markdown = correct_markdown_text(markdown_text)
    output_path.write_text(corrected_markdown, encoding="utf-8")

    duration = time.time() - start
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"[postprocess] Correction completed in {minutes:02d}:{seconds:02d}")
