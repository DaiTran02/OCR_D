from __future__ import annotations

import time
from pathlib import Path

from ollama import Client, ResponseError


CORRECTION_INSTRUCTIONS = (
    "You correct Vietnamese text that was read from a scanned document by OCR. "
    "Fix wrong words and spelling. Remove duplicate words, duplicate lines, and "
    "repeated paragraphs, keeping a single copy in the original order. "
    "Keep numbers, dates, names, and abbreviations. Do not add facts. "
    "Do not translate. Return only the corrected text, with the same line breaks."
)


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def split_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split Markdown into (heading, body) pairs.

    A heading is a line that starts with '#'. Text before the first heading
    is returned with an empty heading.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    body_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(body_lines).strip()
        if heading or body:
            sections.append((heading, body))

    for line in markdown_text.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.strip()
            body_lines = []
            continue
        body_lines.append(line)
    flush()
    return sections


def correct_text(client: Client, model: str, text: str) -> str:
    messages = [
        {"role": "system", "content": CORRECTION_INSTRUCTIONS},
        {"role": "user", "content": text},
    ]
    try:
        response = client.chat(
            model=model,
            messages=messages,
            think=True,
            options={"temperature": 0},
        )
    except ResponseError as exc:
        if "think" not in str(exc).lower():
            raise
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": 0},
        )

    content = response.message.content or ""
    corrected = _strip_code_fence(content)
    if not corrected:
        raise ValueError("Ollama returned an empty correction")
    return corrected


def correct_markdown_file(markdown_path: Path, output_path: Path, host: str, model: str) -> None:
    print(f"[ollama] Correcting {markdown_path.name} with {model} at {host}")
    corrected = correct_markdown_text(markdown_path.read_text(encoding="utf-8"), host=host, model=model)
    output_path.write_text(corrected, encoding="utf-8")


def correct_markdown_text(markdown_text: str, host: str, model: str) -> str:
    started = time.time()
    client = Client(host=host)
    pieces: list[str] = []

    for heading, body in split_markdown_sections(markdown_text):
        if heading:
            pieces.extend([heading, ""])
        if not body:
            continue
        label = heading or "preamble"
        print(f"[ollama] Fixing wrong words and duplicates: {label}")
        corrected = correct_text(client, model, body)
        pieces.append(corrected)
        pieces.append("")

    elapsed = int(time.time() - started)
    print(f"[ollama] Correction completed in {elapsed // 60:02d}:{elapsed % 60:02d}")
    return "\n".join(pieces).strip() + "\n"
