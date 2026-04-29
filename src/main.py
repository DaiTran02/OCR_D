from __future__ import annotations

import sys
from pathlib import Path

from pdf2image import convert_from_path

from postprocess import generate_clean_output
from utils import (
    call_paddle_ocr,
    clean_ocr_text,
    ensure_directories,
    image_to_base64,
    load_settings,
    setup_logging,
)


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "output.md"
OUTPUT_CLEAR_FILE = OUTPUT_DIR / "output_clear.md"


def find_pdf_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.pdf"))


def convert_pdf_to_images(pdf_path: Path, poppler_path: str):
    return convert_from_path(
        pdf_path=str(pdf_path),
        poppler_path=poppler_path,
        fmt="png",
    )


def process_pdf(pdf_path: Path, poppler_path: str, api_base_url: str, model_name: str) -> str:
    print(f"Processing PDF: {pdf_path.name}")
    pages = convert_pdf_to_images(pdf_path=pdf_path, poppler_path=poppler_path)

    markdown_sections: list[str] = [f"# OCR Result: {pdf_path.name}", ""]
    for page_number, image in enumerate(pages, start=1):
        print(f"  - Page {page_number}/{len(pages)}")
        image_b64 = image_to_base64(image)
        extracted_text = call_paddle_ocr(
            api_base_url=api_base_url,
            model_name=model_name,
            image_b64=image_b64,
        )

        page_text = clean_ocr_text(extracted_text) or "_No text extracted._"
        markdown_sections.extend(
            [
                f"## Page {page_number}",
                "",
                page_text,
                "",
            ]
        )

    return "\n".join(markdown_sections).strip() + "\n"


def main() -> int:
    setup_logging()
    ensure_directories([INPUT_DIR, OUTPUT_DIR])

    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    pdf_files = find_pdf_files(INPUT_DIR)
    if not pdf_files:
        print(f"No PDF files found in: {INPUT_DIR}", file=sys.stderr)
        return 1

    all_results: list[str] = []
    for pdf_file in pdf_files:
        try:
            result = process_pdf(
                pdf_path=pdf_file,
                poppler_path=settings["POPPLER_PATH"],
                api_base_url=settings["PADDLE_API_URL"],
                model_name=settings["MODEL_NAME"],
            )
            all_results.append(result)
        except Exception as exc:
            print(f"Failed to process {pdf_file.name}: {exc}", file=sys.stderr)
            return 1

    OUTPUT_FILE.write_text("\n\n".join(all_results), encoding="utf-8")
    generate_clean_output(OUTPUT_FILE, OUTPUT_CLEAR_FILE)
    print(f"OCR completed. Output saved to: {OUTPUT_FILE}")
    print(f"Cleaned output saved to: {OUTPUT_CLEAR_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
