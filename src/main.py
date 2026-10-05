from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pdf2image import convert_from_path

from ollama_correct import correct_markdown_file
from postprocess import write_docx
from utils import (
    OCR_VARIANT_ORDER,
    build_ocr_image_variants,
    call_paddle_ocr,
    clean_ocr_text,
    ensure_directories,
    image_to_base64,
    is_runaway_ocr_text,
    load_settings,
    merge_ocr_text_chunks,
    score_ocr_text,
    split_image_vertically,
    setup_logging,
)


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "output.md"
OUTPUT_CLEAR_FILE = OUTPUT_DIR / "output_clear.md"
OUTPUT_DOCX_FILE = OUTPUT_DIR / "output_clear.docx"

# Always-slice configuration.
SEGMENT_COUNT = 5       # Number of vertical slices per page.
SLICE_OVERLAP = 0.30   # Overlap ratio between adjacent slices.

# Concurrency configuration.
# MAX_VARIANT_WORKERS: all 6 preprocessing variants run concurrently in Step 1.
# MAX_SLICE_WORKERS:   all SEGMENT_COUNT slices run concurrently in Step 2.
# MAX_PAGE_WORKERS:    number of pages processed in parallel (increase carefully
#                      based on how many concurrent requests the OCR server can handle).
MAX_VARIANT_WORKERS = 6
MAX_SLICE_WORKERS = SEGMENT_COUNT
MAX_PAGE_WORKERS = 2

# Variants that won on real pages. The other two are only OCR'd when this
# probe comes back weak (blank page, failed read, very little text).
PRIMARY_VARIANT_NAMES = ("sharpened", "threshold", "soft_contrast", "opencv_adaptive")
STRONG_VARIANT_SCORE = 250.0


def find_pdf_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.pdf"))


def convert_pdf_to_images(pdf_path: Path, poppler_path: str):
    return convert_from_path(
        pdf_path=str(pdf_path),
        poppler_path=poppler_path,
        fmt="png",
        dpi=300,
    )


def _select_best_variant(
    image,
    variants: list[tuple[str, object]],
    api_base_url: str,
    model_name: str,
) -> tuple[str, object, str, float]:
    """Run full-page OCR on all preprocessing variants CONCURRENTLY and return the best one.

    Returns:
        (best_variant_name, best_variant_image, best_text, best_score)
    """
    def _ocr_one_variant(variant_name: str, variant_image) -> tuple[str, object, str, float]:
        image_b64 = image_to_base64(variant_image)
        extracted = call_paddle_ocr(
            api_base_url=api_base_url,
            model_name=model_name,
            image_b64=image_b64,
        )
        candidate_text = clean_ocr_text(extracted)
        candidate_score = score_ocr_text(candidate_text)
        print(f"    * Variant {variant_name}: score={candidate_score:.2f}, chars={len(candidate_text)}")
        return variant_name, variant_image, candidate_text, candidate_score

    results: list[tuple[str, object, str, float]] = []
    with ThreadPoolExecutor(max_workers=MAX_VARIANT_WORKERS) as pool:
        futures = [
            pool.submit(_ocr_one_variant, variant_name, variant_image)
            for variant_name, variant_image in variants
        ]
        for future in as_completed(futures):
            results.append(future.result())

    return max(results, key=lambda r: r[3])


def format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _extract_full_text_by_slicing(
    variant_image,
    variant_name: str,
    api_base_url: str,
    model_name: str,
) -> tuple[str, float]:
    """Slice the image vertically and merge OCR results from all slices CONCURRENTLY.

    Returns:
        (merged_text, merged_score)
    """
    slices = split_image_vertically(
        variant_image,
        segment_count=SEGMENT_COUNT,
        overlap_ratio=SLICE_OVERLAP,
    )
    total = len(slices)

    def _ocr_one_slice(chunk_index: int, chunk_image) -> tuple[int, str]:
        image_b64 = image_to_base64(chunk_image)
        extracted = call_paddle_ocr(
            api_base_url=api_base_url,
            model_name=model_name,
            image_b64=image_b64,
        )
        chunk_text = clean_ocr_text(extracted)
        if is_runaway_ocr_text(chunk_text):
            print(
                f"    * Slice {variant_name}#{chunk_index}/{total}: "
                f"rejected runaway text ({len(chunk_text)} chars)"
            )
            return chunk_index, ""
        print(f"    * Slice {variant_name}#{chunk_index}/{total}: chars={len(chunk_text)}")
        return chunk_index, chunk_text

    index_text_pairs: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_SLICE_WORKERS) as pool:
        futures = [
            pool.submit(_ocr_one_slice, idx, chunk_image)
            for idx, chunk_image in enumerate(slices, start=1)
        ]
        for future in as_completed(futures):
            index_text_pairs.append(future.result())

    # Restore original slice order before merging.
    chunk_texts = [text for _, text in sorted(index_text_pairs, key=lambda p: p[0])]
    merged = clean_ocr_text(merge_ocr_text_chunks(chunk_texts))
    return merged, score_ocr_text(merged)


def _process_page(
    page_number: int,
    total_pages: int,
    image,
    api_base_url: str,
    model_name: str,
) -> tuple[int, str]:
    """Process a single page and return (page_number, extracted_text)."""
    print(f"  - Page {page_number}/{total_pages}")

    # Step 1: Score the variants that usually win. Only OCR the rest when the
    # probe score is too low to trust.
    print(f"    [Step 1] Selecting best variant for page {page_number}...")
    primary_variants = build_ocr_image_variants(image, names=PRIMARY_VARIANT_NAMES)
    best_name, best_variant_image, full_page_text, full_page_score = _select_best_variant(
        image=image,
        variants=primary_variants,
        api_base_url=api_base_url,
        model_name=model_name,
    )
    if full_page_score < STRONG_VARIANT_SCORE:
        fallback_names = tuple(name for name in OCR_VARIANT_ORDER if name not in PRIMARY_VARIANT_NAMES)
        print(f"    [Step 1] Probe score {full_page_score:.2f} is low; trying {', '.join(fallback_names)}")
        fallback_variants = build_ocr_image_variants(image, names=fallback_names)
        fallback_name, fallback_image, fallback_text, fallback_score = _select_best_variant(
            image=image,
            variants=fallback_variants,
            api_base_url=api_base_url,
            model_name=model_name,
        )
        if fallback_score > full_page_score:
            best_name, best_variant_image, full_page_text, full_page_score = (
                fallback_name,
                fallback_image,
                fallback_text,
                fallback_score,
            )
    print(f"    -> Best variant: {best_name} (score={full_page_score:.2f}, chars={len(full_page_text)})")

    # Step 2: Always slice the best variant concurrently to capture the full content.
    print(f"    [Step 2] Slicing '{best_name}' into {SEGMENT_COUNT} segments ({MAX_SLICE_WORKERS} concurrent)...")
    sliced_text, sliced_score = _extract_full_text_by_slicing(
        variant_image=best_variant_image,
        variant_name=best_name,
        api_base_url=api_base_url,
        model_name=model_name,
    )
    print(f"    -> Sliced result: score={sliced_score:.2f}, chars={len(sliced_text)}")

    # Step 3: Pick the better result.
    if sliced_score >= full_page_score:
        page_text = sliced_text
        chosen = "sliced"
    else:
        page_text = full_page_text
        chosen = "full_page"

    print(f"    -> Page {page_number} chosen: {chosen}")
    return page_number, page_text or "_No text extracted._"


def process_pdf(pdf_path: Path, poppler_path: str, api_base_url: str, model_name: str) -> str:
    print(f"Processing PDF: {pdf_path.name}")
    pages = convert_pdf_to_images(pdf_path=pdf_path, poppler_path=poppler_path)
    return process_images(pages, title=pdf_path.name, api_base_url=api_base_url, model_name=model_name)


def process_images(pages: list, title: str, api_base_url: str, model_name: str) -> str:
    """OCR already-rendered page images and return the raw Markdown."""
    start_time = time.time()
    total_pages = len(pages)
    print(f"  Total pages: {total_pages} | Page workers: {MAX_PAGE_WORKERS} | "
          f"Variant workers: {MAX_VARIANT_WORKERS} | Slice workers: {MAX_SLICE_WORKERS}")

    page_results: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=MAX_PAGE_WORKERS) as pool:
        futures = [
            pool.submit(_process_page, page_num, total_pages, image, api_base_url, model_name)
            for page_num, image in enumerate(pages, start=1)
        ]
        for future in as_completed(futures):
            page_results.append(future.result())

    # Reassemble pages in original order.
    page_results.sort(key=lambda p: p[0])

    markdown_sections: list[str] = [f"# OCR Result: {title}", ""]
    for page_number, page_text in page_results:
        markdown_sections.extend(
            [
                f"## Page {page_number}",
                "",
                page_text,
                "",
            ]
        )

    end_time = time.time()
    duration = end_time - start_time
    print(f"Finished processing {title} in {format_duration(duration)}")

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
    start_total = time.time()
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
    correct_markdown_file(
        OUTPUT_FILE,
        OUTPUT_CLEAR_FILE,
        host=settings["OLLAMA_HOST"],
        model=settings["OLLAMA_MODEL"],
    )
    write_docx(OUTPUT_CLEAR_FILE, OUTPUT_DOCX_FILE)
    end_total = time.time()
    
    print("-" * 30)
    print(f"OCR completed. Output saved to: {OUTPUT_FILE}")
    print(f"Cleaned output saved to: {OUTPUT_CLEAR_FILE}")
    print(f"Word document saved to: {OUTPUT_DOCX_FILE}")
    print(f"Total time elapsed: {format_duration(end_total - start_total)}")
    print("-" * 30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
