from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageSequence, UnidentifiedImageError

from main import convert_pdf_to_images, process_images
from ollama_correct import correct_markdown_text
from utils import load_settings, setup_logging


setup_logging()
LOGGER = logging.getLogger("ocr_api")

SETTINGS = load_settings()
API_KEY = os.getenv("API_KEY", "").strip()
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024
# Each job already fans out many requests to the OCR server.
JOB_SLOTS = threading.BoundedSemaphore(int(os.getenv("MAX_CONCURRENT_JOBS", "1")))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

app = FastAPI(title="OCR scan to Markdown")


def _check_api_key(x_api_key: str | None) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _save_upload(upload: UploadFile, target: Path) -> None:
    written = 0
    with target.open("wb") as handle:
        while chunk := upload.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            handle.write(chunk)
    if written == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")


def _load_pages(path: Path) -> list:
    with path.open("rb") as handle:
        is_pdf = handle.read(5) == b"%PDF-"

    if is_pdf:
        return convert_pdf_to_images(pdf_path=path, poppler_path=SETTINGS["POPPLER_PATH"])

    try:
        with Image.open(path) as image:
            return [frame.convert("RGB") for frame in ImageSequence.Iterator(image)]
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file. Send a PDF or an image (PNG, JPG, TIFF, BMP, WEBP).",
        ) from None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ocr")
def ocr(
    file: UploadFile = File(..., description="Scanned PDF or image"),
    correct: bool = Query(True, description="Fix wrong and duplicate words with Ollama"),
    x_api_key: str | None = Header(default=None),
) -> Response:
    _check_api_key(x_api_key)

    original_name = Path(file.filename or "scan").name
    suffix = Path(original_name).suffix.lower()
    if suffix and suffix != ".pdf" and suffix not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix}")

    with tempfile.TemporaryDirectory(prefix="ocr_api_") as workdir:
        upload_path = Path(workdir) / f"upload{suffix or '.bin'}"
        _save_upload(file, upload_path)

        with JOB_SLOTS:
            LOGGER.info("OCR job started: %s", original_name)
            pages = _load_pages(upload_path)
            if not pages:
                raise HTTPException(status_code=422, detail="No pages found in the file")

            markdown = process_images(
                pages,
                title=original_name,
                api_base_url=SETTINGS["PADDLE_API_URL"],
                model_name=SETTINGS["MODEL_NAME"],
            )

    headers = {"X-Page-Count": str(len(pages)), "X-OCR-Correction": "skipped"}
    if correct:
        try:
            markdown = correct_markdown_text(
                markdown,
                host=SETTINGS["OLLAMA_HOST"],
                model=SETTINGS["OLLAMA_MODEL"],
            )
            headers["X-OCR-Correction"] = "applied"
        except Exception:
            # Keep the OCR result rather than throwing away minutes of work.
            LOGGER.exception("Ollama correction failed for %s; returning raw OCR", original_name)
            headers["X-OCR-Correction"] = "failed"

    download_name = f"{Path(original_name).stem or 'scan'}.md"
    headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(download_name)}"
    LOGGER.info("OCR job finished: %s (%s pages)", original_name, len(pages))
    return Response(content=markdown, media_type="text/markdown; charset=utf-8", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0").strip(),
        port=int(os.getenv("API_PORT", "8000")),
    )
