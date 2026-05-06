# OCR PDF Project

## Project structure

```text
OCR/
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── input/
├── output/
└── src/
    ├── main.py
    └── utils.py
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

##
Model you should install
https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

Used https://huggingface.co/bmd1905/vietnamese-correction-v2 if you have gpu


## Run

1. Put scanned PDF files into `input/`.
2. Check values in `.env`.
3. Run:

```powershell
python src/main.py
```
