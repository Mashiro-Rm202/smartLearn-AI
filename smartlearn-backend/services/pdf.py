from io import BytesIO

from pypdf import PdfReader

MAX_PAGES = 30


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    reader = PdfReader(BytesIO(pdf_bytes))

    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"PDF must contain at most {MAX_PAGES} pages")

    return [
        {
            "page": page_number,
            "text": (page.extract_text() or "").strip(),
        }
        for page_number, page in enumerate(reader.pages, start=1)
    ]