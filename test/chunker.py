import pprint

from fastapi import FastAPI, File, UploadFile, HTTPException
import fitz
import re
from langchain.schema import Document
from typing import List
import logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## 1. EXTRACT TEXT FROM PDF
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
from fastapi import HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True):
    """
    Extracts text from a PDF using PyMuPDF.
    If the text is missing from a page, applies OCR for better results.

    Args:
        pdf_path (str): Path to the PDF file.
        use_ocr (bool): Whether to use OCR for non-text pages.

    Returns:
        list: List of text content from each page.
    """
    extracted_texts = []

    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()

                # If no text is found and OCR is enabled, apply OCR
                if not text and use_ocr:
                    logger.warning(f"No text found on page {page_num}, applying OCR...")
                    text = apply_ocr_to_page(page)

                extracted_texts.append(text)

        return extracted_texts

    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail="PDF text extraction failed")


def apply_ocr_to_page(page):
    """
    Applies OCR to a PDF page using Tesseract.

    Args:
        page (fitz.Page): A page object from PyMuPDF.

    Returns:
        str: Extracted text using OCR.
    """
    try:
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed on a page: {e}")
        return ""  # Return empty text if OCR fails


def is_valid_paragraph(text):
    if not text:
        return False

    words = text.split()
    if len(words) < 3:
        return False

    return True


def split_text_into_paragraphs(text):
    paragraphs = re.split(r'\n{2,}', text)
    paragraphs = [p.replace('\n', ' ').strip() for p in paragraphs]
    return [p for p in paragraphs if is_valid_paragraph(p)]


def intelligent_text_chunking(paragraphs, min_size=300, max_size=500):
    documents = []
    current_doc = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        if current_size + para_size > max_size:
            if current_doc:
                documents.append("\n\n".join(current_doc))
                current_doc = []
                current_size = 0

        current_doc.append(para)
        current_size += para_size

        if current_size >= min_size:
            documents.append("\n\n".join(current_doc))
            current_doc = []
            current_size = 0

    if current_doc and len("\n\n".join(current_doc)) >= min_size:
        documents.append("\n\n".join(current_doc))

    return documents

## todo : add metadata from .bib file
def pdf_to_langchain_docs(pdf_bytes, filename, min_chunk_size=300, max_chunk_size=500):
    try:
        with open(filename, "wb") as f:
            f.write(pdf_bytes)

        pages_text = extract_text_from_pdf(filename)
        langchain_docs = []

        for page_num, text in enumerate(pages_text):
            paragraphs = split_text_into_paragraphs(text)
            chunked_documents = intelligent_text_chunking(paragraphs, min_chunk_size, max_chunk_size)

            for chunk in chunked_documents:
                langchain_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "chunk_size": len(chunk)
                        }
                    )
                )

        return langchain_docs
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail="PDF processing failed")


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()
    documents = pdf_to_langchain_docs(file_bytes, file.filename)
    response = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    return {"documents": response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)