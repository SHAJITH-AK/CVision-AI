"""PDF validation, text extraction and structure detection."""
import re
from typing import Dict, List

import pdfplumber

MAX_PAGES = 15
MIN_TEXT_CHARS = 150  # below this we assume a scanned / image-only PDF


class ResumeParseError(Exception):
    """Raised with a user-friendly message when a PDF cannot be used."""


def is_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(5) == b"%PDF-"
    except OSError:
        return False


def extract_text(path: str) -> Dict:
    if not is_pdf(path):
        raise ResumeParseError("The file is not a valid PDF (missing PDF header).")
    try:
        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
            if n_pages == 0:
                raise ResumeParseError("The PDF has no pages.")
            if n_pages > MAX_PAGES:
                raise ResumeParseError(f"The PDF has {n_pages} pages; the limit is {MAX_PAGES}. Upload only your resume.")
            pages = [(p.extract_text() or "") for p in pdf.pages]
    except ResumeParseError:
        raise
    except Exception as exc:  # corrupt, encrypted, etc.
        raise ResumeParseError(
            "The PDF could not be read. It may be corrupted or password-protected.") from exc

    text = "\n".join(pages).replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < MIN_TEXT_CHARS:
        raise ResumeParseError(
            "Very little text could be extracted. This usually means the PDF is a scan or an image. "
            "Export your resume as a text-based PDF (e.g. 'Save as PDF' from Word or Google Docs) and try again. "
            "OCR is not enabled in this version.")
    return {"text": text, "pages": n_pages, "characters": len(text)}


# ---------- structure detection ----------
_SECTION_PATTERNS = {
    "education": r"^\s*(education|academic (background|qualifications?)|qualifications?)\b",
    "experience": r"^\s*((work|professional|relevant) )?experience\b|^\s*employment( history)?\b|^\s*internships?\b",
    "projects": r"^\s*(academic |personal |key )?projects?\b",
    "certifications": r"^\s*(certifications?|certificates?|licenses?|courses)\b",
    "skills": r"^\s*(technical )?skills\b|^\s*core competencies\b",
}
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{8,}\d)(?!\d)")
_LINK = re.compile(r"(linkedin\.com|github\.com)", re.I)
_YEARS = re.compile(r"(\d{1,2})\+?\s*(?:years?|yrs?)\b(?:\s+of)?\s*(?:\w+\s+){0,2}experience", re.I)
_DEGREE = re.compile(r"\b(b\.?\s?sc|b\.?\s?tech|b\.?\s?e\b|b\.?c\.?a|bachelor|master|m\.?\s?sc|m\.?\s?tech|mca|mba|ph\.?d|diploma)\b", re.I)
_DATE_RANGE = re.compile(r"((?:19|20)\d{2})\s*[-–—to]+\s*((?:19|20)\d{2}|present|current)", re.I)


def detect_sections(text: str) -> Dict[str, bool]:
    found = {k: False for k in _SECTION_PATTERNS}
    for line in text.splitlines():
        s = line.strip()
        if not s or len(s) > 60:
            continue
        for key, pat in _SECTION_PATTERNS.items():
            if re.search(pat, s, re.I):
                found[key] = True
    # fall back to content signals when headings were not parsed
    if not found["education"] and _DEGREE.search(text):
        found["education"] = True
    return found


def detect_contact(text: str) -> Dict[str, bool]:
    return {"email": bool(_EMAIL.search(text)), "phone": bool(_PHONE.search(text)),
            "profile_link": bool(_LINK.search(text))}


def estimate_years_experience(text: str) -> float:
    """Best-effort estimate: explicit 'N years of experience' or the sum of date ranges. May be inaccurate."""
    explicit = [int(m.group(1)) for m in _YEARS.finditer(text)]
    if explicit:
        return float(max(explicit))
    total = 0
    for m in _DATE_RANGE.finditer(text):
        start = int(m.group(1))
        end_raw = m.group(2).lower()
        end = 2026 if end_raw in ("present", "current") else int(end_raw)
        if 0 < end - start <= 15:
            total += end - start
    return float(min(total, 30))
