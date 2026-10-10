"""Text cleaning shared by training and inference.

The SAME function is used in train_model.py and at prediction time so the
model never sees differently-processed text than it was trained on.
"""
import re

# Tokens that normal punctuation stripping would destroy.
_TOKEN_MAP = [
    (r"c\+\+", " cpp "),
    (r"c#", " csharp "),
    (r"\.net\b", " dotnet "),
    (r"node\.?js", " nodejs "),
    (r"react\.?js", " react "),
    (r"vue\.?js", " vue "),
    (r"express\.?js", " expressjs "),
    (r"scikit[- ]learn", " scikitlearn "),
    (r"ci\s*/\s*cd", " cicd "),
    (r"\bml\b", " machine learning "),
]

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"(\+?\d[\d\s().-]{8,}\d)")
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")

STOPWORDS = frozenset(
    """a an and are as at be been by for from has have in is it its of on or that the
    to was were will with this these those i me my we our you your he she they them
    their his her not no but if then than so such can could would should may might
    also into over under about across per via using used use""".split()
)


def clean_text(text: str, remove_stopwords: bool = False) -> str:
    """Lower-case, remove URLs/emails/phones, normalise tech tokens."""
    if not text:
        return ""
    t = text.lower()
    t = _URL.sub(" ", t)
    t = _EMAIL.sub(" ", t)
    t = _PHONE.sub(" ", t)
    for pattern, repl in _TOKEN_MAP:
        t = re.sub(pattern, repl, t)
    t = _NON_ALNUM.sub(" ", t)
    t = _SPACES.sub(" ", t).strip()
    if remove_stopwords:
        t = " ".join(w for w in t.split() if w not in STOPWORDS)
    return t
