"""Text preprocessing.

Self-contained on purpose: no NLTK corpus downloads are needed, so training and
inference always use exactly the same pipeline on every machine / container.
"""
from __future__ import annotations

import html
import re
from typing import Iterable

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
SPACE_PATTERN = re.compile(r"\s+")

STOP_WORDS = frozenset(
    """a about above after again against all am an and any are aren as at be because been before
    being below between both but by can cannot could couldn did didn do does doesn doing don down
    during each few for from further had hadn has hasn have haven having he her here hers herself
    him himself his how i if in into is isn it its itself just let me more most mustn my myself no
    nor not now of off on once only or other ought our ours ourselves out over own same shan she
    should shouldn so some such than that the their theirs them themselves then there these they
    this those through to too under until up very was wasn we were weren what when where which
    while who whom why will with won would wouldn you your yours yourself yourselves s t d ll m o
    re ve y""".split()
)


def lemmatize(token: str) -> str:
    """Very small rule-based plural stripper (replaces the WordNet lemmatizer)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def clean_text(text: object) -> str:
    if text is None:
        return ""

    normalized = html.unescape(str(text)).lower().strip()
    if not normalized:
        return ""

    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = HTML_PATTERN.sub(" ", normalized)
    normalized = NON_ALPHA_PATTERN.sub(" ", normalized)
    normalized = SPACE_PATTERN.sub(" ", normalized).strip()

    tokens: list[str] = []
    for token in normalized.split():
        if token in STOP_WORDS:
            continue
        lemma = lemmatize(token)
        if len(lemma) > 1:
            tokens.append(lemma)

    return " ".join(tokens)


def combine_fields(title: object, body: object) -> str:
    title_text = "" if title is None else str(title).strip()
    body_text = "" if body is None else str(body).strip()
    return f"{title_text} {body_text}".strip()


def batch_clean(texts: Iterable[object]) -> list[str]:
    return [clean_text(text) for text in texts]
