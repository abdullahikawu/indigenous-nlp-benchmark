"""
Group 07 — Gbagyi Indigenous NLP Benchmark
CSC 406 Assignment 1

Core, importable implementation for data collection, Unicode-safe
preprocessing, custom tokenisation, Zipf analysis, and n-gram models.

This module is the single source of truth. The notebook imports from it.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import numpy as np
import requests
from bs4 import BeautifulSoup, Comment

LOGGER = logging.getLogger("group07.gbagyi")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_JSONL_PATH = REPO_ROOT / "data" / "gbagyi" / "raw" / "raw_data_group_07.jsonl"
PROCESSED_CORPUS_PATH = REPO_ROOT / "data" / "gbagyi" / "processed" / "cleaned_corpus_group_07.txt"
TEST_PATH = REPO_ROOT / "tests" / "test_gbagyi_unseen.txt"

USER_AGENT = (
    "Group07-GbagyiNLP/1.0 "
    "(CSC406 Indigenous Language AI Benchmark; university coursework; "
    "polite educational scrape)"
)
REQUEST_TIMEOUT = 25
REQUEST_DELAY_SECONDS = 0.7
MAX_RETRIES = 3

# Protestant New Testament chapter inventory (standard book codes used by bible.com).
NT_CHAPTERS: List[Tuple[str, int]] = [
    ("MAT", 28),
    ("MRK", 16),
    ("LUK", 24),
    ("JHN", 21),
    ("ACT", 28),
    ("ROM", 16),
    ("1CO", 16),
    ("2CO", 13),
    ("GAL", 6),
    ("EPH", 6),
    ("PHP", 4),
    ("COL", 4),
    ("1TH", 5),
    ("2TH", 3),
    ("1TI", 6),
    ("2TI", 4),
    ("TIT", 3),
    ("PHM", 1),
    ("HEB", 13),
    ("JAS", 5),
    ("1PE", 5),
    ("2PE", 3),
    ("1JN", 5),
    ("2JN", 1),
    ("3JN", 1),
    ("JUD", 1),
    ("REV", 22),
]

GBAGYI_BIBLE_EDITIONS = (
    {"version_id": 1621, "abbr": "GAW", "label": "Alkawali Woiwoyi (Biblica 1997)"},
    {"version_id": 4607, "abbr": "GNB", "label": "Gbagyi Contemporary Bible (Biblica 2025)"},
)

ENGLISH_FUNCTION_WORDS = {
    "the", "and", "of", "to", "in", "is", "for", "that", "this", "with",
    "from", "are", "was", "were", "be", "as", "by", "on", "or", "an",
    "it", "at", "not", "you", "your", "we", "they", "their", "have",
    "has", "had", "but", "if", "which", "will", "can", "all", "about",
    "into", "than", "then", "also", "more", "other", "when", "where",
    "who", "what", "how", "there", "these", "those", "his", "her",
    "its", "our", "them", "him", "she", "he", "i", "a",
}

NAVIGATION_NOISE = {
    "listen", "highlight", "copy", "compare", "share", "sign up",
    "sign in", "currently selected", "learn more", "popular bible",
    "want to have your highlights", "copyright", "all rights reserved",
    "used with permission", "bible app", "get the youversion",
}

# Catalogue / encyclopedia pages may be stored in raw JSONL for provenance
# but must not contribute English sentences to the processed Gbagyi corpus.
PROVENANCE_ONLY_MARKERS = (
    "wikipedia.org",
    "scriptureearth.org",
    "bible.com/versions/",
    "bible.com/languages/",
)

ENGLISH_CATALOGUE_PHRASES = (
    "bible versions",
    "biblica is a global ministry",
    "you can help wikipedia",
    "displacement from lands",
    "machine-translated version",
    "consider adding a topic",
    "help the kids in your life",
    "other versions by biblica",
    "alternative language names",
    "currently selected",
    "learn more about",
    "get the youversion",
    "this page was last edited",
    "from wikipedia, the free encyclopedia",
    "jump to navigation",
    "jump to search",
    "read the bible",
    "audio bible",
    "download the bible",
    "new testament in",
    "the gbagyi contemporary bible",
    "considered the contemporary language",
    "bibelen på",
    "hoffnung für alle",
    "kinh thánh",
)

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?።])\s+")
TOKEN_RE = re.compile(
    r"[0-9]+(?:[.,][0-9]+)*"
    r"|[\w]+(?:-[\w]+)*"
    r"|[^\w\s]",
    flags=re.UNICODE,
)
WORD_TOKEN_RE = re.compile(r"[\w]+(?:-[\w]+)*", flags=re.UNICODE)

_ROBOTS_CACHE: Dict[str, Optional[RobotFileParser]] = {}


def configure_logging(level: int = logging.INFO) -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)


def repo_paths() -> Dict[str, Path]:
    return {
        "root": REPO_ROOT,
        "raw": RAW_JSONL_PATH,
        "processed": PROCESSED_CORPUS_PATH,
        "test": TEST_PATH,
    }


# ---------------------------------------------------------------------------
# HTTP / robots
# ---------------------------------------------------------------------------

def _robots_parser(base_url: str) -> Optional[RobotFileParser]:
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin in _ROBOTS_CACHE:
        return _ROBOTS_CACHE[origin]
    robots_url = f"{origin}/robots.txt"
    parser = RobotFileParser()
    try:
        response = requests.get(
            robots_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code >= 400:
            _ROBOTS_CACHE[origin] = None
            return None
        parser.parse(response.text.splitlines())
        _ROBOTS_CACHE[origin] = parser
        return parser
    except requests.RequestException:
        _ROBOTS_CACHE[origin] = None
        return None


def robots_allowed(url: str) -> bool:
    parser = _robots_parser(url)
    if parser is None:
        return True
    return parser.can_fetch(USER_AGENT, url)


def fetch_url(url: str, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Polite GET with retries. Never bypasses robots.txt or access controls."""
    record: Dict[str, Any] = {
        "url": url,
        "ok": False,
        "status_code": None,
        "text": "",
        "error": None,
        "robots_allowed": robots_allowed(url),
    }
    if not record["robots_allowed"]:
        record["error"] = "disallowed_by_robots"
        LOGGER.warning("robots.txt disallows %s", url)
        return record

    client = session or requests.Session()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en;q=0.8",
    }
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            record["status_code"] = response.status_code
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            record["text"] = response.text
            record["ok"] = True
            return record
        except requests.RequestException as exc:
            last_error = str(exc)
            LOGGER.info("attempt %s failed for %s: %s", attempt, url, exc)
            time.sleep(min(2 ** attempt, 8))
    record["error"] = last_error
    return record


# ---------------------------------------------------------------------------
# Source URL inventory
# ---------------------------------------------------------------------------

def gbagyi_bible_chapter_urls() -> List[str]:
    urls: List[str] = []
    for edition in GBAGYI_BIBLE_EDITIONS:
        for book, n_chapters in NT_CHAPTERS:
            for chapter in range(1, n_chapters + 1):
                urls.append(
                    f"https://www.bible.com/bible/{edition['version_id']}/"
                    f"{book}.{chapter}.{edition['abbr']}"
                )
    return urls


def supplementary_gbagyi_urls() -> List[str]:
    return [
        "https://www.bible.com/versions/1621-gaw-alkawali-woiwoyi",
        "https://www.bible.com/versions/4607-gnb-gbagyi-nyizeyenya-baibwulu-shekwoyi-%C6%81%C9%99dagbma",
        "https://www.bible.com/languages/gbr",
        "https://en.wikipedia.org/wiki/Gbagyi_language",
        "https://en.wikipedia.org/wiki/Gbagyi_people",
        "https://www.scriptureearth.org/00eng.php?iso=gbr",
    ]


def default_gbagyi_urls() -> List[str]:
    return gbagyi_bible_chapter_urls() + supplementary_gbagyi_urls()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _strip_noise_nodes(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda node: isinstance(node, Comment)):
        comment.extract()


def _clean_extracted_line(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    text = CONTROL_CHAR_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_navigation(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in NAVIGATION_NOISE)


def extract_verses_from_next_data(html: str) -> List[str]:
    """Extract verse strings from YouVersion Next.js payload when present."""
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        return []
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    verses: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            content = node.get("content")
            usfm = str(node.get("usfm") or node.get("verseUsfm") or "")
            if isinstance(content, str) and content.strip() and ("." in usfm or node.get("type") == "verse"):
                cleaned = _clean_extracted_line(content)
                if cleaned and not cleaned.isdigit():
                    verses.append(cleaned)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return verses


def extract_verses_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    verses: List[str] = []

    for selector in (
        "span.ChapterContent_verse__57FIw",
        "span[class*='ChapterContent_verse']",
        "span[data-usfm]",
        "p.verse",
        "span.verse",
    ):
        nodes = soup.select(selector)
        if not nodes:
            continue
        for node in nodes:
            label = node.select_one("[class*='ChapterContent_label'], .label, sup")
            content_nodes = node.select("[class*='ChapterContent_content'], .content")
            if content_nodes:
                text = " ".join(_clean_extracted_line(n.get_text(" ", strip=True)) for n in content_nodes)
            else:
                text = _clean_extracted_line(node.get_text(" ", strip=True))
                if label:
                    label_text = _clean_extracted_line(label.get_text(" ", strip=True))
                    if text.startswith(label_text):
                        text = text[len(label_text):].strip()
            text = re.sub(r"^\d+\s*", "", text)
            if text:
                verses.append(text)
        if verses:
            return verses
    return verses


def extract_main_text(html: str, url: str) -> str:
    """Best-effort body-text extraction for non-verse pages."""
    verses = extract_verses_from_next_data(html)
    if not verses:
        verses = extract_verses_from_html(html)
    if verses:
        return "\n".join(verses)

    soup = BeautifulSoup(html, "html.parser")
    _strip_noise_nodes(soup)
    main = soup.find("main") or soup.find("article") or soup.find("div", {"id": "mw-content-text"}) or soup.body
    if main is None:
        return ""
    paragraphs = []
    for element in main.find_all(["p", "li", "h1", "h2", "h3"]):
        text = _clean_extracted_line(element.get_text(" ", strip=True))
        if text and not _looks_like_navigation(text):
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_raw_text(html: str, url: str) -> str:
    return extract_main_text(html, url)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def scrape_to_jsonl(url_list: Sequence[str], output_path: str | Path) -> int:
    """
    Scrape text from URLs and write JSON Lines with integer IDs.

    Returns the number of successfully written documents.
    """
    configure_logging()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    written = 0
    seen_urls = set()
    seen_hashes = set()
    today = date.today().isoformat()

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for url in url_list:
            if url in seen_urls:
                LOGGER.info("skip duplicate URL %s", url)
                continue
            seen_urls.add(url)
            fetched = fetch_url(url, session=session)
            time.sleep(REQUEST_DELAY_SECONDS)
            if not fetched["ok"]:
                LOGGER.warning("failed %s [%s] %s", url, fetched["status_code"], fetched["error"])
                continue
            raw_text = extract_raw_text(fetched["text"], url)
            if not raw_text or len(raw_text.split()) < 3:
                LOGGER.info("empty/short extraction for %s", url)
                continue
            digest = hash(re.sub(r"\s+", " ", raw_text).strip().lower())
            if digest in seen_hashes:
                LOGGER.info("skip duplicate document body from %s", url)
                continue
            seen_hashes.add(digest)
            written += 1
            record = {
                "id": written,
                "url": url,
                "date_retrieved": today,
                "raw_text": raw_text,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            LOGGER.info("wrote document %s from %s (%s chars)", written, url, len(raw_text))

    return written


# ---------------------------------------------------------------------------
# Normalisation, sentences, tokenizer
# ---------------------------------------------------------------------------

def remove_html_markup(text: str) -> str:
    return HTML_TAG_RE.sub(" ", text)


def remove_control_characters(text: str) -> str:
    text = text.replace("\r", " ")
    return CONTROL_CHAR_RE.sub("", text)


def normalize_unicode(text: str) -> str:
    """NFC normalisation preserves Gbagyi letters, including ɓ and diacritics."""
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\t", " ")
    text = re.sub(r"[ \u00a0\u1680\u180e\u2000-\u200b\u202f\u205f\u3000]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def clean_document_text(text: str) -> str:
    text = remove_html_markup(text)
    text = remove_control_characters(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


def split_sentences(text: str) -> List[str]:
    """Transparent sentence segmentation. Does not treat raw HTML lines as sentences."""
    sentences: List[str] = []
    for block in re.split(r"\n+", text):
        block = block.strip()
        if not block:
            continue
        parts = SENTENCE_SPLIT_RE.split(block)
        if len(parts) == 1 and not re.search(r"[.!?]", block):
            sentences.append(block)
            continue
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def is_mostly_english(sentence: str) -> bool:
    """
    Detect English-majority catalogue/UI sentences.

    Single-letter tokens are ignored: Gbagyi independently uses a, n, o, i
    as function items, so counting them as English would delete authentic verses.
    """
    tokens = [t.lower() for t in WORD_TOKEN_RE.findall(sentence)]
    tokens = [tok for tok in tokens if len(tok) > 1]
    if not tokens:
        return False
    english = sum(1 for tok in tokens if tok in ENGLISH_FUNCTION_WORDS)
    return (english / len(tokens)) >= 0.45


def has_gbagyi_script_evidence(text: str) -> bool:
    """True if the string contains Gbagyi-relevant Latin extensions (ɓ, ə, etc.)."""
    return any(
        unicodedata.category(ch).startswith("L") and ord(ch) > 127
        for ch in text
    )


def is_catalogue_or_encyclopedia_url(url: str) -> bool:
    lowered = url.lower()
    return any(marker in lowered for marker in PROVENANCE_ONLY_MARKERS)


def is_english_catalogue_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    return any(phrase in lowered for phrase in ENGLISH_CATALOGUE_PHRASES)


def is_boilerplate(sentence: str) -> bool:
    lowered = sentence.lower()
    if _looks_like_navigation(lowered):
        return True
    if is_english_catalogue_sentence(sentence):
        return True
    if re.fullmatch(r"[\d\W]+", sentence):
        return True
    return False


def quality_filter_sentence(sentence: str, source_url: str) -> bool:
    """
    Keep only authentic Gbagyi running text.

    Wikipedia, Scripture Earth, bible.com language-index pages, and
    bible.com version-catalogue pages are provenance-only. Chapter pages
    on bible.com/bible/ contribute Gbagyi running text after English-UI
    and held-out filters.
    """
    sentence = sentence.strip()
    if len(sentence) < 2:
        return False
    letters = [ch for ch in sentence if unicodedata.category(ch).startswith("L")]
    if len(letters) < 2:
        return False
    if is_boilerplate(sentence) or is_english_catalogue_sentence(sentence):
        return False
    if is_mostly_english(sentence):
        return False

    url = source_url.lower()
    if "wikipedia.org" in url or "scriptureearth.org" in url:
        return False
    if "bible.com/languages/" in url:
        return False
    if "bible.com/versions/" in url:
        return False
    if "bible.com/bible/" in url:
        return True
    return has_gbagyi_script_evidence(sentence)


def word_token_sequence(tokenized_line: str) -> List[str]:
    return [tok for tok in tokenized_line.split(" ") if WORD_TOKEN_RE.fullmatch(tok)]


def is_contiguous_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    n, m = len(needle), len(haystack)
    if n == 0 or n > m:
        return False
    for i in range(m - n + 1):
        if list(haystack[i : i + n]) == list(needle):
            return True
    return False


def load_heldout_word_sequences(test_path: str | Path = TEST_PATH) -> Tuple[set[str], List[List[str]]]:
    """Load instructor test lines for train/test decontamination only."""
    exact: set[str] = set()
    sequences: List[List[str]] = []
    path = Path(test_path)
    if not path.exists():
        return exact, sequences
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        exact.add(line)
        seq = word_token_sequence(line)
        if len(seq) >= 4:
            sequences.append(seq)
    return exact, sequences


def is_heldout_contamination(
    tokenized: str,
    exact_test: set[str],
    test_sequences: Sequence[Sequence[str]],
) -> Optional[str]:
    """
    Decontaminate training against the official unseen file.

    1. Exact tokenized-line match.
    2. Contiguous word-sequence containment either way when the shorter
       sequence has ≥ 4 word tokens.

    Shared function words alone are not sufficient. Non-identical related
    verses (e.g. aɓi vs aɓeye) are kept.
    """
    if tokenized in exact_test:
        return "heldout_exact"
    seq = word_token_sequence(tokenized)
    if len(seq) < 4:
        return None
    for test_seq in test_sequences:
        if is_contiguous_subsequence(test_seq, seq) or is_contiguous_subsequence(seq, test_seq):
            return "heldout_containment"
    return None


def custom_tokenizer(text: str) -> str:
    """
    Custom Unicode-aware Gbagyi tokenizer.

    - lowercase with str.lower() (Unicode-safe; Ɓ -> ɓ)
    - detach punctuation
    - keep internal hyphens (bui-bui, zaho-zahoyi)
    - single-space output
    """
    if text is None:
        return ""
    text = clean_document_text(str(text)).lower()
    tokens = TOKEN_RE.findall(text)
    tokens = [tok for tok in tokens if tok.strip()]
    return " ".join(tokens)


def tokenize_sentence(sentence: str) -> str:
    return custom_tokenizer(sentence)


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
    return records


def build_processed_corpus(
    raw_path: str | Path = RAW_JSONL_PATH,
    output_path: str | Path = PROCESSED_CORPUS_PATH,
) -> Dict[str, Any]:
    records = load_jsonl(raw_path)
    exact_test, test_sequences = load_heldout_word_sequences(TEST_PATH)
    seen_sentences: set[str] = set()
    kept: List[str] = []
    contributed: Dict[str, int] = defaultdict(int)
    stats = {
        "documents": len(records),
        "raw_sentences": 0,
        "duplicates": 0,
        "filtered": 0,
        "heldout_exact": 0,
        "heldout_containment": 0,
        "final_sentences": 0,
        "source_urls": sorted({str(r.get("url", "")) for r in records}),
        "contributing_urls": [],
    }

    for record in records:
        url = str(record.get("url", ""))
        cleaned = clean_document_text(str(record.get("raw_text", "")))
        for sentence in split_sentences(cleaned):
            stats["raw_sentences"] += 1
            if not quality_filter_sentence(sentence, url):
                stats["filtered"] += 1
                continue
            tokenized = tokenize_sentence(sentence)
            if not tokenized:
                stats["filtered"] += 1
                continue
            contamination = is_heldout_contamination(tokenized, exact_test, test_sequences)
            if contamination:
                stats[contamination] = int(stats.get(contamination, 0)) + 1
                stats["filtered"] += 1
                continue
            if tokenized in seen_sentences:
                stats["duplicates"] += 1
                continue
            seen_sentences.add(tokenized)
            kept.append(tokenized)
            contributed[url] += 1

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8", newline="\n")
    stats["final_sentences"] = len(kept)
    stats["contributing_urls"] = sorted(contributed)
    stats["contributed_sentence_counts"] = dict(contributed)
    stats["provenance"] = provenance_rows(records, dict(contributed))
    return stats


def classify_source_url(url: str) -> Dict[str, str]:
    """Map a stored URL to a human-readable provenance class."""
    lowered = url.lower()
    if "en.wikipedia.org/wiki/gbagyi_language" in lowered:
        return {
            "source_name": "Wikipedia: Gbagyi language",
            "language_version": "English encyclopedia (not Gbagyi running text)",
            "source_class": "encyclopedia_english",
        }
    if "en.wikipedia.org/wiki/gbagyi_people" in lowered:
        return {
            "source_name": "Wikipedia: Gbagyi people",
            "language_version": "English encyclopedia (not Gbagyi running text)",
            "source_class": "encyclopedia_english",
        }
    if "scriptureearth.org" in lowered:
        return {
            "source_name": "Scripture Earth Gbagyi (gbr) index",
            "language_version": "English catalogue",
            "source_class": "catalogue_english",
        }
    if "bible.com/languages/" in lowered:
        return {
            "source_name": "Bible.com language index (gbr)",
            "language_version": "English navigation / language catalogue",
            "source_class": "catalogue_english",
        }
    if "bible.com/versions/1621" in lowered:
        return {
            "source_name": "Bible.com version page: Alkawali Woiwoyi (GAW)",
            "language_version": "English version-catalogue page (not processed as Gbagyi)",
            "source_class": "catalogue_version",
        }
    if "bible.com/versions/4607" in lowered:
        return {
            "source_name": "Bible.com version page: Gbagyi Contemporary Bible (GNB)",
            "language_version": "English version-catalogue page (not processed as Gbagyi)",
            "source_class": "catalogue_version",
        }
    if "/bible/" in lowered and lowered.endswith(".gaw"):
        return {
            "source_name": "Biblica Alkawali Woiwoyi (GAW 1621) chapter",
            "language_version": "Gbagyi (GAW / Alkawali Woiwoyi)",
            "source_class": "gbagyi_scripture_gaw",
        }
    if "/bible/" in lowered and lowered.endswith(".gnb"):
        return {
            "source_name": "Biblica Gbagyi Contemporary Bible (GNB 4607) chapter",
            "language_version": "Gbagyi (GNB / Shekwoyi Ɓədagbma)",
            "source_class": "gbagyi_scripture_gnb",
        }
    return {
        "source_name": urlparse(url).netloc or "unknown",
        "language_version": "unclassified fetched page",
        "source_class": "other",
    }


def provenance_rows(
    records: Sequence[Dict[str, Any]],
    contributed: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    One row per stored URL: source identity, retrieval date, contribution.

    HTTP status is not a JSONL field (autograder schema is id/url/date_retrieved/raw_text).
    Every stored record was a successful fetch; failed URLs were omitted.
    """
    rows: List[Dict[str, Any]] = []
    for record in records:
        url = str(record.get("url", ""))
        meta = classify_source_url(url)
        n_sent = int(contributed.get(url, 0))
        if meta["source_class"] in {"encyclopedia_english", "catalogue_english"}:
            decision = "raw provenance only; English catalogue/encyclopedia excluded from processed corpus"
        elif meta["source_class"] == "catalogue_version":
            decision = "raw provenance only; bible.com version-catalogue pages excluded from processed Gbagyi text"
        elif n_sent:
            decision = "accepted as authentic Gbagyi scripture after verse extraction, English-UI filter, and held-out decontamination"
        else:
            decision = "fetched scripture page contributed no unique kept sentences after filtering"
        rows.append(
            {
                "url": url,
                "source_name": meta["source_name"],
                "language_version": meta["language_version"],
                "source_class": meta["source_class"],
                "date_retrieved": str(record.get("date_retrieved", "")),
                "http_status": "200 (successful fetch; failed URLs were not stored)",
                "contributed_processed_sentences": n_sent,
                "contributed_processed_gbagyi": n_sent > 0,
                "filtering_decision": decision,
            }
        )
    return rows


def load_corpus_lines(path: str | Path = PROCESSED_CORPUS_PATH) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip()]


def corpus_tokens(lines: Sequence[str], words_only: bool = False) -> List[str]:
    tokens: List[str] = []
    for line in lines:
        for tok in line.split(" "):
            if not tok:
                continue
            if words_only and not WORD_TOKEN_RE.fullmatch(tok):
                continue
            tokens.append(tok)
    return tokens


def corpus_statistics(path: str | Path = PROCESSED_CORPUS_PATH) -> Dict[str, Any]:
    lines = load_corpus_lines(path)
    tokens = corpus_tokens(lines)
    word_tokens = corpus_tokens(lines, words_only=True)
    lengths = [len(line.split(" ")) for line in lines]
    freq = Counter(word_tokens)
    return {
        "sentences": len(lines),
        "tokens": len(tokens),
        "word_tokens": len(word_tokens),
        "vocabulary": len(set(word_tokens)),
        "token_vocabulary": len(set(tokens)),
        "avg_sentence_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_sentence_length": float(np.median(lengths)) if lengths else 0.0,
        "min_sentence_length": int(min(lengths)) if lengths else 0,
        "max_sentence_length": int(max(lengths)) if lengths else 0,
        "most_frequent": freq.most_common(20),
    }


def validate_jsonl(path: str | Path = RAW_JSONL_PATH) -> List[str]:
    errors: List[str] = []
    required = {"id", "url", "date_retrieved", "raw_text"}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc})")
                continue
            missing = required - set(entry)
            if missing:
                errors.append(f"line {line_no}: missing {missing}")
                continue
            if not isinstance(entry["id"], int):
                errors.append(f"line {line_no}: id must be int")
            if not all(isinstance(entry[k], str) for k in ("url", "date_retrieved", "raw_text")):
                errors.append(f"line {line_no}: url/date_retrieved/raw_text must be str")
            if not entry["raw_text"].strip():
                errors.append(f"line {line_no}: empty raw_text")
    return errors


def validate_processed_corpus(path: str | Path = PROCESSED_CORPUS_PATH) -> List[str]:
    errors: List[str] = []
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return [f"not valid UTF-8: {exc}"]
    if "\r" in text:
        errors.append("carriage return present")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    for line_no, line in enumerate(lines, 1):
        if line == "":
            errors.append(f"line {line_no}: empty line")
            continue
        if line != line.lower():
            errors.append(f"line {line_no}: not lowercase")
        if line.startswith(" ") or line.endswith(" "):
            errors.append(f"line {line_no}: leading/trailing space")
        if "  " in line:
            errors.append(f"line {line_no}: consecutive spaces")
        if "\t" in line:
            errors.append(f"line {line_no}: tab")
        if any(ord(ch) < 32 for ch in line):
            errors.append(f"line {line_no}: control character")
    return errors


# ---------------------------------------------------------------------------
# Stop words (attested only)
# ---------------------------------------------------------------------------

NP_FIELD_MS = (
    "Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi "
    "(Niger State speaker interviews; NP/pronoun/demonstrative examples)."
)
BIBLICA_GAW = "Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text."
BIBLICA_GNB = "Biblica (1997/2025), Gbagyi Contemporary Bible (GNB), form attested in collected chapter text."

GBAGYI_STOP_WORDS: List[Dict[str, str]] = [
    {
        "gbagyi": "mi",
        "english": "I / my",
        "category": "pronoun / possessive",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Examples: mi in relative clauses; omi 'my'.",
    },
    {
        "gbagyi": "omi",
        "english": "my",
        "category": "possessive determiner",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Examples: omi dnanwu 'my aunt'; omi mapwi 'my sister'.",
    },
    {
        "gbagyi": "wo",
        "english": "his / him / her",
        "category": "pronoun / possessive",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Example: wo zukwoi 'his hoe'. Also " + BIBLICA_GAW,
    },
    {
        "gbagyi": "wa",
        "english": "he / she (3sg subject)",
        "category": "pronoun",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Example: n wa mi nyi 'that he/she built it'. Also " + BIBLICA_GAW + " (wa zhin ...).",
    },
    {
        "gbagyi": "wu",
        "english": "he / she",
        "category": "pronoun",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Example (11c): wu 'za ... 'he/she [guides] a person'.",
    },
    {
        "gbagyi": "ɓa",
        "english": "they (3pl)",
        "category": "pronoun",
        "confidence": "attested",
        "source": NP_FIELD_MS + " 3pl in relative clauses. Also " + BIBLICA_GAW,
    },
    {
        "gbagyi": "ye",
        "english": "this (near demonstrative). In scripture orthography the same spelling also occurs as a high-frequency verb/particle; those uses are not collapsed here.",
        "category": "demonstrative / function word",
        "confidence": "attested (demonstrative); other uses vary",
        "source": NP_FIELD_MS + " yè 'this' (ēɓí yè 'this child'). Tone often unmarked in digital text.",
    },
    {
        "gbagyi": "yi",
        "english": "that (distant demonstrative) / anaphoric particle",
        "category": "demonstrative / particle",
        "confidence": "attested",
        "source": NP_FIELD_MS + " yî 'that' (ēɓí yî 'that child'). Also " + BIBLICA_GAW,
    },
    {
        "gbagyi": "ho",
        "english": "the (optional determiner)",
        "category": "determiner",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Muta ho pye-pyei lo 'the car is fast'; hò as optional determiner.",
    },
    {
        "gbagyi": "lo",
        "english": "tense/aspect or predicative particle",
        "category": "auxiliary / particle",
        "confidence": "attested",
        "source": NP_FIELD_MS + " lo as a tense/predicative particle (Muta ho pye-pyei lo; ɓa bmya lo).",
    },
    {
        "gbagyi": "o",
        "english": "on / in (preposition); also coordinating 'and' in some examples",
        "category": "preposition / conjunction",
        "confidence": "attested (polysemous: preposition and coordinator)",
        "source": NP_FIELD_MS + " Dual function: preposition o tebulu 'on the table' / o Kuta 'in Kuta'; conjunction fwaiza o wo zukwoi 'farmer and his hoe'. Senses are not collapsed.",
    },
    {
        "gbagyi": "n",
        "english": "relativizer / linker 'that'; also a high-frequency clitic linker",
        "category": "relativizer / conjunction",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Relativiser ń / n. Also " + BIBLICA_GAW + " as a clitic linker.",
    },
    {
        "gbagyi": "nu",
        "english": "determiner / copular-focus particle",
        "category": "determiner / particle",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Determiner/focus nu (zhni shaknu mii nu). Also " + BIBLICA_GAW,
    },
    {
        "gbagyi": "zhni",
        "english": "be / become (copula)",
        "category": "auxiliary / copula",
        "confidence": "attested",
        "source": NP_FIELD_MS + " zhni shaknu 'is a pot/potter'. Also " + BIBLICA_GNB,
    },
    {
        "gbagyi": "zhin",
        "english": "be / become (copula)",
        "category": "auxiliary / copula",
        "confidence": "attested",
        "source": BIBLICA_GAW + " Orthographic counterpart of zhni (Ibrahim ɓei zhin Ishaku dada). Treated as a spelling variant, not a separate lexeme.",
    },
    {
        "gbagyi": "kwo",
        "english": "it (3sg inanimate / resumptive)",
        "category": "pronoun",
        "confidence": "attested",
        "source": NP_FIELD_MS + " kwo tu o asha nyi 'it [is] on the shelf'.",
    },
    {
        "gbagyi": "ge",
        "english": "quotative / complementizer 'that' (naming, reported speech)",
        "category": "conjunction / complementizer",
        "confidence": "attested",
        "source": BIBLICA_GAW + " MAT.1:21 wa tu wo 'ye ge Yesu 'he shall call his name Jesus'. Parallel to an English complementizer in the same verse.",
    },
    {
        "gbagyi": "nya",
        "english": "of (associative / genitive)",
        "category": "preposition / genitive marker",
        "confidence": "attested",
        "source": BIBLICA_GAW + " MAT.1:18 Zafun Gyi-gyi-yi nya 'of the Holy Spirit'. The NP field manuscript also shows genitive-like juxtaposition; nya is common in collected scripture.",
    },
    {
        "gbagyi": "to",
        "english": "not / negative particle",
        "category": "negation / particle",
        "confidence": "attested",
        "source": BIBLICA_GAW + " MAT.1:18 to ɓai gye ajen. Also " + BIBLICA_GNB + " (to ... m negation).",
    },
    {
        "gbagyi": "ntu",
        "english": "so that / in order that (purpose)",
        "category": "conjunction",
        "confidence": "attested",
        "source": BIBLICA_GAW + " MAT.1:19–21 ntu ge ... purpose/result linker.",
    },
    {
        "gbagyi": "ntuge",
        "english": "because / so that (purpose-reason)",
        "category": "conjunction",
        "confidence": "attested",
        "source": BIBLICA_GNB + " MAT.1:19 Ntuge Yusufu ...; fused form of ntu + ge in the Contemporary Bible orthography.",
    },
    {
        "gbagyi": "gmanyi",
        "english": "one / some (quantifier)",
        "category": "determiner / quantifier",
        "confidence": "attested",
        "source": NP_FIELD_MS + " ōzā gmànyí 'one person'; àzà gmànyí 'some people'.",
    },
    {
        "gbagyi": "vnyanya",
        "english": "all / whole",
        "category": "quantifier / determiner",
        "confidence": "attested",
        "source": NP_FIELD_MS + " yàɓà vnyānyā 'the whole banana'. Also " + BIBLICA_GNB,
    },
    {
        "gbagyi": "ama",
        "english": "but",
        "category": "conjunction",
        "confidence": "attested (Hausa loan used as a Gbagyi conjunction in these publications)",
        "source": BIBLICA_GAW + " MAT.1:18 Ama to ɓai gye... Hausa loan ama 'but', conventional in written Gbagyi Christian texts (Newman, Hausa; Biblica Gbagyi text).",
    },
    {
        "gbagyi": "sai",
        "english": "then / only / except",
        "category": "conjunction / particle",
        "confidence": "attested (Hausa loan used as a Gbagyi particle in these publications)",
        "source": BIBLICA_GAW + " MAT.1:19 Sai owo nugun-yi ... Hausa loan sai, conventional in written Gbagyi Christian texts.",
    },
    {
        "gbagyi": "har",
        "english": "until / even",
        "category": "conjunction / preposition",
        "confidence": "attested (Hausa loan used as a Gbagyi function word in this publication)",
        "source": BIBLICA_GAW + " MAT.1:25 har wa eɓi nugbayi ma. Hausa loan har 'until', used as a Gbagyi function word in this publication.",
    },
    {
        "gbagyi": "shi",
        "english": "then / and then (sequential)",
        "category": "conjunction / particle",
        "confidence": "attested",
        "source": BIBLICA_GAW + " MAT.1:24 Shi Yusufu ɓo kun agyewyi 'Then Joseph woke from sleep'. Sequential particle in narrative scripture.",
    },
    {
        "gbagyi": "ma",
        "english": "and (coordinator in some constructions) / 'give birth to' as a content verb — listed here only as the high-frequency coordinator/particle sense when not the main verb",
        "category": "conjunction / particle",
        "confidence": "uncertain (polysemous)",
        "source": BIBLICA_GAW + " Verbal 'bear/give birth' (wo ɓa ma eɓi) and high-frequency particle uses both occur. Do not treat every ma as a stop word.",
    },
    {
        "gbagyi": "ga",
        "english": "clause-final / focus particle (function word; exact force varies by dialect)",
        "category": "particle",
        "confidence": "uncertain",
        "source": BIBLICA_GAW + " High-frequency clause-final particle in collected scripture. No single agreed English gloss in the grammars we could inspect.",
    },
    {
        "gbagyi": "na",
        "english": "associative / high-frequency linker (exact sense varies)",
        "category": "particle / linker",
        "confidence": "uncertain",
        "source": BIBLICA_GAW + " High-frequency linker/associative in collected scripture. Published grammars we inspected do not give a single agreed gloss.",
    },
    {
        "gbagyi": "nyi",
        "english": "locative / relational particle (often clause-final)",
        "category": "particle",
        "confidence": "attested",
        "source": NP_FIELD_MS + " Clause-final nyí / nyi. Also " + BIBLICA_GAW,
    },
    {
        "gbagyi": "ɓe",
        "english": "come (light verb / motion; also appears in serial constructions)",
        "category": "auxiliary / light verb",
        "confidence": "attested",
        "source": NP_FIELD_MS + " ōzā ń wō ɓé lō nyí 'the person who is coming'. Also " + BIBLICA_GAW,
    },
    {
        "gbagyi": "ɓei",
        "english": "past / sequential auxiliary appearing before zhin in GAW genealogies",
        "category": "auxiliary",
        "confidence": "attested",
        "source": BIBLICA_GAW + " MAT.1: Ibrahim ɓei zhin Ishaku dada. Distinct from ɓe 'come' in the NP field manuscript; treated as an attested scripture-orthography auxiliary, not an invented lexeme.",
    },
    {
        "gbagyi": "a",
        "english": "they / impersonal plural prefix or pronoun (context-dependent)",
        "category": "pronoun / agreement",
        "confidence": "uncertain (prefix vs independent pronoun)",
        "source": NP_FIELD_MS + " a- plural noun prefix (aza 'people', aɓi 'children'). Independent high-frequency a also occurs in " + BIBLICA_GAW,
    },
    {
        "gbagyi": "ku",
        "english": "to / and (light preposition; also in the Hausa-origin phrase ku gode 'give thanks')",
        "category": "preposition / particle",
        "confidence": "uncertain",
        "source": BIBLICA_GAW + " Occurs in light-verb/prepositional collocations including ku gode (Hausa gode 'thank'). Not a confidently labelled native Gbagyi stem.",
    },
]


def stop_word_forms() -> List[str]:
    return [row["gbagyi"] for row in GBAGYI_STOP_WORDS]


# ---------------------------------------------------------------------------
# Zipf
# ---------------------------------------------------------------------------

def fit_zipf_law(token_list: Sequence[str] | str) -> Tuple[float, Dict[str, int]]:
    """
    Fit log(f) = C - s log(r) by ordinary least squares.

    Returns (s, frequency_dict). Punctuation-only tokens are excluded so that
    the rank-frequency law is estimated on word types.
    """
    if isinstance(token_list, str):
        tokens = token_list.split()
    else:
        tokens = list(token_list)
    words = [tok for tok in tokens if WORD_TOKEN_RE.fullmatch(tok)]
    frequency_dict = dict(Counter(words))
    if len(frequency_dict) < 2:
        return 0.0, frequency_dict

    ranked = sorted(frequency_dict.values(), reverse=True)
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    freqs = np.asarray(ranked, dtype=float)
    log_r = np.log(ranks)
    log_f = np.log(freqs)
    slope, intercept = np.polyfit(log_r, log_f, 1)
    s = float(-slope)
    return s, frequency_dict


def zipf_regression_details(frequency_dict: Dict[str, int]) -> Dict[str, float]:
    ranked = sorted(frequency_dict.values(), reverse=True)
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    freqs = np.asarray(ranked, dtype=float)
    log_r = np.log(ranks)
    log_f = np.log(freqs)
    slope, intercept = np.polyfit(log_r, log_f, 1)
    predicted = slope * log_r + intercept
    ss_res = float(np.sum((log_f - predicted) ** 2))
    ss_tot = float(np.sum((log_f - np.mean(log_f)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    return {
        "s": float(-slope),
        "C": float(intercept),
        "slope": float(slope),
        "r_squared": r2,
        "n_types": float(len(ranked)),
    }


def plot_zipf(
    frequency_dict: Dict[str, int],
    details: Dict[str, float],
    output_path: Optional[str | Path] = None,
):
    import matplotlib.pyplot as plt

    ranked = sorted(frequency_dict.values(), reverse=True)
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    freqs = np.asarray(ranked, dtype=float)
    log_r = np.log(ranks)
    fitted = np.exp(details["C"] - details["s"] * log_r)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.loglog(ranks, freqs, "o", markersize=3.2, alpha=0.55, color="#1f4e79", label="Observed")
    ax.loglog(ranks, fitted, "-", color="#c0392b", linewidth=2.0, label="OLS fit")
    ax.set_xlabel("Rank (r)")
    ax.set_ylabel("Frequency (f)")
    ax.set_title("Gbagyi rank–frequency distribution (Group 07)")
    equation = f"log(f) = {details['C']:.3f} − {details['s']:.3f} log(r)"
    ax.text(
        0.05,
        0.12,
        f"{equation}\ns = {details['s']:.4f}\nR² = {details['r_squared']:.4f}\nN = {int(details['n_types'])} types",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#bbbbbb"},
    )
    ax.legend(frameon=False)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=160)
    return fig


# ---------------------------------------------------------------------------
# Language models
# ---------------------------------------------------------------------------

class UnigramModel:
    """Maximum-likelihood unigram model with optional Add-1 smoothing."""

    def __init__(self) -> None:
        self.unigrams: Dict[str, int] = {}
        self.total_tokens = 0
        self.vocab_size = 0

    def fit(self, corpus_file_path: str) -> int:
        counts: Counter[str] = Counter()
        for line in Path(corpus_file_path).read_text(encoding="utf-8").splitlines():
            tokens = [tok for tok in line.strip().split(" ") if tok]
            counts.update(tokens)
        self.unigrams = dict(counts)
        self.total_tokens = sum(counts.values())
        self.vocab_size = len(counts)
        return self.total_tokens

    def get_probability(self, word: str, add_one: bool = False) -> float:
        count = self.unigrams.get(word, 0)
        if add_one:
            return (count + 1) / (self.total_tokens + self.vocab_size)
        if self.total_tokens == 0:
            return 0.0
        return count / self.total_tokens


class BigramModel:
    """
    Bigram language model with Laplace (Add-1) smoothing.

    P(w2 | w1) = (count(w1, w2) + 1) / (count(w1) + V)

    If w1 is unseen, count(w1) = 0, so P = 1 / V.
    """

    def __init__(self) -> None:
        self.unigrams: Dict[str, int] = defaultdict(int)
        self.bigrams: Dict[Tuple[str, str], int] = defaultdict(int)
        self.vocab_size = 0

    def fit(self, corpus_file_path: str) -> int:
        unigram_counts: Counter[str] = Counter()
        bigram_counts: Counter[Tuple[str, str]] = Counter()
        total_bigrams = 0
        with Path(corpus_file_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                tokens = [tok for tok in line.strip().split(" ") if tok]
                if not tokens:
                    continue
                unigram_counts.update(tokens)
                for left, right in zip(tokens, tokens[1:]):
                    bigram_counts[(left, right)] += 1
                    total_bigrams += 1
        self.unigrams = defaultdict(int, unigram_counts)
        self.bigrams = defaultdict(int, bigram_counts)
        self.vocab_size = len(unigram_counts)
        return total_bigrams

    def get_probability(self, w1: str, w2: str) -> float:
        if self.vocab_size <= 0:
            raise ValueError("Model has an empty vocabulary; call fit() first.")
        bigram_count = self.bigrams.get((w1, w2), 0)
        unigram_count = self.unigrams.get(w1, 0)
        return (bigram_count + 1) / (unigram_count + self.vocab_size)

    def compute_perplexity(self, test_file_path: str) -> float:
        """
        PP = exp( -(1/N) sum log P(w_i | w_{i-1}) )
        which equals 2 ** ( -(1/N) sum log2 P(w_i | w_{i-1}) ).
        N is the number of predicted bigrams (tokens after the first in each sentence).
        """
        log_prob_sum = 0.0
        n_bigrams = 0
        with Path(test_file_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                tokens = [tok for tok in line.strip().split(" ") if tok]
                if len(tokens) < 2:
                    continue
                for left, right in zip(tokens, tokens[1:]):
                    probability = self.get_probability(left, right)
                    if probability <= 0.0:
                        return float("inf")
                    log_prob_sum += math.log(probability)
                    n_bigrams += 1
        if n_bigrams == 0:
            return float("inf")
        return math.exp(-log_prob_sum / n_bigrams)

    def perplexity_report(self, test_file_path: str) -> Dict[str, Any]:
        lines = [line for line in Path(test_file_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        tokens = corpus_tokens(lines)
        pp = self.compute_perplexity(test_file_path)
        n_bigrams = sum(max(len(line.split()), 0) - 1 for line in lines if len(line.split()) >= 2)
        return {
            "test_sentences": len(lines),
            "test_tokens": len(tokens),
            "test_bigrams": n_bigrams,
            "vocab_size": self.vocab_size,
            "perplexity": pp,
        }


def independent_perplexity_check(model: BigramModel, test_file_path: str) -> float:
    """Recompute perplexity with log2 to confirm numerical equivalence."""
    log2_sum = 0.0
    n = 0
    for line in Path(test_file_path).read_text(encoding="utf-8").splitlines():
        tokens = [tok for tok in line.strip().split(" ") if tok]
        for left, right in zip(tokens, tokens[1:]):
            log2_sum += math.log2(model.get_probability(left, right))
            n += 1
    return 2 ** (-log2_sum / n) if n else float("inf")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_collection(url_list: Optional[Sequence[str]] = None) -> int:
    urls = list(url_list) if url_list is not None else default_gbagyi_urls()
    LOGGER.info("collecting from %s URLs", len(urls))
    return scrape_to_jsonl(urls, RAW_JSONL_PATH)


def run_full_pipeline(collect: bool = False) -> Dict[str, Any]:
    configure_logging()
    summary: Dict[str, Any] = {
        "date": datetime.now(timezone.utc).isoformat(),
        "collected_documents": None,
    }
    if collect or not RAW_JSONL_PATH.exists():
        summary["collected_documents"] = run_collection()
    summary["jsonl_errors"] = validate_jsonl(RAW_JSONL_PATH)
    summary["build"] = build_processed_corpus()
    summary["corpus_errors"] = validate_processed_corpus()
    stats = corpus_statistics()
    summary["stats"] = stats
    lines = load_corpus_lines()
    words = corpus_tokens(lines, words_only=True)
    s, freq = fit_zipf_law(words)
    details = zipf_regression_details(freq)
    summary["zipf"] = {"s": s, **details}
    unigram = UnigramModel()
    unigram.fit(str(PROCESSED_CORPUS_PATH))
    bigram = BigramModel()
    bigram_count = bigram.fit(str(PROCESSED_CORPUS_PATH))
    summary["bigram_count"] = bigram_count
    summary["unigram_vocab"] = unigram.vocab_size
    if TEST_PATH.exists():
        report = bigram.perplexity_report(str(TEST_PATH))
        report["perplexity_log2_check"] = independent_perplexity_check(bigram, str(TEST_PATH))
        summary["perplexity"] = report
    return summary


if __name__ == "__main__":
    configure_logging()
    # Recollect from the live web only when the JSONL snapshot is missing.
    result = run_full_pipeline(collect=not RAW_JSONL_PATH.exists())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
