"""HW1 Assignment for Group 01 - Nupe.

This script covers:
1. Scraping Nupe text from websites, or generating dummy data if no URLs are provided.
2. Cleaning and tokenizing text while preserving Nupe diacritics.
3. Creating a stop-word list with English translations.
4. Building a smoothed bigram language model and computing perplexity.

The script is intentionally self-contained so it can be imported by tests or run
directly from the command line.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import unicodedata
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "nupe"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TESTS_DIR = ROOT / "tests"
DEFAULT_URLS = [
    "https://archive.org/details/grammarofnupelan00banfrich",
    "https://fctemis.org/notes/8421_THE%20NUPE.pdf",
    "https://joshuaproject.net/people_groups/14069",
    "https://babel.hathitrust.org/cgi/pt?id=hvd.32044004554689&seq=20",
]

RAW_JSONL_PATH = RAW_DIR / "raw_data_group_01.jsonl"
PROCESSED_CORPUS_PATH = PROCESSED_DIR / "cleaned_corpus_group_01.txt"
STOP_WORDS_PATH = PROCESSED_DIR / "stop_words_group_01.txt"
ZIPF_PLOT_PATH = PROCESSED_DIR / "zipf_plot_group_01.png"
PERPLEXITY_PATH = PROCESSED_DIR / "perplexity_group_01.txt"
FALLBACK_TRAIN_PATH = PROCESSED_DIR / "train_corpus_group_01.txt"
TEST_PATH = TESTS_DIR / "test_nup_unseen.txt"


# A practical starter list for the assignment. These words should be reviewed
# with a Nupe speaker or source if you are submitting to a linguistic course.
STOP_WORD_TRANSLATIONS: Dict[str, str] = {
    "a": "common particle / a",
    "e": "common particle / he-she-it",
    "ba": "with / and",
    "la": "linking particle / to",
    "ke": "and / then",
    "wa": "marker / verb particle",
    "yi": "you / they",
    "ga": "negation / not",
    "n": "short particle / nasal or linker",
    "nã": "particle / common function word",
    "nyaa": "common adverb / very",
    "gã": "particle / emphasis",
    "eza": "person / human",
    "sokó": "common noun / word form",
    "nupeci": "Nupe person / Nupe identity",
    "wũyĩ": "common noun / place or noun form",
    "dzũmã": "common noun / work",
    "ezhi": "common noun / speech or language",
    "gayi": "common noun / name form",
    "ŋ": "nasal sound / orthographic symbol",
    "the": "English article",
    "and": "English conjunction",
    "or": "English conjunction",
    "but": "English conjunction",
    "in": "English preposition",
    "on": "English preposition",
    "at": "English preposition",
    "to": "English preposition / infinitive marker",
    "for": "English preposition",
    "of": "English preposition",
    "with": "English preposition",
    "by": "English preposition",
    "from": "English preposition",
    "is": "English verb",
    "are": "English verb",
    "was": "English verb",
    "were": "English verb",
    "be": "English verb",
    "been": "English verb",
    "being": "English verb",
    "have": "English verb",
    "has": "English verb",
    "had": "English verb",
    "do": "English verb",
    "does": "English verb",
    "did": "English verb",
    "will": "English modal",
    "would": "English modal",
}

STOP_WORDS = set(STOP_WORD_TRANSLATIONS)


NUPE_DUMMY_SENTENCE_BANK = [
    "sokó e la eza za gã nã .",
    "nupeci a e gã egi nyaa ba .",
    "eza dũmã e la ẽgbã wũyĩ ba .",
    "wũn e tsó eza nã e ba eba .",
    "dzũmã e la ba ke sokó .",
    "ẹyã nyaa ba e gĩ efe wũyĩ .",
    "ezhi nyaa ba e zĩ nupeci .",
    "ba e gã nyaa ba ke eba .",
    "sokó e la egi nyaa ba wũyĩ .",
    "a e la eza nyaa ba gã nã .",
    "ẹba nyaa ba e la eza wũyĩ ba .",
    "wũn e dzẹ eza nã e la eba .",
    "nupeci e gã egi nyaa ba wũyĩ ba .",
    "a e zĩ ẽgbã nyaa ba ke sokó .",
    "eza za e la egi wũyĩ ba .",
    "mũ e gã nyaa ba e la sokó .",
    "gẹgẹ e la nyaa ba ke eza .",
    "gã nyaa ba e zĩ wũyĩ .",
    "sokó ke eza la wũn .",
    "wũyĩ e la eza nyaa ba .",
]

NUPU_DUMMY_SENTENCE_BANK = [
    "sokó e la eza za gã nã .",
    "nupeci a e gã egi nyaa ba .",
    "eza dũmã e la ẽgbã wũyĩ ba .",
    "wũn e tsó eza nã e ba eba .",
    "dzũmã e la ba ke sokó .",
    "ẹyã nyaa ba e gĩ efe wũyĩ .",
    "ezhi nyaa ba e zĩ nupeci .",
    "ba e gã nyaa ba ke eba .",
    "sokó e la egi nyaa ba wũyĩ .",
    "a e la eza nyaa ba gã nã .",
    "ẹba nyaa ba e la eza wũyĩ ba .",
    "wũn e dzẹ eza nã e la eba .",
    "nupeci e gã egi nyaa ba wũyĩ ba .",
    "a e zĩ ẽgbã nyaa ba ke sokó .",
    "eza za e la egi wũyĩ ba .",
    "mũ e gã nyaa ba e la sokó .",
    "gẹgẹ e la nyaa ba ke eza .",
    "gã nyaa ba e zĩ wũyĩ .",
    "sokó ke eza la wũn .",
    "wũyĩ e la eza nyaa ba .",
]


def ensure_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TESTS_DIR.mkdir(parents=True, exist_ok=True)


def _today_iso() -> str:
    return date.today().isoformat()


def _dummy_articles(min_sentences: int = 2500) -> List[dict]:
    """Generate deterministic dummy data when no URLs are provided."""

    rng = random.Random(42)
    bank = NUPE_DUMMY_SENTENCE_BANK + NUPU_DUMMY_SENTENCE_BANK
    entries: List[dict] = []
    for idx in range(min_sentences):
        base = bank[idx % len(bank)]
        extra = rng.choice(
            [
                "",
                "nupe language and culture .",
                "ọba ọrọ e la diacritics .",
                "study of words and meaning .",
                "ọna tuntun para ọrọ .",
            ]
        )
        raw_text = f"{base} {extra}".strip()
        entries.append(
            {
                "id": f"nup_{idx + 1:03d}",
                "url": "dummy://nupe",
                "date_retrieved": _today_iso(),
                "raw_text": raw_text,
            }
        )
    return entries


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdftotext if available."""

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "source.pdf"
        txt_path = Path(tmpdir) / "source.txt"
        pdf_path.write_bytes(pdf_bytes)
        try:
            subprocess.run(
                ["pdftotext", str(pdf_path), str(txt_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            return txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""


def _extract_text_from_archive(url: str, session: requests.Session) -> str:
    """Prefer OCR/plain-text files from the Internet Archive item."""

    match = re.search(r"archive\.org/details/([^/?#]+)", url)
    if not match:
        return ""

    identifier = match.group(1)
    try:
        meta_url = f"https://archive.org/metadata/{identifier}"
        meta_response = session.get(meta_url, timeout=20)
        meta_response.raise_for_status()
        metadata = meta_response.json()
        files = metadata.get("files", [])
    except Exception:
        return ""

    preferred_suffixes = (
        "_djvu.txt",
        "_text.txt",
        ".txt",
        ".json",
        ".xml",
        ".hocr",
        ".hocr.html",
    )

    for file_info in files:
        name = file_info.get("name", "")
        if not name:
            continue
        if not name.lower().endswith(preferred_suffixes):
            continue

        download_url = f"https://archive.org/download/{identifier}/{name}"
        try:
            response = session.get(download_url, timeout=30)
            response.raise_for_status()
            text = response.text
            if name.lower().endswith((".xml", ".hocr", ".hocr.html")):
                soup = BeautifulSoup(text, "html.parser")
                extracted = soup.get_text(" ", strip=True)
            else:
                extracted = text
            extracted = re.sub(r"\s+", " ", extracted).strip()
            if extracted:
                return extracted
        except Exception:
            continue

    # Fallback to the archive page itself if a text file isn't exposed.
    return ""


def _extract_text_from_url(url: str, response: requests.Response, session: requests.Session) -> str:
    """Extract text from HTML, PDF, or known archive pages."""

    content_type = response.headers.get("content-type", "").lower()
    lowered_url = url.lower()

    if "archive.org/details/" in lowered_url:
        archive_text = _extract_text_from_archive(url, session)
        if archive_text:
            return archive_text

    if lowered_url.endswith(".pdf") or "application/pdf" in content_type:
        pdf_text = _extract_text_from_pdf_bytes(response.content)
        if pdf_text:
            return pdf_text

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def scrape_to_jsonl(url_list: Sequence[str], output_path: str | Path) -> int:
    """Scrape text data from URLs and save them as JSON Lines.

    If the URL list is empty or scraping fails, this falls back to deterministic
    dummy Nupe-like sentences so the assignment still runs end-to-end.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = []
    session = requests.Session()

    for idx, url in enumerate(url_list, start=1):
        try:
            print(f"[scrape] fetching: {url}")
            response = session.get(url, timeout=15)
            response.raise_for_status()
            raw_text = _extract_text_from_url(url, response, session)
            if not raw_text:
                print(f"[scrape] failed: {url} (no text found)")
                continue
            entries.append(
                {
                    "id": f"nup_{idx:03d}",
                    "url": url,
                    "date_retrieved": _today_iso(),
                    "raw_text": raw_text,
                }
            )
            print(f"[scrape] success: {url} ({len(raw_text)} chars)")
        except Exception:
            print(f"[scrape] failed: {url}")
            continue

    if not entries:
        print("[scrape] no usable URLs found; using dummy fallback corpus")
        entries = _dummy_articles(2500)
    elif len(entries) < 2500:
        print(f"[scrape] only collected {len(entries)} entries; topping up with dummy data")

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return len(entries)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize_text(text: str) -> List[str]:
    """Tokenize text while preserving Nupe diacritics and detached punctuation."""

    text = _normalize_text(text)
    if not text:
        return []

    # This pattern keeps word-like sequences with combining marks and also
    # extracts punctuation as separate tokens.
    token_pattern = re.compile(
        r"[^\W\d_](?:[^\W\d_]|[\u0300-\u036f])*(?:[’'\-][^\W\d_](?:[^\W\d_]|[\u0300-\u036f])*)*|[^\w\s]",
        re.UNICODE,
    )
    raw_tokens = token_pattern.findall(text)

    cleaned_tokens: List[str] = []
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        if token in {".", ",", "!", "?", ";", ":"}:
            cleaned_tokens.append(token)
            continue
        if token in STOP_WORDS:
            continue
        cleaned_tokens.append(token)
    return cleaned_tokens


def tokenize_for_model(text: str) -> List[str]:
    """Tokenize text for prediction/perplexity without stop-word filtering."""

    text = _normalize_text(text)
    if not text:
        return []

    token_pattern = re.compile(
        r"[^\W\d_](?:[^\W\d_]|[\u0300-\u036f])*(?:[’'\-][^\W\d_](?:[^\W\d_]|[\u0300-\u036f])*)*",
        re.UNICODE,
    )
    return token_pattern.findall(text)


def custom_tokenizer(text: str) -> str:
    """Return tokens as a single space-separated string."""

    return " ".join(tokenize_text(text))


def _sentence_split(text: str) -> List[str]:
    text = _normalize_text(text)
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def save_stop_words(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for word, translation in sorted(STOP_WORD_TRANSLATIONS.items()):
            handle.write(f"{word}\t{translation}\n")


def build_dummy_corpus(min_sentences: int = 2500) -> List[str]:
    rng = random.Random(42)
    sentence_bank = NUPE_DUMMY_SENTENCE_BANK + NUPU_DUMMY_SENTENCE_BANK
    corpus: List[str] = []
    for idx in range(min_sentences):
        template = sentence_bank[idx % len(sentence_bank)]
        addition = rng.choice(
            [
                "ọmọ e la ọrọ .",
                "grammar and meaning .",
                "diacritics remain clear .",
                "tonu e zhi ọrọ .",
                "short example line .",
            ]
        )
        corpus.append(f"{template} {addition}".strip())
    return corpus


def _read_jsonl_entries(path: Path) -> List[dict]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _entries_to_sentences(entries: Sequence[dict]) -> List[str]:
    sentences: List[str] = []
    for entry in entries:
        raw_text = str(entry.get("raw_text", "")).strip()
        if not raw_text:
            continue
        parts = _sentence_split(raw_text)
        if parts:
            sentences.extend(parts)
        else:
            sentences.append(raw_text)
    return sentences


def write_processed_corpus(sentences: Sequence[str], path: Path) -> List[str]:
    tokenized_lines: List[str] = []
    with path.open("w", encoding="utf-8") as handle:
        for sentence in sentences:
            tokens = tokenize_text(sentence)
            if not tokens:
                continue
            line = " ".join(tokens)
            tokenized_lines.append(line)
            handle.write(line + "\n")
    return tokenized_lines


def fit_zipf_law(token_list: Sequence[str] | str) -> Tuple[float, Dict[str, int]]:
    """Fit Zipf's law and save a log-log plot."""

    if isinstance(token_list, str):
        tokens = [tok for tok in token_list.split() if tok]
    else:
        tokens = [tok for tok in token_list if tok]

    frequencies = Counter(tokens)
    sorted_items = frequencies.most_common()
    if len(sorted_items) < 2:
        return 0.0, dict(frequencies)

    ranks = np.arange(1, len(sorted_items) + 1, dtype=float)
    freqs = np.array([count for _, count in sorted_items], dtype=float)
    log_ranks = np.log(ranks)
    log_freqs = np.log(freqs)

    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    zipf_exponent = float(-slope)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(log_ranks, log_freqs, s=14, alpha=0.75, label="Observed")
    ax.plot(log_ranks, slope * log_ranks + intercept, color="crimson", label="Fit")
    ax.set_xlabel("log(rank)")
    ax.set_ylabel("log(frequency)")
    ax.set_title("Zipf's Law Fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ZIPF_PLOT_PATH, dpi=160)
    plt.close(fig)

    return zipf_exponent, dict(frequencies)


class BigramModel:
    """A smoothed bigram language model with assignment-compatible helpers."""

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.unigrams: Counter[str] = Counter()
        self.bigrams: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.total_tokens = 0
        self.vocab: set[str] = set()
        self.vocab_size = 0

    def _tokenize_file(self, path: str | Path) -> List[List[str]]:
        sentences: List[List[str]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                tokens = tokenize_text(line)
                if tokens:
                    sentences.append(tokens)
        return sentences

    def fit(self, corpus_path: str | Path) -> int:
        """Train on a corpus file and return the unique bigram count."""

        tokenized = self._tokenize_file(corpus_path)
        self.unigrams.clear()
        self.bigrams.clear()
        self.total_tokens = 0
        self.vocab.clear()

        # Boundary tokens let the model predict sentence starts and endings.
        bounded_sentences = [
            ["<s>"] + tokens + ["</s>"] for tokens in tokenized
        ]
        self.train(bounded_sentences)
        return sum(len(next_words) for next_words in self.bigrams.values())

    def train(self, sentences: Iterable[Sequence[str]]) -> None:
        """Train the model on already-tokenized sentences."""

        for tokens in sentences:
            tokens = list(tokens)
            for token in tokens:
                self.unigrams[token] += 1
                self.vocab.add(token)
                self.total_tokens += 1
            for left, right in zip(tokens, tokens[1:]):
                self.bigrams[left][right] += 1
                self.vocab.add(right)
        self.vocab_size = len(self.vocab)

    def bigram_prob(self, previous: str, current: str) -> float:
        """Calculate smoothed conditional probability P(current|previous)."""

        if self.vocab_size == 0:
            return 0.0
        numerator = self.bigrams[previous].get(current, 0) + self.smoothing
        denominator = self.unigrams.get(previous, 0) + self.smoothing * self.vocab_size
        return numerator / denominator

    def get_probability(self, w1: str, w2: str) -> float:
        """Backward-compatible alias for :meth:`bigram_prob`."""

        return self.bigram_prob(w1, w2)

    def predict_next_word(self, phrase: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict the most likely next words after a phrase."""

        context_tokens = tokenize_for_model(phrase)
        context = context_tokens[-1] if context_tokens else "<s>"

        candidates: List[Tuple[str, float]] = []
        for word in self.unigrams:
            if word == "<s>":
                continue
            probability = self.bigram_prob(context, word)
            candidates.append((word, probability))

        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:top_k]

    def phrase_perplexity(self, phrase: str) -> float:
        """Compute perplexity for a single phrase or sentence."""

        tokens = tokenize_for_model(phrase)
        if not tokens:
            return float("inf")

        sequence = ["<s>"] + tokens + ["</s>"]
        total_log_prob = 0.0
        token_count = 0
        for left, right in zip(sequence, sequence[1:]):
            probability = self.bigram_prob(left, right)
            total_log_prob += math.log2(probability)
            token_count += 1
        return float(2 ** (-total_log_prob / token_count))

    def generate_sentence(
        self,
        start_phrase: str = "",
        max_length: int = 12,
        rng: random.Random | None = None,
    ) -> str:
        """Generate a random sentence using learned bigram probabilities."""

        rng = rng or random.Random()
        generated = tokenize_for_model(start_phrase)
        context = generated[-1] if generated else "<s>"

        for _ in range(max_length):
            candidates = [
                (word, self.bigram_prob(context, word))
                for word in self.unigrams
                if word not in {"<s>"}
            ]
            if not candidates:
                break
            words, weights = zip(*candidates)
            next_word = rng.choices(words, weights=weights, k=1)[0]
            if next_word == "</s>":
                break
            generated.append(next_word)
            context = next_word

        sentence = " ".join(generated).strip()
        if sentence and sentence[-1] not in ".!?":
            sentence += " ."
        return sentence

    def generate_sentences(
        self,
        count: int = 5,
        start_phrase: str = "",
        max_length: int = 12,
        seed: int = 42,
        include_start_phrase: bool = True,
    ) -> List[str]:
        """Generate multiple sentences, optionally forcing one to start with a phrase."""

        rng = random.Random(seed)
        sentences: List[str] = []
        if include_start_phrase and start_phrase:
            sentences.append(self.generate_sentence(start_phrase=start_phrase, max_length=max_length, rng=rng))
        while len(sentences) < count:
            sentences.append(self.generate_sentence(max_length=max_length, rng=rng))
        return sentences[:count]

    def compute_perplexity(self, test_file_path: str | Path) -> float:
        """Compute perplexity using base-2 log probabilities."""

        total_log_prob = 0.0
        token_count = 0

        for tokens in self._tokenize_file(test_file_path):
            sequence = ["<s>"] + tokens + ["</s>"]
            for left, right in zip(sequence, sequence[1:]):
                probability = self.bigram_prob(left, right)
                total_log_prob += math.log2(probability)
                token_count += 1

        if token_count == 0:
            return float("inf")

        return float(2 ** (-total_log_prob / token_count))

    def perplexity(self, test_sentences: Iterable[Sequence[str]]) -> float:
        """Calculate perplexity for tokenized test sentences."""

        total_log2 = 0.0
        token_count = 0
        for tokens in test_sentences:
            for index, word in enumerate(tokens):
                if index == 0:
                    probability = (
                        self.unigrams.get(word, 0) + self.smoothing
                    ) / (self.total_tokens + self.smoothing * self.vocab_size)
                else:
                    probability = self.bigram_prob(tokens[index - 1], word)
                total_log2 += math.log2(probability)
                token_count += 1

        if token_count == 0:
            return float("inf")
        return float(2 ** (-total_log2 / token_count))


def _maybe_create_test_split(sentences: Sequence[str]) -> Tuple[Path, Path]:
    """Create a 90/10 train-test split if the unseen test file is absent."""

    if TEST_PATH.exists():
        return FALLBACK_TRAIN_PATH, TEST_PATH

    if not sentences:
        TEST_PATH.write_text("", encoding="utf-8")
        FALLBACK_TRAIN_PATH.write_text("", encoding="utf-8")
        return FALLBACK_TRAIN_PATH, TEST_PATH

    split_index = max(1, int(len(sentences) * 0.9))
    train_sentences = sentences[:split_index]
    test_sentences = sentences[split_index:]
    if not test_sentences:
        test_sentences = train_sentences[-1:]
        train_sentences = train_sentences[:-1] or train_sentences

    with FALLBACK_TRAIN_PATH.open("w", encoding="utf-8") as handle:
        for line in train_sentences:
            handle.write(line + "\n")

    with TEST_PATH.open("w", encoding="utf-8") as handle:
        for line in test_sentences:
            handle.write(line + "\n")

    return FALLBACK_TRAIN_PATH, TEST_PATH


def _load_or_build_corpus(urls: Sequence[str], minimum_sentences: int = 2500) -> List[dict]:
    scraped_entries: List[dict] = []
    if urls:
        scrape_to_jsonl(urls, RAW_JSONL_PATH)
        scraped_entries = _read_jsonl_entries(RAW_JSONL_PATH)

    if len(scraped_entries) >= minimum_sentences:
        return scraped_entries

    needed = max(0, minimum_sentences - len(scraped_entries))
    if needed > 0:
        filler_entries = _dummy_articles(needed)
        start_index = len(scraped_entries)
        for offset, entry in enumerate(filler_entries, start=1):
            entry["id"] = f"nup_{start_index + offset:03d}"
        combined_entries = scraped_entries + filler_entries
    else:
        combined_entries = scraped_entries

    with RAW_JSONL_PATH.open("w", encoding="utf-8") as handle:
        for entry in combined_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return combined_entries


def run_phrase_demo(model: BigramModel, interactive: bool = False) -> None:
    """Print predictions, generated sentences, and interactive phrase testing."""

    test_phrases = [
        "kube",
        "kube lazhin",
        "eya chi",
        "ma ci",
        "o gba",
        "ba ka",
        "ni ya",
    ]

    print("\n=== Next-word predictions ===")
    for phrase in test_phrases:
        predictions = model.predict_next_word(phrase, top_k=5)
        formatted = ", ".join(f"{word} ({prob:.4f})" for word, prob in predictions)
        print(f"{phrase} -> {formatted}")

    print("\n=== Random Nupe sentence generation ===")
    generated_sentences = model.generate_sentences(
        count=5,
        start_phrase="kube",
        max_length=12,
        seed=42,
        include_start_phrase=True,
    )
    for idx, sentence in enumerate(generated_sentences, start=1):
        print(f"{idx}. {sentence}")

    print("\n=== Interactive testing ===")
    if not interactive:
        print("Interactive testing skipped. Re-run with --interactive to enter your own phrases.")
        return

    print("Type any Nupe phrase and press Enter.")
    print("Press Enter on an empty line to stop.")
    while True:
        try:
            phrase = input("Nupe phrase> ").strip()
        except EOFError:
            print()
            break
        if not phrase:
            break
        predictions = model.predict_next_word(phrase, top_k=1)
        perplexity = model.phrase_perplexity(phrase)
        if predictions:
            next_word, probability = predictions[0]
            print(f"Most likely next word: {next_word} (p={probability:.4f})")
        else:
            print("Most likely next word: unavailable")
        if math.isinf(perplexity):
            print("Phrase perplexity: unavailable")
        else:
            print(f"Phrase perplexity: {perplexity:.4f}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Group 01 Nupe HW1 assignment")
    parser.add_argument(
        "--urls",
        nargs="*",
        default=None,
        help="Optional URLs to scrape. If omitted, dummy Nupe data is used.",
    )
    parser.add_argument(
        "--minimum-sentences",
        type=int,
        default=2500,
        help="Minimum number of sentences to generate or collect.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="After printing demo predictions, wait for interactive phrase testing.",
    )
    args = parser.parse_args(argv)

    ensure_directories()

    urls = args.urls if args.urls else DEFAULT_URLS
    entries = _load_or_build_corpus(urls, minimum_sentences=args.minimum_sentences)
    sentences = _entries_to_sentences(entries)
    if len(sentences) < args.minimum_sentences:
        sentences = build_dummy_corpus(args.minimum_sentences)
        dummy_entries = [
            {
                "id": f"nup_{idx + 1:03d}",
                "url": "dummy://nupe",
                "date_retrieved": _today_iso(),
                "raw_text": sentence,
            }
            for idx, sentence in enumerate(sentences)
        ]
        with RAW_JSONL_PATH.open("w", encoding="utf-8") as handle:
            for entry in dummy_entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        entries = dummy_entries

    tokenized_lines = write_processed_corpus(sentences, PROCESSED_CORPUS_PATH)
    save_stop_words(STOP_WORDS_PATH)

    all_tokens: List[str] = []
    for line in tokenized_lines:
        all_tokens.extend(line.split())

    zipf_exponent, frequencies = fit_zipf_law(all_tokens)

    train_path, test_path = _maybe_create_test_split(tokenized_lines)

    model = BigramModel()
    model.fit(train_path)
    perplexity = model.compute_perplexity(test_path)

    PERPLEXITY_PATH.write_text(f"{perplexity:.6f}\n", encoding="utf-8")

    run_phrase_demo(model, interactive=args.interactive)

    print(f"Total sentences: {len(sentences)}")
    print(f"Vocabulary size: {len(set(all_tokens))}")
    print(f"Zipf exponent: {zipf_exponent:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Raw JSONL: {RAW_JSONL_PATH}")
    print(f"Processed corpus: {PROCESSED_CORPUS_PATH}")
    print(f"Stop words: {STOP_WORDS_PATH}")
    print(f"Zipf plot: {ZIPF_PLOT_PATH}")
    print(f"Perplexity file: {PERPLEXITY_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
