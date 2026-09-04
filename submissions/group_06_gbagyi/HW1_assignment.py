from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

STOP_WORDS = {'n': 'and / of / in (context-dependent linker)', 'ɓa': 'they / them / people', 'wa': 'he / she / it (subject marker)', 'nu': 'is / was (copula or auxiliary)', 'yi': 'and / with (context-dependent linker)', 'ge': 'that / to / for (context-dependent complementizer)', 'mi': 'I / me', 'fye': 'you / your (context-dependent)', 'lo': 'in / at / on (locative)', 'ɓei': 'was / were (past auxiliary)', 'ɓo': 'not / no', 'a': 'and / then', 'to': 'not / do not', 'na': 'to / for / that (context-dependent)', 'ye': 'say / be / become (context-dependent verb)', 'nyi': 'this / here (deictic)', 'nya': 'there / that place (deictic)', 'kwo': 'all / every / whole', 'ga': 'with / at (context-dependent)', 'ya': 'go / come / do (context-dependent verb)', 'dna': 'then / and then', 'shi': 'then / when', 'ntu': 'because / for', 'ho': 'you (object or second-person form)', 'ku': 'to / for', 'ɓe': 'be / exist (auxiliary or copula)', 'm': 'my / me (bound form)', 'gye': 'see / know', 'da': 'father / parent (in kinship compounds)', 'ama': 'but', 'aba': 'father / at (context-dependent)', 'woi': 'who / which', 'ɓai': 'they / them (variant or context-dependent form)', 'mwa': 'also / again (context-dependent)', 'wu': 'you (second-person form)', 'tnu': 'place / time (context-dependent noun)', 'kwa': 'how / manner', 'fya': 'face / front / before (context-dependent)', 'ta': 'with / at (context-dependent)'}
TAG_RE = re.compile(r"<[^>]*>", re.DOTALL)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

def _normalize_spaces(text):
    text = unicodedata.normalize("NFC", text)
    text = TAG_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def _split_punctuation(text):
    tokens, current = [], []
    def flush():
        if current:
            tokens.append("".join(current))
            current.clear()
    for char in text:
        category = unicodedata.category(char)
        if char.isspace():
            flush()
        elif char.isalnum() or category.startswith("M"):
            current.append(char)
        elif char in "'’ʼ" and current:
            current.append(char)
        else:
            flush()
            tokens.append(char)
    flush()
    return tokens

def custom_tokenizer(text, remove_stop_words=False):
    tokens = [token.lower() for token in _split_punctuation(_normalize_spaces(text))]
    if remove_stop_words:
        tokens = [token for token in tokens if token not in STOP_WORDS]
    return " ".join(tokens)

def fit_zipf_law(token_list):
    import numpy as np
    frequencies = token_list if isinstance(token_list, Counter) else Counter(token_list)
    frequencies = Counter(dict(frequencies.most_common()))
    ranked = sorted(frequencies.values(), reverse=True)
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    slope, _ = np.polyfit(np.log(ranks), np.log(np.asarray(ranked, dtype=float)), 1)
    return float(-slope), frequencies

def zipf_fit_details(frequencies):
    import numpy as np
    ranked = sorted(frequencies.values(), reverse=True)
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    log_ranks = np.log(ranks)
    log_freqs = np.log(np.asarray(ranked, dtype=float))
    slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
    prediction = slope * log_ranks + intercept
    ss_res = float(np.sum((log_freqs - prediction) ** 2))
    ss_tot = float(np.sum((log_freqs - log_freqs.mean()) ** 2))
    return {
        "s": float(-slope),
        "C": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0,
        "ranks": ranks,
        "frequencies": np.asarray(ranked, dtype=float),
    }

class BigramModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.vocab_size = 0

    def fit(self, corpus_file_path):
        self.unigrams.clear()
        self.bigrams.clear()
        with Path(corpus_file_path).open(encoding="utf-8") as corpus:
            for line in corpus:
                tokens = line.strip().split()
                self.unigrams.update(tokens)
                self.bigrams.update(zip(tokens, tokens[1:]))
        self.vocab_size = len(self.unigrams)
        return sum(self.bigrams.values())

    def get_probability(self, w1, w2):
        if self.vocab_size == 0:
            raise ValueError("Fit the model before requesting probabilities")
        return (self.bigrams.get((w1, w2), 0) + 1) / (
            self.unigrams.get(w1, 0) + self.vocab_size
        )

    def compute_perplexity(self, test_file_path):
        total_log_probability, event_count = 0.0, 0
        with Path(test_file_path).open(encoding="utf-8") as test_file:
            for line in test_file:
                tokens = line.strip().split()
                for w1, w2 in zip(tokens, tokens[1:]):
                    total_log_probability += math.log(self.get_probability(w1, w2))
                    event_count += 1
        if event_count == 0:
            raise ValueError("Test file contains no bigram events")
        return math.exp(-total_log_probability / event_count)
