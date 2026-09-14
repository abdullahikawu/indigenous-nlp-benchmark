import re
import math
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(int)
        self.vocab_size = 0
        self.total_tokens = 0

    def fit(self, corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        tokens = re.findall(r"\b\w+\b", text)
        tokens = ["<s>"] + tokens + ["</s>"]

        self.unigram_counts = Counter(tokens)
        self.total_tokens = len(tokens)
        self.vocab_size = len(self.unigram_counts)

        for i in range(len(tokens) - 1):
            self.bigram_counts[(tokens[i], tokens[i + 1])] += 1

        return self.bigram_counts

    def compute_perplexity(self, test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        tokens = re.findall(r"\b\w+\b", text)
        tokens = ["<s>"] + tokens + ["</s>"]

        log_prob_sum = 0
        N = len(tokens) - 1

        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            prob = (self.bigram_counts.get((w1, w2), 0) + 1) / (
                self.unigram_counts.get(w1, 0) + self.vocab_size
            )
            log_prob_sum += math.log2(prob)

        perplexity = 2 ** (-log_prob_sum / N)
        return perplexity


if __name__ == "__main__":
    model = BigramModel()
    model.fit("data/nup/processed/cleaned_corpus_group_02.txt")
    print("Vocab size:", model.vocab_size)