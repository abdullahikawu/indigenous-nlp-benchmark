from collections import Counter
import math

TRAIN_FILE = "data/gbagyi/processed/cleaned_corpus_group_08.txt"
TEST_FILE = "tests/test_gbagyi_unseen.txt"


def read_sentences(path):
    sentences = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line.split())

    return sentences


# Load training and test data.
train_sentences = read_sentences(TRAIN_FILE)
test_sentences = read_sentences(TEST_FILE)

# Flatten training/test tokens.
train_tokens = [
    word
    for sentence in train_sentences
    for word in sentence
]

test_tokens = [
    word
    for sentence in test_sentences
    for word in sentence
]

# ============================================================
# UNIGRAM MODEL
# ============================================================

unigram_counts = Counter(train_tokens)

N = len(train_tokens)
V = len(unigram_counts)


def unigram_probability(word):
    return (unigram_counts[word] + 1) / (N + V)


unigram_log_probability = 0.0

for word in test_tokens:
    probability = unigram_probability(word)
    unigram_log_probability += math.log(probability)

unigram_perplexity = math.exp(
    -unigram_log_probability / len(test_tokens)
)


# ============================================================
# BIGRAM MODEL
# ============================================================

bigram_counts = Counter()
context_counts = Counter()

for sentence in train_sentences:

    previous = "<s>"

    for word in sentence:
        bigram_counts[(previous, word)] += 1
        context_counts[previous] += 1

        previous = word


def bigram_probability(previous, word):
    return (
        bigram_counts[(previous, word)] + 1
    ) / (
        context_counts[previous] + V
    )


bigram_log_probability = 0.0
bigram_token_count = 0

for sentence in test_sentences:

    previous = "<s>"

    for word in sentence:

        probability = bigram_probability(previous, word)

        bigram_log_probability += math.log(probability)

        bigram_token_count += 1

        previous = word


bigram_perplexity = math.exp(
    -bigram_log_probability / bigram_token_count
)


# ============================================================
# RESULTS
# ============================================================

print("======================================")
print("GBAGYI N-GRAM LANGUAGE MODEL")
print("======================================")

print("Training sentences:", len(train_sentences))
print("Training tokens:", N)
print("Vocabulary size (V):", V)
print("Unigram perplexity:", unigram_perplexity)
print("Bigram perplexity:", bigram_perplexity)
print()

