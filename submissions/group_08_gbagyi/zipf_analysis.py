from collections import Counter
import math

INPUT_FILE = "data/gbagyi/processed/cleaned_corpus_group_08.txt"
OUTPUT_FILE = "submissions/group_08_gbagyi/zipf_rank_frequency.png"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    tokens = f.read().split()

# Keep only word tokens and exclude punctuation.
words = [token for token in tokens if token.isalpha()]

# Count word frequencies.
frequency = Counter(words)

# Sort by frequency from highest to lowest.
rank_frequency = frequency.most_common()

ranks = []
frequencies = []

for rank, (word, count) in enumerate(rank_frequency, start=1):
    ranks.append(rank)
    frequencies.append(count)

# Calculate log10 values.
log_ranks = [math.log10(rank) for rank in ranks]
log_frequencies = [math.log10(count) for count in frequencies]

# Linear regression: log(f) = C - s log(r)
n = len(log_ranks)

mean_x = sum(log_ranks) / n
mean_y = sum(log_frequencies) / n

numerator = sum(
    (x - mean_x) * (y - mean_y)
    for x, y in zip(log_ranks, log_frequencies)
)

denominator = sum(
    (x - mean_x) ** 2
    for x in log_ranks
)

slope = numerator / denominator
intercept = mean_y - slope * mean_x

s = -slope

# Plot Zipf's Law.
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(log_ranks, log_frequencies, s=10)
plt.xlabel("log10(Rank)")
plt.ylabel("log10(Frequency)")
plt.title("Zipf's Law for the Gbagyi Corpus")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300)
plt.close()

print("======================================")
print("GBAGYI ZIPF'S LAW ANALYSIS")
print("======================================")
print("Total word tokens:", len(words))
print("Unique vocabulary:", len(frequency))
print("Estimated slope:", slope)
print("Estimated Zipf exponent (s):", s)
print("Intercept (C):", intercept)
print("Plot saved to:", OUTPUT_FILE)
