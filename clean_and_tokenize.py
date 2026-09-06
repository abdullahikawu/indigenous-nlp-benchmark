import json
import re
import unicodedata

INPUT_FILE = "data/gbagyi/raw/raw_data_group_08.jsonl"
OUTPUT_FILE = "data/gbagyi/processed/cleaned_corpus_group_08.txt"


def normalize_text(text):
    """
    Normalize Unicode without removing Gbagyi diacritics.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    return text


def tokenize(text):
    """
    Rule-based tokenizer.
    Keeps Gbagyi letters and Unicode diacritics together.
    Separates punctuation from words.
    """
    text = normalize_text(text)

    # Separate punctuation from words.
    text = re.sub(r'([.,!?;:"“”‘’()\[\]{}])', r' \1 ', text)

    # Normalize whitespace.
    text = re.sub(r'\s+', ' ', text).strip()

    return text.split()


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        sentence_count = 0

        for line in infile:
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            raw_text = record.get("raw_text", "").strip()

            if not raw_text:
                continue

            tokens = tokenize(raw_text)

            if tokens:
                outfile.write(" ".join(tokens) + "\n")
                sentence_count += 1

    print("Cleaning and tokenization complete.")
    print("Output file:", OUTPUT_FILE)
    print("Sentence/line count:", sentence_count)


if __name__ == "__main__":
    main()
