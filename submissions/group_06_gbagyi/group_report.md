# Group 06 — Gbagyi

## Indigenous Language AI Benchmark

### Group Members
- Name: MAGAJI AMINU IBRAHIM
  - Student ID: U22/FNS/CSC/1185
  - Institutional Email:

- Name: Abdulrauf Yusuf Kuta
  - Student ID: U22/FNS/CSC/1046
  - Institutional Email:

- Name: Abiodun Adejare Emmanuel
  - Student ID: U22/FEA/SED/1019
  - Institutional Email:

- Name: Ahmed Hassan
  - Student ID: U22/FNS/CSC/1088
  - Institutional Email:

- Name: Aliyu
  - Student ID: U22/FNS/CSC/1321
  - Institutional Email:

- Name: Baba nurudeen
  - Student ID: U22/fns/csc/1117
  - Institutional Email:

- Name: Balogu Alexander-Ugonna Ikezurumba
  - Student ID: U22/FNS/CSC/1042
  - Institutional Email:

- Name: Bashirat Ahmad
  - Student ID: U22/fea/sed/1342
  - Institutional Email:

- Name: Ekunnrin Adesina Joseph
  - Student ID: U22/FNS/CSC/1186
  - Institutional Email:

- Name: Fatima Binta Adamu
  - Student ID: U22/fea/sed/1117
  - Institutional Email:

- Name: HARUNA A. MUHAMMAD
  - Student ID: U22/FEA/SED/1424
  - Institutional Email:

- Name: Hauwa adamu safo
  - Student ID: U22/fea/sed/1402
  - Institutional Email:

- Name: Ibrahim Hussaini
  - Student ID: U22/FNS/CSC/1081
  - Institutional Email:

- Name: IBRAHIM ISMAIL
  - Student ID: U22/FNS/CSC/1260
  - Institutional Email:

- Name: Ibrahim Muhammad Jiya
  - Student ID: U22/FNS/CSC/1217
  - Institutional Email:

- Name: Isah shuaibu Evuti
  - Student ID: U20/fns/csc/1123
  - Institutional Email:

- Name: Jibrin Suleiman Gimba
  - Student ID: U22/FNS/CSC/1083
  - Institutional Email:

- Name: Joshua Emmanuel Shekwolo
  - Student ID: U22/FNS/CSC/1193
  - Institutional Email:

- Name: Matthias Yohanna
  - Student ID: U22/FNS/CSC/1243
  - Institutional Email:

- Name: Mohammed Nazif Mohammed
  - Student ID: U22/fns/csc/1329
  - Institutional Email:

- Name: Muhammad A Aliyu
  - Student ID: U22/FEA/SED/1299
  - Institutional Email:

- Name: Muhammad Abdullahi
  - Student ID: U22/FNS/CSC/1261
  - Institutional Email:

- Name: Muhammad Halima
  - Student ID: U22/FNS/CSC/1191
  - Institutional Email:

- Name: Muhammad Shehu
  - Student ID: U22/FNS/CSC/1340
  - Institutional Email:

- Name: Musa Idris Mohammed
  - Student ID: U22/fea/sed/1391
  - Institutional Email:

- Name: Mustapha Isah
  - Student ID: U22/FNS/CSC/1264
  - Institutional Email:

- Name: Mustapha Yusuf
  - Student ID: U22/fns/csc/1164
  - Institutional Email:

- Name: Rufai Faridah ajolayo
  - Student ID: U22/fea/sed/1010
  - Institutional Email:

- Name: SAIDU MOHAMMED
  - Student ID: U22/FNS/CSC/1086
  - Institutional Email:

- Name: Solomon Olamide Peter
  - Student ID: U22/FNS/CSC/1173
  - Institutional Email:

- Name: Suleiman Aliyu Saifudeen
  - Student ID: U22/fns/csc/1306
  - Institutional Email:

- Name: TAUHEED MUSA
  - Student ID: U22/FEA/SED/1417
  - Institutional Email:

- Name: Tukura Danladi Anselm
  - Student ID: U22/FNS/CSC/1132
  - Institutional Email:

- Name: USMAN KABIR
  - Student ID: U22/FNS/CSC/1084
  - Institutional Email:

- Name: Uwaisa muhammad
  - Student ID: U22/fea/sed/1128
  - Institutional Email:

- Name: Yahaya Saidu Ahmaf
  - Student ID: U22/FNS/CSC/1065
  - Institutional Email:

- Name: Yunusa Mohammed Ibrahim
  - Student ID: U22/FNS/CSC/1126
  - Institutional Email:

- Name: YUSUF AMINU ABDULLAHI
  - Student ID: U22/FNS/CSC/1029
  - Institutional Email:

- Name: Zubairu Usman Idris
  - Student ID: U22/FEA/SED/1071
  - Institutional Email:
### Language
Gbagyi (Gbagyi-Nkwa)

## 1. Data Collection
The Gbagyi corpus was collected from the Biblica® Open Gbagyi Contemporary Bible, New Testament 2025, provided through Open.Bible under a CC BY-SA license.

Data was retrieved programmatically using Python `requests`. The downloaded Bible artifact was extracted and the Gbagyi text was segmented into sentence units. The final raw corpus contains 10,647 sentence records, exceeding the required minimum of 2,500 sentences.

Each raw record was stored in JSONL format with an ID, source URL, UTC retrieval timestamp, and raw text. The raw corpus was saved as:

`data/gbagyi/raw/raw_data_group_06.jsonl`

Source:
`https://preview.open.bible/bibles/68de939641a2a80e0f049a7d`

The collection process was performed directly by the group using Python rather than using a pre-tokenized or third-party NLP dataset.

## 2. Text Normalization and Tokenization
The raw corpus was normalized using Python `re` and Unicode normalization (`unicodedata.normalize("NFC")`). HTML/XML tags and control characters were removed, whitespace was standardized, and text was lowercased while preserving Gbagyi-specific characters and diacritics.

Punctuation was detached from neighboring words using a custom regular-expression rule. Tokenization was performed using a custom whitespace-based tokenizer (`split()`), without NLTK or spaCy.

The raw corpus contained 10,647 sentence records and 246,991 total tokens before punctuation filtering. The cleaned corpus contains 10,647 non-empty sentences and was validated as UTF-8. No attached punctuation errors were found.

A curated list of 32 Gbagyi functional-word candidates was documented with English glosses. The cleaned corpus was exported to:

`data/gbagyi/processed/cleaned_corpus_group_06.txt`

The preprocessing preserved Gbagyi orthographic features, including characters such as `ə` and other diacritics/tone marks.

## 3. Zipf's Law Analysis
The Zipf analysis was performed on the cleaned Gbagyi corpus using word-frequency counts and rank ordering. Punctuation-only tokens were excluded from the analysis.

The cleaned corpus contained 153,579 word tokens and a vocabulary size of 5,831 unique word types.

The log-log relationship between frequency and rank was fitted using:

`log(f) = C - s log(r)`

The estimated Zipf exponent was:

`s = 1.458240`

with an intercept of approximately:

`C = 12.4201`

The frequency-rank plot showed the expected downward trend: a relatively small number of words occur very frequently, while many word types occur less frequently. The relatively high estimated exponent indicates a strong concentration of frequency among the most common word types.

Gbagyi orthographic features, including subdot characters, tone marks, and other diacritics, were preserved during preprocessing. These distinctions can increase the number of unique word types because differently marked forms may be treated as distinct tokens.

## 4. N-Gram Language Modeling
The unigram and bigram language models were implemented from scratch using Python dictionary-based frequency counts.

The bigram model included sentence boundary markers `<s>` and `</s>`. Laplace (Add-1) smoothing was applied to the bigram probabilities to handle unseen bigrams.

Training statistics:
- Training sentences: 10,647
- Vocabulary size: 5,862
- Smoothing method: Laplace (Add-1)

The bigram probability was calculated using the smoothed form:

`P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + 1) / (count(w_{i-1}) + V)`

where `V` is the vocabulary size.

## 5. Perplexity
The bigram model was evaluated on the instructor-provided unseen Gbagyi test set.

The test set contained 15 sentences, with 157 tokens evaluated. There were 77 occurrences of words not seen in the training vocabulary.

Using Laplace (Add-1) smoothing, the resulting bigram perplexity was:

**Bigram Perplexity: 2096.351318**

The relatively high perplexity reflects the difficulty of predicting unseen Gbagyi word sequences, particularly because the test set contains many word occurrences that were not present in the training corpus.

## 6. Conclusion
This project developed and evaluated a Gbagyi language corpus and a simple statistical language model using an empirical NLP workflow.

The group collected 10,647 sentence records from a licensed Gbagyi source, then applied Unicode-aware normalization and custom tokenization while preserving Gbagyi orthographic features and diacritics. The cleaned corpus contained 153,579 word tokens and 5,831 vocabulary types after punctuation filtering.

The Zipf analysis produced an estimated exponent of 1.458240, showing the expected frequency-rank relationship in the corpus. A unigram and Laplace-smoothed bigram model were implemented from scratch and evaluated on the unseen test set. The resulting bigram perplexity was 2096.351318.

The results demonstrate both the usefulness and challenges of building NLP resources for an indigenous language such as Gbagyi, especially where orthographic variation, diacritics, and limited training data affect vocabulary size and language-model prediction.
