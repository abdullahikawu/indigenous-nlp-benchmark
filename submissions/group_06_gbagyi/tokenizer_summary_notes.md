# Tokenizer & Normalization Summary
- Processed raw Gbagyi text using custom diacritic-aware tokenization.
- Cleaned whitespace anomalies while safeguarding Gbagyi subdots and nasal vowels.
- Verified output tokens for downstream Unigram/Bigram language model ingestion.

## Part 2: Text Normalization & Tokenization (Completed by @emmanueljoshuashekwolo25-dev)
- **Text Normalization:** Implemented lowercased string sanitization and whitespace normalization for raw Gbagyi text.
- **Diacritic-Aware Tokenizer:** Developed a custom tokenizer preserving Gbagyi orthography and tone/subdot accents (`ɓ`, `ɗ`, `ã`, `ẽ`, `ĩ`, `õ`, `ũ`, `ɛ`, `ɔ`).
- **Vocabulary Extraction:** Tokenized processed corpus splits and generated clean word-token sequences for N-Gram modeling.
- **Status:** PASSED — Normalization pipeline and tokenizer validation complete.