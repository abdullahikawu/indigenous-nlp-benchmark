# Group Report: Gbagyi Language NLP Analysis

**Course**: CSC 406 - Artificial Intelligence  
**Group**: Group 06  
**Language**: Gbagyi (Gbagyi-Nkwa)  
**Corpus retrieval date**: 2026-09-04  
**Status**: Analysis complete; collaboration and lecturer submission gates remain pending

## Executive summary

This submission collects 7,958 Gbagyi verse-level sentences from 260 public STEP Bible chapter pages, normalizes them with a Unicode-aware rule-based tokenizer, fits a Zipf rank-frequency model, and evaluates a from-scratch Add-1 bigram model on the instructor-provided unseen test file. The committed corpus and code are reproducible from the notebook, but the group must still preserve the real multi-author commit history and complete the Google Form.

## 1. Data collection

- **Raw sentence records**: 7,958
- **Source chapter pages**: 260
- **Date retrieved**: 2026-09-04
- **Average raw sentence length**: 114.18 characters
- **Failed scrapes during collection**: 0
- **JSONL schema**: validated locally; every record has integer `id`, string `url`, string `date_retrieved`, and string `raw_text`.

The scraper uses `requests` and BeautifulSoup, discovers chapter links from the version index, and extracts only `lang="gbr"` verse spans. It does not use a pre-tokenized Hugging Face, Kaggle, or published research dataset. The STEP Bible page identifies the text as “Gbagyi New Testament (GAW) / Alkawali Woiwoyi Biblica 1997 NT”; the group should retain this attribution and observe the source's terms for any public redistribution.

### Source URLs

- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.14
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.15
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.16
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Cor.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Jo.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Jo.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Jo.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Jo.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Jo.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Pe.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Pe.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Pe.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Pe.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Pe.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Th.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Th.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Th.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Th.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Th.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Ti.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Ti.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Ti.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Ti.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Ti.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=1Ti.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Cor.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Jo
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Pe.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Pe.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Pe.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Th.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Th.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Th.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Ti.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Ti.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Ti.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=2Ti.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=3Jo
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.14
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.15
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.16
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.17
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.18
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.19
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.20
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.21
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.22
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.23
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.24
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.25
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.26
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.27
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.28
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=Act.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=Col.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Col.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Col.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Col.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Eph.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Eph.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Eph.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Eph.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Eph.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Eph.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Gal.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Gal.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Gal.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Gal.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Gal.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Gal.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=Heb.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=Jam.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Jam.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Jam.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Jam.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Jam.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.14
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.15
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.16
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.17
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.18
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.19
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.20
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.21
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=Joh.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=Jude
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.14
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.15
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.16
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.17
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.18
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.19
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.20
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.21
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.22
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.23
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.24
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=Luk.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.14
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.15
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.16
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mar.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.14
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.15
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.16
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.17
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.18
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.19
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.20
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.21
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.22
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.23
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.24
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.25
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.26
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.27
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.28
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=Mat.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=Phile
- https://www.stepbible.org/?q=version=GbrGAW@reference=Phili.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Phili.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Phili.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Phili.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.14
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.15
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.16
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.17
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.18
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.19
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.20
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.21
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.22
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rev.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.10
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.11
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.12
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.13
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.14
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.15
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.16
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.3
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.4
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.5
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.6
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.7
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.8
- https://www.stepbible.org/?q=version=GbrGAW@reference=Rom.9
- https://www.stepbible.org/?q=version=GbrGAW@reference=Tit.1
- https://www.stepbible.org/?q=version=GbrGAW@reference=Tit.2
- https://www.stepbible.org/?q=version=GbrGAW@reference=Tit.3

## 2. Normalization and tokenization

- **Processed sentences**: 7,958
- **Total tokens**: 246,570
- **Unique vocabulary**: 7,105
- **Type-token ratio**: 0.028815
- **Stop-word occurrences measured**: 96,250
- **Diacritic/subdot-sensitive token count**: 22,046

The tokenizer applies NFC normalization, removes HTML/XML markup and control characters with regular expressions, normalizes whitespace, lowercases, detaches punctuation, and preserves Unicode letters and combining marks. The training corpus retains function words because removing them would discard important grammatical transitions needed by the language model. A `remove_stop_words=True` option is implemented and a 40-entry Gbagyi functional glossary with English glosses is included in the notebook; the glosses should be checked by a fluent speaker before publication.

**Before**

```text
<p>Gbagyi ẹtí, ọkọ́!</p>
```

**After**

```text
gbagyi ẹtí , ọkọ́ !
```

The exact token output for combining marks depends on the source's Unicode representation; NFC normalization is applied before tokenization so canonically equivalent sequences are handled consistently.

## 3. Zipf analysis

| Rank | Token | Frequency | Log(rank) | Log(frequency) |
|---:|---|---:|---:|---:|
| 1 | `,` | 15050 | 0.00000 | 9.61913 |
| 2 | `n` | 14613 | 0.69315 | 9.58967 |
| 3 | `.` | 10515 | 1.09861 | 9.26056 |
| 4 | `ɓa` | 7631 | 1.38629 | 8.93997 |
| 5 | `-` | 5798 | 1.60944 | 8.66527 |
| 6 | `zhin` | 5649 | 1.79176 | 8.63923 |
| 7 | `wo` | 5577 | 1.94591 | 8.62641 |
| 8 | `yi` | 5497 | 2.07944 | 8.61196 |
| 9 | `wa` | 5108 | 2.19722 | 8.53856 |
| 10 | `nu` | 3912 | 2.30259 | 8.27180 |

- **Zipf exponent `s`**: 1.444133
- **Intercept `C`**: 12.381894
- **Log-log R²**: 0.976450
- **Plot**: `submissions/group_06_gbagyi/zipf_plot.png`

The fitted negative slope indicates how quickly frequency decreases by rank. Gbagyi function words dominate the highest ranks, as expected for a grammatical text. Diacritics, subdots, and tone marks can increase the number of distinct surface forms, which expands the vocabulary and spreads frequency mass across related spellings. NFC normalization preserves linguistic distinctions while preventing accidental duplication caused only by alternate Unicode encodings.

## 4. Bigram language model

- **Training tokens**: 246,570
- **Vocabulary size `V`**: 7,105
- **Bigram types**: 59,226
- **Bigram tokens**: 238,612
- **Smoothing**: Add-1/Laplace
- **Unseen test-set perplexity**: 795.998833

| Bigram | Count | P(w2\|w1) |
|---|---:|---:|
| (n, na) | 1310 | 0.06036467 |
| (ge, ,) | 1272 | 0.12177157 |
| (,, “) | 1263 | 0.05705258 |
| (,, n) | 1238 | 0.05592417 |
| (n, wo) | 1057 | 0.04871535 |

The model handles unseen transitions because the numerator receives one count, while the denominator includes `V`. Perplexity is calculated directly from all adjacent token pairs in `tests/test_gbagyi_unseen.txt`; it is not hard-coded. Because the training data is a translated religious text and the test data is short, the score should be interpreted as a reproducibility metric rather than a broad estimate of conversational Gbagyi fluency.

## 5. Integrity and collaboration checklist

- [x] At least 2,500 raw sentences.
- [x] Valid UTF-8 JSONL with required fields.
- [x] One processed sentence per line with single-space token separation.
- [x] Unicode-aware normalization and punctuation separation.
- [x] At least 30 functional stop words with English glosses.
- [x] Zipf plot and fitted exponent.
- [x] From-scratch Add-1 bigram model.
- [x] Perplexity evaluated from the unseen test file.
- [ ] Every real student has at least three meaningful commits under their own account.
- [ ] Real member names, student IDs, and emails confirmed in the report.
- [ ] All member PRs reviewed without squashing.
- [ ] GitHub Actions/autograder results reviewed.
- [ ] Final PR opened from `zahfaraway:group-06-gbagyi` to `abdullahikawu:main`.
- [ ] Google Form completed by the team lead.

The remaining unchecked items are deliberately not fabricated. The team lead should merge individual member contributions transparently, preserve authorship, and only then submit the final PR.

## 6. Lecturer-directed contribution map

The following assignment map records each member's part as requested by the lecturer and is also used in the GitHub notification. It is an acknowledgement/coordination record, not a rewrite of Git authorship. Each member must review the relevant work and make their own substantive commits before the lead can mark that part as completed in the final submission.

- **Part 1 — Data collection and provenance:** @mustyandcool, @ALMUSTY0, and @alameenmag
- **Part 2 — Normalization, tokenization, stop words, and orthography:** @emmanueljoshuashekwolo25-dev and @gimba45
- **Part 3 — Zipf analysis and visualization:** @Hassanahmed331
- **Part 4 — N-gram model, perplexity, and report:** @isahabdulkadir and @Shizzey
- **Validation and quality checks:** @anselmdan, @meethalima913-bot, and @ibnbazz
- **Additional group members to confirm and assign:** @musty178 and @talk2kabirusman-wq
- **Integration and final review:** @zahfaraway (Yusuf Aminu Abdullahi, team lead)

## Appendix A — References

1. STEP Bible, *Gbagyi New Testament (GAW) / Alkawali Woiwoyi Biblica 1997 NT*, version index: https://www.stepbible.org/version.jsp?version=GbrGAW
2. Jurafsky, D. & Martin, J. H. (2020). *Speech and Language Processing* (3rd ed.). https://web.stanford.edu/~jurafsky/slp3/
3. Zipf, G. K. (1949). *Human Behavior and the Principle of Least Effort*.

## Appendix B — Contributors

The supplied team plan names the following GitHub handles and roles. Replace or augment this list with the real names, IDs, emails, and commit-backed contributions before submission:

- `mustyandcool` — notebook metadata and scraper work.
- `ALMUSTY0` — web sources and raw corpus
- `anselmdan` — JSONL and coverage validation
- `emmanueljoshuashekwolo25-dev` — normalization and tokenizer
- `gimba45` — stop-word and orthography resources.
- `Hassanahmed331` — Zipf analysis
- `isahabdulkadir` — unigram and bigram model
- `Shizzey` — perplexity and report
- `alameenmag` — sources, references, and metrics
- `meethalima913-bot` — notebook execution and format validation
- `ibnbazz` — data quality and duplicate checks
- `musty178` — additional group member; assigned part to be confirmed
- `talk2kabirusman-wq` — additional group member; assigned part to be confirmed

**Team lead**: Yusuf Aminu Abdullahi (`zahfaraway`) — integration and validation. The lead must not claim another member's work as their own.

## Part 2: Data Quality & Duplicate Inspection (Completed by @ibnbazz)
- **Duplicate Verification:** Checked raw and processed corpus files; removed redundant/repeated sentence entries.
- **Sanitization:** Stripped HTML/XML tags, whitespace anomalies, and broken characters.
- **Diacritic Check:** Confirmed that Gbagyi subdots, tone marks, and special characters (`ɓ`, `ɗ`, `ã`, `ẽ`, `ĩ`, `õ`, `ũ`, `ɛ`, `ɔ`) remain fully preserved without loss.
- **Status:** Dataset verified and clean for Zipf analysis and N-Gram modeling.
- ## Part 6: Notebook Execution & Format Validation (Completed by @meethalima913-bot)
- **Notebook Execution:** Automated dry-run of all code cells in `HW1_assignment.ipynb` to verify zero runtime errors.
- **Output Inspection:** Validated execution outputs, plot generation, and matrix outputs.
- **Submission Audit:** Verified file paths, directory structures, and markdown formatting standards for Group 06 submission.
- **Status:** PASSED — Notebook is fully executed and ready for submission.


## Part 3: Zipf's Law & Frequency Analysis (Completed by @Hassanahmed331)
- **Rank-Frequency Distribution:** Evaluated word frequency distribution across unique Gbagyi corpus tokens.
- **Zipfian Dynamics:** Plotted log-rank vs log-frequency curves; confirmed power-law distribution characteristic of natural languages.
- **Top Vocabulary Metrics:** Identified high-frequency functional words and long-tail rare lexical items in the Gbagyi corpus.
- **Status:** PASSED — Zipf frequency analysis completed and verified.
- "docs: add zipf law analysis 
talk2kabirusman-wq
## Supplementary Integration
Assisted with branch merge review, formatting consistency checks, and repository coordination.
