# Group 02 - Nupe Language NLP Report

**Course:** Indigenous Language AI Benchmark  
**Assignment:** NLP Assignment 1  
**Language:** Nupe  
**Date:** September 2026  


## Team Members

Alamin Ha -  Project Lead 
Zakson01  - Data Collection 
Amma-py (ABDULAZIZ ABDULLAHI) -  Documentation 
[Member 4 Name]  [ID]  [Role] 

## Project Overview

This project involved building NLP tools for the Nupe language, including:

1. **Data Collection** - Scraping 2,500+ Nupe sentences
2. **Text Processing** - Custom tokenization and cleaning
3. **Zipf's Law Analysis** - Frequency distribution analysis
4. **Language Modeling** - Bigram model with Laplace smoothing


## Technical Details

### Data Statistics
- **Total Sentences:** 2,500
- **Vocabulary Size:** 45
- **Total Tokens:** 12,506

### Zipf's Law Results
- **Exponent (α):** 1.3657
- **R-squared:** 0.8262

### Language Model Results
- **Bigram Perplexity:** 45.2607
- **Unique Bigrams:** 724
- **Improvement:** 27.4% over Unigram


## Nupe Language Features

The Nupe language has unique characteristics:
- Uses diacritics: ẹ, ọ, n̄
- Tonal markers affect meaning
- Rich oral tradition

### Example Nupe Words
Nupe - English 
guba - two
eya - friend 
emi - house
tsanka - trousers



## Challenges and Solutions

### Challenge 1: Limited Nupe Text Sources
**Solution:** Identified multiple online sources including news sites and blogs

### Challenge 2: Preserving Diacritics
**Solution:** Used custom regex patterns to preserve ẹ, ọ, n̄

### Challenge 3: Tokenization
**Solution:** Implemented rule-based tokenizer without using NLTK

## References

This work was completed as part of the Indigenous Language AI Benchmark project.

**PR #5:** https://github.com/abdullahikawu/indigenous-nlp-benchmark/pull/5
