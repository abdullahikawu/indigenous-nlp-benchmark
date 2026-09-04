# Group 6 — Gbagyi Team Plan

## Repository

- Fork: `zahfaraway/indigenous-nlp-benchmark`
- Working branch: `group-06-gbagyi`
- Final pull request target: `abdullahikawu/indigenous-nlp-benchmark:main`

## Required files

```text
data/gbagyi/raw/raw_data_group_06.jsonl
data/gbagyi/processed/cleaned_corpus_group_06.txt
submissions/group_06_gbagyi/HW1_assignment.ipynb
submissions/group_06_gbagyi/group_report.md
```

## Work ownership

Each member must complete a real part of the assignment and make at least
three meaningful commits using their own GitHub account. Do not create empty
commits, reuse another person's identity, or squash away the individual
commit history.

Suggested ownership:

1. **Data collection**
   - Identify suitable public Gbagyi text sources.
   - Implement the `requests` scraper in the notebook.
   - Collect at least 2,500 sentences.
   - Export valid UTF-8 JSONL and document URLs and retrieval dates.
2. **Normalization and tokenization**
   - Remove markup, control characters, and excess whitespace.
   - Preserve Gbagyi diacritics, subdots, and tone marks.
   - Implement the custom tokenizer.
   - Add at least 30 functional stop words and English translations.
   - Export the processed corpus.
3. **Zipf analysis**
   - Calculate token frequencies and ranks.
   - Fit the log-log linear model.
   - Produce the Zipf plot and estimate the exponent `s`.
   - Write the language-specific interpretation.
4. **N-gram model and report**
   - Implement unigram and bigram counts from scratch.
   - Implement Add-1/Laplace smoothing.
   - Evaluate against `tests/test_gbagyi_unseen.txt`.
   - Record the final perplexity and complete the report.

Ownership does not prevent collaboration. Every member should understand the
code they submit and record their actual contribution in the report.

## Branch workflow

Create a personal feature branch from the group branch:

```bash
git fetch origin
git switch -c YOUR-NAME-task origin/group-06-gbagyi
```

After making a meaningful change:

```bash
git add submissions/group_06_gbagyi data/gbagyi
git commit -m "Describe the actual change"
git push -u origin YOUR-NAME-task
```

Open a pull request from the feature branch into `group-06-gbagyi`. The team
lead merges the work without squashing the commits. Avoid editing the notebook
simultaneously with another member; notebook merge conflicts are difficult to
resolve.

## Final checklist

- [ ] At least 2,500 Gbagyi sentences collected by submitted Python code.
- [ ] Raw JSONL has the required `id`, `url`, `date_retrieved`, and `raw_text` fields.
- [ ] Processed text has one sentence per line and single-space token separation.
- [ ] Diacritics, subdots, and tone marks are preserved.
- [ ] Stop-word list contains at least 30 functional words with translations.
- [ ] Zipf rank-frequency plot and exponent are included.
- [ ] Bigram model uses Add-1 smoothing implemented from scratch.
- [ ] Perplexity is calculated on `tests/test_gbagyi_unseen.txt`.
- [ ] Every member has at least three meaningful commits.
- [ ] The final report lists real member contributions.
- [ ] The final PR title is `[Assignment Submission] Group 06 - Gbagyi`.
- [ ] The Google Form contains the PR URL, source URLs, corpus metrics,
  vocabulary size, and final perplexity.
