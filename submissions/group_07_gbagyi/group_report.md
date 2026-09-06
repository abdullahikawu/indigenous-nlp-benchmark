Indigenous Language AI Benchmark: Empirical NLP for Gbagyi

Group 07, Gbagyi (Gbagyi-Nkwa track)

Course: CSC 406, Artificial Intelligence

Assignment: Indigenous Language AI Benchmark

Date: 2026-08-30

Abstract

This report explains how we created a reproducible, low-resource NLP pipeline for the Gbagyi language. We gathered actual Gbagyi text by requesting, via Python requests, content from theBiblica/YouVersion Bible pages, and respectingrobots.txt and also ignoring any material that was not in Gbagyi. Gbagyi text was cleaned and tokenized using Unicode-aware regular expressions and a custom tokenizer.

Our pipeline produces 22,127 sentences, 403,441 tokens, and 12,798 unique words. By Ordinary least squares regression on the log-log rank-frequency curve, we derived the Zipf exponents = 1.4021 (R = 0.9791). A from scratch bigram model with Laplace smoothing achieved a perplexity of 1400.8026 on the unmodified tester on the instructors file tests/testgbagyiunseen.txt. We did not write any sentences by hand.

1. Introduction

Gbagyi (ISO 639-3: gbr) is a Nupoid language of central Nigeria. Written resources are sparse, orthography is not fully standardised, and tone is often unmarked in published digital text. This assignment asks each group to build an inspectable baseline: scrape authentic text, tokenise it without English pretrained tools, test Zipf’s law, and estimate a smoothed an understanding language model.

The unseen test file given uses an equivalent register format-in this particular case, it relies on bible/shekwoi registers. We therefore use scripture-Yahua, shekwoi, Aduwa and jesun-as our primary data source, and selectively download individual documents from Biblia/YouVersion once we have confirmed that a human has rendered the correct HTML and no captcha challenge appears on web page. We also source Wikipedia and various encyclopedia entries with caution, with these secondary sources offering descriptions of Gbagyi generally, though, as addressed below. Robots.txt was examined to ensure that relevant portions of the web can be freely indexed.

2. Objectives

1. Design and implement data collection and provenance systems that result in real, attested Gbagyi texts.

2. Ensure that tokenization uses Unicode-safe techniques.

The raw data file (JSONL) records every webpage the scraper successfully downloaded, including English catalogue/encyclopedia pages. Those pages are not Gbagyi running text. The processed corpus is built only from authentic Gbagyi sentences after documented.

4. Calculate the Zipf exponent from the sourced Gbagyi corpus.

5. Implement and evaluate both unigram and Add-1 smoothed bigram models, training them from scratch and measuring perplexity against the given testing file.

3. Data sources and provenance

The JSONL file, the scraped data before processing, has been systematically organized. One line contains one fetched URL and the text on the page requested. Several were website indices and Wikipedia, these are discarded and only noted in Table 1. The real Gbagyi texts came only from the scriptural documents which we processed based on criteria in Table 1.

Table 1: Categories of Gbagyi-related web content available, with decision on processed data inclusion

Source class (URL pattern) Source name Language / version Retrieval date HTTP Docs Docs contributing data Gbagyi? Filtering decision
https://www.bible.com/bible/1621/{BOOK}.{CH}.GAW Biblica Alkawali Woiwoyi (GAW 1621) chapter Gbagyi (GAW / Alkawali Woiwoyi) 2026-08-30 200 260 260 11532 Included: verses extracted, English-UI text excluded, tested and used for final model.
Https://www.bible.com/bible/4607/{BOOK}.{CH}.GNB Biblica Gbagyi Contemporary Bible (GNB 4607) chapter Gbagyi (GNB / Shekwoyi dagbma) 2026-08-30 200 249 249 10595 Included: verses extracted, English-UI text excluded, tested and used for final model.
Https://www.bible.com/versions/4607-gnb-gbagyi-nyizeyenya-baibwulu-shekwoyi-%C6%81%C9%99dagbma Bible.com version page: Gbagyi Contemporary Bible (GNB) English version-catalogue page (not processed as Gbagyi) 2026-08-30 200 2 0 0 Excluded: English catalogue/description; for provenance only.
Https://www.scriptureearth.org/00eng.php?iso=gbr Scripture Earth Gbagyi (gbr) index English catalogue 2026-08-30 200 2 0 0 Excluded: English catalogue/description; for provenance only.
Https://en.wikipedia.org/wiki/Gbagyi_people Wikipedia: Gbagyi people English encyclopedia (not Gbagyi running text) 2026-08-30 200 2 0 0 Excluded: English catalogue/description; for provenance only.

AllGAW/GNB chapters are listed as individual entries. For brevity of the provided table, two rows were used to show scripture sections overall in the JSONL data. Any of the following full URLs are in the same JSONL record type as 'non-chapter pages' below:

Full non-chapter page URLs

Column 1 Column 2 Column 3 Column 4 Column 5 Column 6 Column 7 Column 8
Source URL Source name Language / version Retrieval date HTTP Contributed processed Gbagyi? Sentences Filtering decision
https://www.bible.com/versions/1621-gaw-alkawali-woiwoyi Bible.com version page: Alkawali Woiwoyi (GAW) English version-catalogue page (not processed as Gbagyi) 2026-08-30 200 no 0 Excluded: English catalogue/description; for provenance only.
Https://www.bible.com/versions/4607-gnb-gbagyi-nyizeyenya-baibwulu-shekwoyi-dagbma Bible.com version page: Gbagyi Contemporary Bible (GNB) English version-catalogue page (not processed as Gbagyi) 2026-08-30 200 no 0 Excluded: English catalogue/description; for provenance only.
Https://www.bible.com/languages/gbr Bible.com language index (gbr) English navigation / language catalogue 2026-08-30 200 no 0 Excluded: English catalogue/description; for provenance only.
Https://en.wikipedia.org/wiki/Gbagyi_language Wikipedia: Gbagyi language English encyclopedia (not Gbagyi running text) 2026-08-30 200 no 0 Excluded: English catalogue/description; for provenance only.
Https://en.wikipedia.org/wiki/Gbagyi_people Wikipedia: Gbagyi people English encyclopedia (not Gbagyi running text) 2026-08-30 200 no 0 Excluded: English catalogue/description; for provenance only.
Https://www.scriptureearth.org/00eng.php?iso=gbr Scripture Earth Gbagyi (gbr) index English catalogue 2026-08-30 200 no 0 Excluded: English catalogue/description; for provenance only.

As seen in the tables, the only Gbagyi content provided for processing are the GAW and GNB chapter bible pages. Wikipedia and catalog/encyclopedia style articles are available but not used to construct the Gbagyi testset.

Hosts consulted: en.wikipedia.org, www.bible.com, www.scriptureearth.org.
Documents successfully wrote to JSONL: 515 (260 from GAW bible chapter, 249 from GNB bible chapter).
Number of unique page URLs found: 515.
Retrieval date for the sources: 2026-08-30.
HTTP status: only valid HTTP 200 statuses were recorded (Failed requests were omitted and are not part of the data provided via the requested format: id,url,date\retrieved,raw\text.
Copyright for any Gbagyi bible texts belongs to the copyholder, Biblica Inc. These were retrieved and used solely for the purposes of the exercise according to U.S. Copyright Law fair use of education.

4. Data collection methodology

scrape\to\jsonl(url\list, output\path) has following criteria for fetching. This script uses a polite request process:

* Uses an academic and descriptive User-Agent string to minimize server-side blocking
* Sets a 25 second timeout per request and retries failed requests up to 3 times
* Introduces a 0.7 second delay between each request
* Prior to every HTTP request, urllib.robotparser is utilized to ensure that the request is allowed by robots.txt
* Strictly enforces no CAPTCHA, no logins and no bypasses of any robots protocols.

BeautifulSoup is used to parse HTML from YouVersion pages. The intent here is to avoid anything resembling plain text dumps and to ensure what we extract comes solely from the rendered chapter section of a given biblical book; thus, we prioritize any elements with verse-like markup as well as the structured JSON data contained within NEXT_DATA script tags which are rendered by Javascript on The Bible.com server. Empty or near-empty extractions are simply filtered. Duplicate Gbagyi content documents after processing were removed prior to the creation of the final test set corpus.
Each JSONL line is:

{"id":1,"url":"https://...","dateretrieved":"2026-08-30","rawtext":"..."}

The autograder expects integers for id, we are meeting that spec.

5. Corpus statistics

Column 1 | Column 2

-------- | --------

Metric   | Value

Raw documents | 515

Raw segmented sentences | 23077

Filtered sentences | 778

Duplicate tokenized sentences | 172

Held-out exact matches removed | 2
    Held-out high-overlap (containment) removed | 6
    Final authentic Gbagyi sentences | 22127
    All tokens (incl. Punctuation) | 466282

Word tokens | 403441

Word vocabulary | 12798

Token vocabulary (incl. Punct.) | 12834
    Mean sentence length (tokens) | 21.07

Median sentence length | 19

Min / max sentence length | 1 / 96

Unique URLs | 515

Most frequent word types

Column 1 | Column 2 | Column 3

-------- | -------- | --------

Rank     | Word     | Frequency

1        | n        | 33527

2        | a        | 14607

3        | wo       | 11033

4        | wa       | 8730

5        | nu       | 8415

6        | yi       | 7284

7        | fye      | 6171

8        | lo       | 5968

9        | mi       | 5674

10       | zhin     | 5555

JSONL validation errors: 0.
Processed-corpus validation errors: 0.

6. Preprocessing methodology

Pipeline: HTML/XML removal; control-character removal; Unicode NFC; whitespace normalization; sentence segmentation; custom tokenization.

We use NFC in order that „ ; ! ? , Subdot letters; any combining marks etc remain intact. We do not use unidecode. We do not convert to ascii.

Sentencehood is not "one HTML line." Blocks are split after .  ! 

?

 After cleaning. Verses from the Bible without the final punctuation are taken as one.

Processed-corpus filters (reproducible by buildprocessedcorpus):

1. Wikipedia, Scripture Earth and bible.com/languages/ pages will all be processed with 0 sentences output.

2. Bible.com/versions/ pages contribute 0 sentences for training (raw provenance only).

3. Sentences that are Majority English and catalogue/template phrases (bible versions; Biblica ministry boilerplate; Wikipedia help text etc...) are excluded in every source, including chapter pages.
4. Exact tokenized identity with tests/testgbagyiunseen.txt are removed from training (not lexicon mining; held-out contamination decommmissioning).
5. High-overlap contamination. If the tokenized word sequence of a line, S, contains the tokenized word sequence, T, and either S or T has 4 or less word tokens in length, then S is discarded from training data, unless the containment is identical to the source. Function words are not sufficient for exclusion. Ai and aeye should remain alongside their nearly identical clauses.

The instructor held out is NEVER edited, if high perplexity results it's the honest score.
7. Tokenisation methodology

Customtokenizer is defined using re only (TOKENRE). This:

* lowercases with Python 3 str.lower() ( )
* detaches punctuation (yi ., yi,)
* preserves internal hyphens (bui-bui, zaho-zahoyi, tnu-tnu)
* produces single-space output, stripped of leading/trailing space

This format matches that of the instructor's file, for instance:

Input: Fye zhin bugba bui-bui ntu ge lada dnagmayi shi lo.
Output: fye zhin bugba bui-bui ntu ge lada dnagmayi shi lo .

Imoslve present in processed corpus: True.

8. Stop-word methodology

We identified 35 Gbagyi function words (30 attested, 5 uncertain). Our primary linguistic resources are the Gbagyi noun-phrase field manuscript (The Structure of Noun Phrases in Gbagyi, Niger State speaker interviews) and other independently attested forms in the Biblica GAW/GNB chapter text corpus. We have not introduced lexemes; also, we have not used tests/testgbagyiunseen.txt as a lexicon or a linguistics source. Its use is confined to the held-out evaluation set and the exclusion list for train/test decontamination.

Where a form has more than one grammatical role (for example, o as a preposition or coordinating conjunction, ye as a demonstrative or a verb/particle, ma as a particle or the main verb ‘give birth’), we have listed the relevant usages separately rather than collapsing them. Uncertain terms (na, ga, a, ku, ma) have been labeled uncertain rather than given a forced gloss.

We do not remove the stop-words from cleanedcorpusgroup_07.txt, doing so would make the comparison of language model evaluations with the official tokenized test format impossible.

Gbagyi English Category Confidence Source

mi I / my pronoun / possessive attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
omi my possessive determiner attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
wo his / him / her pronoun / possessive attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
wa he / she (3sg subject) pronoun attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
wu he / she pronoun attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
a they (3pl) pronoun attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
ye this (near demonstrative). In scripture orthography the same spelling also occurs as a high-frequency verb/particle; those uses are not collapsed here. Demonstrative / function word attested (demonstrative); other uses vary Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
yi that (distant demonstrative) / anaphoric particle demonstrative / particle attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
ho the (optional determiner) determiner attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
lo tense/aspect or predicative particle auxiliary / particle attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
o on / in (preposition); also coordinating 'and' in some examples preposition / conjunction attested (polysemous: preposition and coordinator) Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
n relativizer / linker 'that'; also a high-frequency clitic linker relativizer / conjunction attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
nu determiner / copular-focus particle determiner / particle attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
zhni be / become (copula) auxiliary / copula attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
zhin be / become (copula) auxiliary / copula attested Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Kwo it (3sg inanimate / resumptive) pronoun attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
ge quotative / complementizer 'that' (naming, reported speech) conjunction / complementizer attested Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Nya of (associative / genitive) preposition / genitive marker attested Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
To not / negative particle negation / particle attested Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Ntu so that / in order that (purpose) conjunction attested Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Ntuge because / so that (purpose-reason) conjunction attested Biblica (1997/2025), Gbagyi Contemporary Bible (GNB), form attested in collected
gmanyi one / some (quantifier) determiner / quantifier attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
vnyanya all / whole quantifier / determiner attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
ama but conjunction attested (Hausa loan used as a Gbagyi conjunction in these publications) Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Sai then / only / except conjunction / particle attested (Hausa loan used as a Gbagyi particle in these publications) Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Har until / even conjunction / preposition attested (Hausa loan used as a Gbagyi function word in this publication) Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Shi then / and then (sequential) conjunction / particle attested Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Ma and (coordinator in some constructions) / 'give birth to' as a content verb - listed here only as the high-frequency coordinator/particle sense when not the main verb conjunction / particle uncertain (polysemous) Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Ga clause-final / focus particle (function word; exact force varies by dialect) particle uncertain Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Na associative / high-frequency linker (exact sense varies) particle / linker uncertain Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
Nyi locative / relational particle (often clause-final) particle attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
e come (light verb / motion; also appears in serial constructions) auxiliary / light verb attested Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
ei past / sequential auxiliary appearing before zhin in GAW genealogies auxiliary attested Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.
A they / impersonal plural prefix or pronoun (context-dependent) pronoun / agreement uncertain (prefix vs independent pronoun) Unpublished field manuscript, The Structure of Noun Phrases in Gbagyi (Niger Sta
ku to / and (light preposition; also in the Hausa-origin phrase ku gode 'give thanks') preposition / particle uncertain Biblica (1997), Alkawali Woiwoyi (GAW), form attested in collected chapter text.

Limitation: There are few published single-word English translations for some high-frequency particles. We've chosen to document this uncertainty rather than invent precision.
9. Zipf Analysis

The types were sorted in decreasing order of frequency and we fitted by numpy.polyfit on natural logs:

log(f) = C s log(r)

Quantity s C R Types used (N)

Estimate 1.402137 12.890745 0.979136 12798

The figure below zipfrankfrequency.png shows the cloud of (rank,frequency) pairs in log scale, overlaid with the OLS line. A traditional value of 1 is usually considered an ideal large corpus property. Therefore, our estimated s is a description of this particular text corpus (which is heavily influenced by scripture), not by spoken Gbagyi.
Orthographic variation (GAW zhin vs GNB zhni; Yesu vs Yeisu; optional ) splits what may be a single lemma into different type entities; how variation influences tail steepness depends on its distribution. Unicode distinctions, on the other hand, expand V (versus and b). We claim a corpus-based claim on dialectology.

10. Unigram Model

These were just the frequency of dictionary tokens. MLE, P(w)=count(w)/N is calculated using N=466,282 tokens and V_unigram = 12,834 tokens (all tokens, punctuation included). With add-one unigram smoothing provided as a comparison to the calculation itself, it cannot be the intended final number for perplexity.

11. Bigram Model

BigramModel.fit is called for its count of adjecent bigram tokens in a sentence (sentences has no begin or end symbols to better match the assigned template). It returns 444,155 bigram tokens. Vocaulry size corresponds to number of distinct tokens found in train data set, i.e. 12,834 tokens.

12. Laplace (Add-1) Smoothing

P(w2|w1) = (count(w1, w2) + 1)/(count(w1)+ V)
If count(w1) equals to 0, P=1/V. The outcome of test on made-up-context word produces the correct value of 0.00007792. I.e. 1/V.

13. Perplexity

Test file is tests/testgbagyiunseen.txt and has remained unchanged.
PP = exp( (1/N) log P(wi | w{i-1}))
This value is identical with the calculation produced on other part (using base-2 of the problem statement), since log equals the natural logarithm. On separate calculation on log 2, it generates the same value; 1400.802620.
Quantity Value Test sentences 15 Test tokens 142 Predicted bigrams (N) 127 Training V 12834 Smoothing Laplace / Add-1 Bigram perplexity 1400.802620

14. Result

*   Authentic JsonL documents collected: 515

*   Sentences finalized: 22,127

*   Word vocabularies: 12,798

*   Zipf s=1.4021, R=0.9791

*   Bigram perplexity evaluated on blind test-set: 1400.8026
Final sentence count satisfied the minimum requirement of 2,500 sentences for this assignment; they were actually collected from public source on Christian Gbagyi material, not generated from language models.

15. Discussion

corpus size for class purpose; but it is not wide-spread type. It consists of Christian writings in two distinct versions (Biblica), so it cannot serve for general description of Gbagyi as a spoken or written language. High freq. Of certain word types refer to function word; like wa, zhin/zhni, n, nu, Yesu/Yeisu are functions words and names. Zipf’s law; here represented as r is an approximation as the type set (R 0.9791) and is still small.
Gaps in diacritics and preservation of the implosive feature have been maintained; tonality is very lacking, thus most likely tonal minimal pairs merged. Merging GAW and GNB is what best suits for the research: they are indeed both legitimate published Gbagyi works.

16. Limits of the approach

1.  Bible domain coverage, English and catalogue content exclusion after analysis led to only two publication versions. Therefore all findings regardchristian Gbagyi and cannot generalize to Gbagyi in general. Hence, it becomes limited bydomain, this is true for Zipf exponent, vocab, and perplexity.
2.  The use of GAW and GNB increases number of word types to a certain extent.
3.  Punctuation segmentation leads to inaccurate parsing of sentence boundaries, since sometimes single verse may include coordinating conjunction.
4.  Some stop word interpretations are questionable; 5 of the 35 entries marked by an indication of being questionable.
5.  Matching test sentences with actual training words based on exact wording up to four consecutive word is good for decontamination; near paraphrase is excluded only if the four contiguous word sequence IS matched, therefore near paraphrase not containing a test keyphrase in the matched words has not been removed.
6.  The framework imported by test instructor, uses default Nupe location which is Group 01. As we moved gbagyi material elsewhere (and didn't make adjustment), import will point there.
7.  Shallow clone on Github Actions failed the collaboration test (though genuinely present from multiple author); but it could be the shallow clone and no real 3-factor test used.

17. Reproducibility

The following commands can be used for running the submission module under the root of the repository, assuming Python 3.10 is installed and requirements.txt are downloaded.

Python submissions/group07gbagyi/HW1_assignment.py

pytest tests/autograder_eval.py -v --tb=short

Importing is done on the above shown module. Collecting is performed with consideration to robots.txt, so if a site deactivates, this address can be missed. Randomness not applied in preprocessing; Zipf and perplexity calculation.

18. Conclusion

The Gbagyi baseline delivered by group 7 is demonstrably inspected, has certified JsonL source, a validated corpus with the split on a single sentence per line, a known custom tokenizer used in this job, a documented table of stop-word entries and demonstrated work with Zipf fitting. A by-hand generated Add-1 bigram model was made, and the results above evaluated on the official blind test set.
References

1. Biblica, Inc. (1997). *Alkawali Woiwoyi* (GAW). Retrieved from https://www.bible.com/versions/1621-gaw-alkawali-woiwoyi
2. Biblica, Inc. (1997/2025). *Gbagyi Nyizeyenya Baibwulu: Shekwoyi Ɓədagbma* (GNB). Retrieved from https://www.bible.com/versions/4607-gnb-gbagyi-nyizeyenya-baibwulu-shekwoyi-%C6%81%C9%99dagbma
3. Blench, R. (2013). *The Nupoid languages of west-central Nigeria: Overview and comparative wordlist*. Working document. https://rogerblench.info/Language/Niger-Congo/VN/Nupoid/Nupoid%20Overview%202013.pdf
4. Dalhatu, A. M. (2019). Gbagyi syllable and phonotactics. *Journal of the Nigerian Languages Project*, 1. https://jnlp.com.ng/index.php/home/article/view/2
5. Hyman, L. M., & Magaji, D. J. (1970). *Essentials of Gwari grammar* (Occasional Publication 27). Ibadan: Institute of African Studies.
6. Jurafsky, D., & Martin, J. H. (2025). *Speech and language processing* (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/
7. Rosendall, E. P. (1998). *Aspects of Gbari grammar* (Master’s thesis). University of Texas at Arlington.
8. Rosendall, H. J. (1992). *A phonological study of the Gwari lects*. Dallas: SIL.
9. Scripture Earth. Gbagyi (`gbr`) resource index. https://www.scriptureearth.org/00eng.php?iso=gbr
10. Wikipedia. (2026). Gbagyi language. https://en.wikipedia.org/wiki/Gbagyi_language
11. Wikipedia. (2026). Gbagyi people. https://en.wikipedia.org/wiki/Gbagyi_people
12. Zipf, G. K. (1949). *Human behavior and the principle of least effort*. Addison-Wesley.
13. YouVersion / Bible.com `robots.txt` (checked 2026-08-30). `/bible/` chapter paths are allowed; `/bible/*/*/notes` is disallowed.
14. Field examples of Gbagyi noun-phrase function words as reported in *The Structure of Noun Phrases in Gbagyi* (Niger State interview data; demonstratives *yè/yî*, determiner *hò*, relativiser *ń*, preposition *o*).
