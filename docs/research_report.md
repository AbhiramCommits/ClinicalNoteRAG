# ClinicalNoteRAG: Citation-Grounded Retrieval-Augmented Question Answering over Synthetic Clinical and Payment Records

## Abstract

Generative AI in healthcare treatment, payment, and operations (TPO) must
ground claims in source evidence, cite it, and abstain when the record lacks an
answer. ClinicalNoteRAG is a reproducible research system over entirely
synthetic FHIR R4 bundles and synthetic clinical/payment notes. It generates
300 patients and 1,478 documents, injects fake PHI, de-identifies notes with
typed reversible placeholders, chunks notes and FHIR facts, embeds 13,605
chunks, and serves hybrid dense/BM25 retrieval over FAISS or pgvector. Answers
are extractive and cited, checked sentence by sentence, and replaced by an
explicit abstention when evidence is missing. On a 235-item closed-world
benchmark, the baseline reaches Recall@5 1.000, MRR 0.941 [0.918, 0.962], exact
match 1.000, hallucination rate 0.000, false-answer rate 0.000, and abstention
precision/recall 1.000/1.000. A patient-disjoint comparison shows a trained
fallback reranker improves MRR from 0.929 to 1.000 (paired delta +0.071, 95% CI
[0.014, 0.129], p=0.011), while a 12-step Qwen2.5-0.5B LoRA adapter fails the
citation contract (94.3% unsupported-claim rate). These are upper-bound
synthetic results, not clinical validation.

## 1. Background

Large language models can answer medical questions fluently, but fluency is not
evidence that an answer is grounded in a specific record [Singhal et al., 2023;
Thirunavukarasu et al., 2023]. TPO questions are narrower and auditable: given
this encounter or claim, what does the documentation say, which code or amount
supports it, and where is the source? An unsupported sentence is a safety,
compliance, and financial risk.

RAG conditions generation on retrieved passages [Lewis et al., 2020; Gao et
al., 2023], but does not guarantee faithfulness. Systems can still hallucinate,
cite the wrong passage, over-answer when evidence is absent, or leak
identifiers into indexes and logs. Evaluation often reports a single aggregate
without uncertainty and conflates retrieval quality with answer quality.

This work builds the full path from synthetic corpus generation to served,
cited answers and measured evaluation. Code-generated FHIR and templated notes
give exact ground truth for answers, codes, claims, denial reasons, and
supporting chunks. The benchmark is therefore closed-world and easy, but the
engineering and measurement questions are inspectable end to end.

## 2. Methodology

### 2.1 Synthetic corpus

`clinicalnoterag.data.synthea_gen` generates Synthea-style FHIR R4 bundles
without downloading patient data. Each patient receives demographics, one to
four encounters, ICD-10-CM conditions, CPT procedures, SNOMED CT vital signs,
RxNorm medication orders, a claim, and a ClaimResponse with a CARC-style denial
reason. The build produced 300 patients, 739 encounters, 1,104 conditions,
1,429 procedures, 2,194 observations, 749 medication requests, 739 claims, and
230 denied claims. Pydantic models validate every resource.

For each encounter, `notes_gen.py` emits a clinical note (Chief Complaint,
History of Present Illness, Assessment and Plan, Discharge Summary) and a
payment note (denial rationale, appeal-letter draft). Notes repeat the
encounter's ICD-10/CPT codes and intentionally embed fake names, MRNs, SSNs,
phones, emails, dates, addresses, and ZIPs. Per-document injected values give
the audit exact truth.

### 2.2 De-identification

`DeidEngine` tries Presidio's AnalyzerEngine and AnonymizerEngine first. If the
package or spaCy model cannot load, it uses a deterministic regex/gazetteer
backend for NAME, MRN, SSN, PHONE, EMAIL, DATE, ADDRESS, and ZIP. Replacements
are typed placeholders such as `[NAME_1]`; placeholder-to-original maps and
spans are written to `artifacts/deid_maps/`, outside the vector store. Overlap
resolution favors longer spans, so an address wins over its ZIP and a full name
over a fragment. The audit re-scans scrubbed text for injected strings and
reports the document leak rate and per-entity recall.

### 2.3 Chunking, embeddings and stores

Notes are split on heading lines, then windowed at 256 whitespace tokens with
32-token overlap. Every Condition, Procedure, Observation, MedicationRequest,
Encounter, Claim, and ClaimResponse becomes one fact chunk with metadata for
patient, encounter, resource, code system, code, service date, claim and denial
code. The corpus produced 13,605 chunks (7,693 fact, 5,912 note).

Embeddings are pluggable. The build used
`sentence-transformers/all-MiniLM-L6-v2` (384-d). The offline fallback is
hashed TF-IDF over word unigrams, bigrams, and character 3–5-grams, reduced to
512 dimensions and L2-normalized. The selected backend is recorded in every
ingest artifact.

`FaissStore` uses an `IndexFlatIP`/`IndexIDMap2` with a SQLite sidecar;
filtered search scans only SQL-selected rowids and scores them exactly, which
prevents cross-patient leakage. `PgVectorStore` uses cosine distance over a
`vector` column with SQL filters. Both pass the same contract suite.

### 2.4 Retrieval, generation, abstention

`HybridRetriever` combines dense search with BM25 and fuses ranks with
reciprocal rank fusion (k=60), optionally after metadata prefiltering. Results
retain dense, BM25, fused scores and rank.

The default generator is deterministic and extractive. It recognizes
diagnosis, procedure, medication, observation, denial-reason, denial-code,
count, total-billed, and appeal questions, renders retrieved facts, and attaches
inline markers such as `[1]`. It never composes content absent from context.
Local HF (Qwen2.5-0.5B-Instruct, flan-t5-base fallback), OpenAI, and LoRA
backends share the prompt contract: answer only from numbered context, cite
every sentence, and output exactly "Insufficient evidence in the available
record" when evidence is absent.

`ClinicalAnswerer` abstains when retrieval is empty, the top fused score is
below 0.012, a hard identifier/code in the question is absent from evidence, or
the generator reports insufficiency. The verifier splits answers into
sentences, requires a citation on each, checks content-token overlap against
the cited chunk, and requires sentence codes to appear there. A failure marks
the answer `unsupported_claim=True`.

### 2.5 Tuning and evaluation

SFT pairs use numbered context plus question as prompt and a cited gold answer
(or abstention target) as completion. Splits are strictly by patient: 24 train
(162 items), 5 validation (30), 5 held-out test (43). A PEFT LoRA adapter
(r=8, alpha=16, dropout=0.05, `q_proj,v_proj`) is trained with TRL SFTTrainer
for 12 steps on CPU. A fallback cross-encoder-style MLP reranker is also
trained on 32 retrieval features and the same splits; it is labeled as the
fallback learned component.

The benchmark has 235 items: 75 treatment, 60 payment, 60 operations, 40
adversarial. Gold answers, codes, chunks, and numeric values come from the
FHIR bundles. Metrics include Recall@k, MRR, exact match, token F1, numeric and
code-aware accuracy, unsupported-claim rate, false-answer rate, abstention
precision/recall, and code micro/macro P/R/F1. Means get 95% percentile
bootstrap CIs (10,000 resamples, seed 42); system comparisons use a paired
bootstrap.

## 3. Experimental setup

The build ran on macOS arm64 (8 CPU, 16 GB RAM, no GPU), Python 3.11.14,
PyTorch 2.15.0, Transformers 5.23.0, Sentence-Transformers 6.3.0, and FAISS
1.16.0. `make reproduce` from a clean generated-data state exited 0 in 20:53.
The main index used MiniLM; tests and offline smoke used hashing. The main
generator was `extractive-offline`. A 12-step LoRA adapter and a fallback
reranker were trained. The commercial API was not run (no `OPENAI_API_KEY`).
Airflow was not installed; the DAG imported through a shim and the same
functions ran locally.

## 4. Results

### 4.1 Data and retrieval

De-identification scrubbed 1,478 notes in 6.006 s. Ingestion processed 1,778
documents into 13,605 chunks in 75.014 s (181.37 chunks/s), with 66.789 s for
embedding (203.70 chunks/s) and 1.223 s for the idempotent FAISS upsert. The
refresh path completed five tasks in 97.276 s; a second upsert inserted 0 and
left the store at 13,605.

On the 235-item benchmark at k=5 (hybrid, seed 42):

| Metric | Value [95% CI] | n |
| --- | --- | ---: |
| Recall@5 | 1.000 [1.000, 1.000] | 195 |
| MRR | 0.941 [0.918, 0.962] | 195 |
| Exact match | 1.000 [1.000, 1.000] | 195 |
| Token F1 | 1.000 [1.000, 1.000] | 195 |
| Numeric accuracy | 1.000 [1.000, 1.000] | 80 |
| Code-aware accuracy | 1.000 [1.000, 1.000] | 135 |
| Hallucination rate | 0.000 [0.000, 0.000] | 195 |
| False-answer rate | 0.000 [0.000, 0.000] | 40 |
| Abstention precision / recall | 1.000 / 1.000 | 235 |
| Code P/R/F1 micro and macro | 1.000 / 1.000 / 1.000 | 135 |

Answer latency was p50 32.969 ms and p95 41.276 ms; mean cost/query was
$0.000000. Overall abstention was 0.170 because 40 of 235 items were
adversarial; answerable abstention was 0.000.

Ablations show retrieval budget dominates. At k=1, dense recall was 0.650 and
exact 0.497; BM25 0.583/0.405; hybrid 0.619/0.451. At k=3 all modes reached
recall 0.985 and exact 0.949–0.959. At k=5 and 10 recall was 1.000. Dense had
the best MRR at k=5 (0.956) versus hybrid 0.941 and BM25 0.923. Hybrid was
slower (p50 33.204 ms at k=5 vs 15.349 dense and 19.347 BM25). Fusion did not
improve recall on this corpus; it preserved ranking while adding a lexical
channel.

### 4.2 Multi-system comparison

The held-out set had 43 items from 5 patients:

| System | Recall@k | MRR | Exact | Token F1 | Hallucination | False answer | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| zero-shot-no-rag | 0.000 | 0.000 | 0.000 | 0.037 | 0.000 | 0.000 | 0 | 0 |
| rag-zero-shot | 1.000 | 0.929 | 1.000 | 1.000 | 0.000 | 0.000 | 42 | 110 |
| rag-tuned (reranker) | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 36 | 46 |
| rag-lora | 1.000 | 0.929 | 0.000 | 0.403 | 0.943 | 0.000 | 12,918 | 21,922 |
| rag-commercial-api | not run — no API key | | | | | | | |

Paired bootstrap deltas (tuned vs zero-shot): MRR **+0.0714 [0.0143, 0.1286],
p=0.011**; recall, exact match, token F1, hallucination, and false answer were
all 0.000 with p=1.000 because the baseline was already perfect. The reranker
improves ranking, not coverage or accuracy, on this easy set. The LoRA adapter
learned style but not citations: 94.3% of answerable answers contain an
unsupported sentence and exact match is 0. The commercial row is not invented.

### 4.3 Training and service

The LoRA run used 162 train and 30 validation examples, 12 steps, CPU fp32, and
341.539 s; validation loss fell from 2.162 to 2.025. The fallback reranker used
272 train and 45 validation pairs, 120 epochs, hidden 32; best validation loss
was 0.185113, and held-out retrieval MRR rose from 0.928571 to 1.000000 with
Recall@5 unchanged at 1.000.

A live uvicorn API received 102 real queries (48 treatment, 16 payment, 30
operations, 8 adversarial); all returned 200. Service p50 was 35.996 ms and p95
112.889 ms; JSONL telemetry gave 35.485/107.180 ms. All adversarial queries
abstained and the other 94 did not. Mean citations per answer was 1.608;
unsupported-claim rate was 0.000. The suite has 76 tests and 69.96% coverage;
the offline smoke path passed 5/5 on 20 patients and 838 chunks.

## 5. Error analysis

The k=5 baseline answered every answerable item exactly, so the useful failures
come from the k=1 stress condition (`artifacts/eval/rag-k1-error-analysis.json`).

1. **T-002 (procedures, recall 0.5).** Only CPT 99213 was retrieved; the
   comprehensive metabolic panel (80053) was omitted. The answer was fluent and
   correctly cited but incomplete. This is retrieval budget, not generation.
2. **T-014 (procedures, recall 0.333).** Only CPT 99283 was retrieved; CPT
   71046 and 99213 were absent. Listing questions need exhaustive encounter
   coverage, which fixed top-k does not guarantee.
3. **T-024 (medications, recall 0.5).** Only metoprolol (866924) was retrieved;
   acetaminophen (313782) was missing. Citation verification passed because
   every cited sentence was supported; coverage failure is invisible to it.
4. **O-001 (encounter count, recall 0.5).** The system answered 1 while the
   gold count was 2, because only one encounter was retrieved. Aggregation is
   acutely sensitive to missing facts.
5. **P-001 (payment denial, recall 0.5).** Only the ClaimResponse was retrieved;
   the Claim was missing. The generator abstained with
   `generator_insufficient`. This is safe behavior but lowers answerable
   coverage; claim questions need scoped expansion across the Claim and
   ClaimResponse pair.
6. **LoRA adapter.** 94.3% unsupported answers and no exact matches. A typical
   failure is a short diagnosis phrase without `[n]` markers; the verifier
   flags every sentence. Twelve steps on 162 examples shift style but not
   citation discipline.

## 6. Discussion

The strongest result is inspectability: every answer carries chunk ids and
quote spans, every sentence is checked, and missing evidence yields abstention.
The k=1 failures show why retrieval coverage must be measured separately from
fluency: three failures produce grammatical, cited, yet incomplete answers.

Hybrid retrieval did not beat dense here; benchmark questions and fact chunks
share code tokens, so both channels find the same evidence and RRF mainly
reorders. The reranker improved ranking because it learned code, identifier,
and lexical-overlap signals that separate gold chunks from other patient facts.

The LoRA result is a useful negative finding. A 0.5B adapter trained for 12
steps on 162 synthetic examples does not learn citation discipline. Adapter
training must be evaluated with the same verifier and abstention metrics as the
base system, not only with loss.

## 7. Limitations

All data are synthetic and templated; notes repeat FHIR codes, making the task
easier than real documentation. The benchmark is closed-world and shares
bundles with the index. No clinician reviewed questions, answers, or citation
spans. De-identification recall is exact only for injected strings and the
synthetic gazetteer; Presidio did not run. The extractive generator's zero
hallucination is structural. The LoRA adapter is tiny and undertrained. The
commercial baseline, Airflow scheduler, full-scale pgvector ingest, and cloud
deployment were not executed; the latter two were contract-tested or rendered
as dry-run manifests. CIs of [1.000, 1.000] reflect a small, easy benchmark.

## 8. Future work

Priorities are an open-world benchmark with paraphrased notes, missing and
conflicting evidence, and clinician-written questions; realistic
de-identification evaluation; coverage-aware retrieval that expands
claim/encounter fact sets; larger generators evaluated with the verifier and
abstention policy; human review of citation spans; durable telemetry and drift
detection; and full-scale pgvector and cloud validation under security and
HIPAA review.

## 9. References

- Cormack, Clarke & Buettcher (2009). Reciprocal rank fusion outperforms
  Condorcet and individual rank learning methods. SIGIR.
- Efron & Tibshirani (1993). An Introduction to the Bootstrap. Chapman & Hall.
- Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models:
  A Survey. arXiv:2312.10997.
- HL7 (2019). FHIR R4 specification. https://hl7.org/fhir/R4/
- Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
  arXiv:2106.09685.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks. NeurIPS.
- Microsoft (2024). Presidio: Data Protection and De-identification SDK.
- Robertson & Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and
  Beyond. Foundations and Trends in Information Retrieval.
- Singhal et al. (2023). Large Language Models Encode Clinical Knowledge.
  Nature.
- Thirunavukarasu et al. (2023). Large language models in medicine. Nature
  Medicine.
- Walonoski et al. (2018). Synthea: generating synthetic patients and the
  synthetic electronic health care record. JAMIA.
