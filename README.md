# basket_ai — Multi-Signal Basket Recommendation Pipeline

A production-style, reproducible **candidate generation pipeline** for basket-based
recommendation systems, built using multiple complementary signals and evaluated via
offline diagnostics and recall-based metrics.

This project focuses explicitly on the **pre-ranking stage** of a recommender system,
where the primary objective is to generate a **high-quality, diverse, and robust
candidate pool** for downstream ranking models.

---

## 🧠 System Architecture (High-Level)

Raw Data
├── Transactions
├── Product metadata
├── Category hierarchy
├── External trends
↓
Feature & Graph Construction
├── Co-occurrence matrix
├── Association rules
├── Category tree
├── Embeddings (Item2Vec)
↓
Candidate Generation (Multi-Signal)
├── Rules-based
├── Co-occurrence
├── Category expansion
├── Embedding similarity
↓
Candidate Pool Diagnostics
├── Coverage
├── Source diversity
├── Recall@K
↓
Learning-to-Rank (XGBoost Ranker)
↓
Offline Evaluation

---

## 1. Motivation and Problem Framing

In large-scale recommender systems, overall performance is often constrained not by
the ranking model itself, but by the **quality of the candidate pool**.

If relevant items are not surfaced during candidate generation, no downstream model
can recover them.

This project addresses the following core questions:

- Can we generate a **non-empty, sufficiently rich candidate pool** for most baskets?
- Do different recommendation signals reinforce each other?
- How stable is candidate generation across basket sizes and sparsity levels?
- Does multi-signal blending improve recall over single-signal approaches?

The project is deliberately designed to isolate and evaluate **candidate generation**
as a standalone system component.

---

## 2. Design Principles

The pipeline is built around senior-level engineering principles.

### 2.1 Separation of Concerns
- Candidate generation is treated as a **distinct system layer**
- Ranking and learning-to-rank are intentionally deferred until candidate quality is validated

### 2.2 Reproducibility Over Convenience
- Large datasets are excluded from version control
- All processed data and artifacts are **fully regenerable via scripts**

### 2.3 Multi-Signal Redundancy
- Multiple weak-to-moderate signals are preferred over a single strong signal
- Agreement across signals is treated as an implicit confidence measure

### 2.4 Offline Validation First
- Candidate quality is validated before any model training
- Emphasis on coverage, diversity, and recall-based diagnostics

---

## 📁 Repository Structure

BASKET_AI/
├── data/
│   ├── external/                 # Raw external datasets
│   ├── generated/                # Derived graphs & embeddings
│   ├── processed/                # Final parquet tables
│
├── notebooks/
│   ├── 01_eda_baseline.ipynb     # Data understanding & design decisions
│   ├── 02_models_reco_candidates.ipynb
│
├── src/
│   ├── data_generation/          # Graphs, embeddings, synthetic users
│   ├── data_processing/          # Basket & transaction builders
│
└── README.md


---

## 3. Data Pipeline Overview

### 3.1 Raw to Processed Baskets

Raw transactional data is transformed into two core parquet tables:

- **baskets.parquet**
  - One row per basket
  - Basket-level metadata

- **basket_items.parquet**
  - One row per `(basket, item)`
  - Canonical basket–item relationship table

These tables are rebuilt locally using:

`src/data_processing/build_baskets_tables.py`

This design ensures that:
- the repository remains lightweight
- data lineage is explicit
- results are reproducible end-to-end

---

## 4. Candidate Generation Signals

Each signal produces candidates **independently**, without knowledge of other signals.

### 4.1 Association Rules (Pair-Based Apriori-lite)

- Basket-level item co-occurrence is converted into pairwise rules
- Both directions (A → B and B → A) are evaluated
- Rules are filtered using:
  - minimum pair frequency
  - confidence threshold
  - lift threshold

This signal captures **strong conditional relationships** between items.

---

### 4.2 Basket Co-occurrence Frequency

- Counts how frequently item pairs appear in the same basket
- Acts as a high-recall, popularity-weighted signal
- Particularly effective for common and medium-frequency items

---

### 4.3 Category-Level Generalization

- Items in the basket are mapped to their categories
- Popular items from those categories are surfaced
- Provides **fallback coverage** when item-level signals are sparse

---

### 4.4 Item Embedding Neighbors (Item2Vec-style)

- Item-to-item embeddings learned from basket sequences
- Nearest neighbors are used as candidates
- Captures **latent semantic similarity** beyond direct co-occurrence

---

## 5. Multi-Signal Blending Strategy

Candidates from all sources are unioned into a single pool.

For each candidate, the system tracks:
- aggregated blended score
- number of supporting signals
- exact source combination

This enables:
- signal ablation analysis
- multi-source reinforcement
- robustness diagnostics

---

## 6. Offline Evaluation Methodology

Because this system focuses on **candidate generation**, evaluation relies on
industry-standard offline diagnostics rather than regression metrics.

### 6.1 Robustness Diagnostics

Across random baskets:
- candidate count distribution
- empty candidate pool rate
- average and maximum number of sources per candidate
- sensitivity to basket size

### 6.2 Hold-Out Recall@K

- One item is removed from each basket
- Candidates are generated from remaining context
- A hit is recorded if the held-out item appears in the top-K candidates

This directly measures **candidate coverage quality**.

### 6.3 Signal Ablation

Recall@K is computed separately for:
- association rules only
- co-occurrence only
- category only
- embedding only
- blended multi-signal pool

This reveals each signal’s marginal contribution.

---

## 7. Key Findings

- Candidate pools are **non-empty for the vast majority of baskets**
- Multi-signal blending consistently improves recall over single signals
- Co-occurrence and rules provide strong baseline coverage
- Embeddings contribute semantic diversity
- Category-level candidates reduce cold-start sparsity
- Basket size correlates moderately with candidate richness

These results validate the pipeline as a **reliable foundation for ranking models**.

---

## 8. Why Ranking Is Intentionally Deferred

This project intentionally stops **before full ranking optimization**.

The objective is to ensure that:
- relevant items are surfaced early
- candidate generation is stable and explainable
- downstream models are not bottlenecked by poor recall

Ranking models can be swapped, retrained, or personalized independently once
candidate quality is guaranteed.

---

## 9. Future Extensions

This pipeline is fully compatible with:

- Learning-to-Rank (XGBoost / LightGBM)
- Feature store integration
- Online inference simulation
- FastAPI-based model serving
- Dashboarding (Looker / Streamlit)

---

## 10. References

- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
- Sarwar et al. (2001). *Item-Based Collaborative Filtering Recommendation Algorithms*.
- Mikolov et al. (2013). *Distributed Representations of Words and Phrases*.
- Covington et al. (2016). *Deep Neural Networks for YouTube Recommendations*.
- Gomez-Uribe & Hunt (2016). *The Netflix Recommender System*.
- Google RecSys Practices: Candidate Generation & Ranking Architecture.
- Industry patterns from Amazon, Netflix, and YouTube recommender pipelines.

---

## Author

**Gizem Totkanlı**  
Data Scientist — ML / DL / AI  

Portfolio Project:  
**Multi-Signal Recommendation Candidate Generation Pipeline**


