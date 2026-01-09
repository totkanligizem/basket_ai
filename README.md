# basket_ai — Multi-Signal Basket Recommendation System (Candidate Generation → Learning-to-Rank)

A production-style, end-to-end **basket-based recommendation pipeline** that covers the
full **pre-ranking → ranking → online inference simulation** flow.

The project is intentionally designed to reflect **real-world recommender system
architecture**, where performance is dominated not by a single model, but by the
**quality, diversity, and robustness of candidate generation**, followed by a
**lightweight but effective ranking stage**.

This repository demonstrates how multiple weak-to-moderate signals can be combined,
diagnosed, and ranked into a stable recommendation system.

---

## 🧠 System Architecture (High-Level)

Raw Data  
├── Transactions  
├── Product metadata  
├── Category hierarchy  
↓  
Feature & Graph Construction  
├── Basket co-occurrence graph  
├── Association rules  
├── Category tree  
├── Item embeddings (Item2Vec-style)  
↓  
Candidate Generation (Multi-Signal)  
├── Rules-based candidates  
├── Co-occurrence candidates  
├── Category expansion  
├── Embedding neighbors  
↓  
Candidate Pool Diagnostics  
├── Coverage & sparsity  
├── Source diversity  
├── Recall@K  
↓  
Learning-to-Rank (LightGBM LambdaRank)  
↓  
Offline Evaluation  
↓  
Online Inference Simulation (Mock Serving)

---

## 1. Motivation & Problem Framing

In large-scale recommender systems, overall performance is often constrained **before**
any sophisticated model is applied.

If relevant items are **not surfaced during candidate generation**, no downstream ranking
model can recover them.

This project focuses on the following core questions:

- Can we reliably generate **non-empty and sufficiently rich candidate pools**?
- Do heterogeneous recommendation signals reinforce each other?
- How stable is candidate generation across basket sizes and sparsity regimes?
- Does a simple learning-to-rank model improve over heuristic blending?
- Can the system be executed in an **online, low-latency serving setting**?

Rather than treating recommendation as a single black-box model, this project explicitly
models the **system layers** used in production recommender pipelines.

---

## 2. Design Principles

### 2.1 Separation of Concerns
- Candidate generation, ranking, and serving are treated as **independent layers**
- Each layer is validated before moving to the next

### 2.2 Reproducibility Over Convenience
- Large raw datasets are excluded from version control
- All processed tables and artifacts are **fully regenerable**
- Deterministic random seeds are used where applicable

### 2.3 Multi-Signal Redundancy
- No single signal is assumed to be sufficient
- Agreement across signals acts as an implicit confidence measure

### 2.4 Offline Validation First
- Candidate quality is evaluated **before** ranking
- Emphasis on recall, coverage, and robustness — not accuracy alone

---

## 📁 Repository Structure

basket_ai/  
├── data/  
│   ├── external/                 # Raw external datasets (not versioned)  
│   ├── generated/                # Graphs, embeddings, rule tables  
│   ├── processed/                # Final parquet tables  
│  
├── notebooks/  
│   ├── 01_eda_baseline.ipynb  
│   ├── 02_models_reco_candidates.ipynb  
│   ├── 03_models_ranking.ipynb  
│  
├── src/  
│   ├── data_generation/          # Rules, graphs, embeddings  
│   ├── data_processing/          # Basket & transaction builders  
│  
├── basket_ai_dbt/                # Analytics Engineering (dbt + BigQuery)  
│  
└── README.md

---

## 3. Notebook Overview

### 3.1 `01_eda_baseline.ipynb` — Data Understanding & Framing

- Exploratory analysis of baskets and item distributions
- Basket size and sparsity diagnostics
- Validation of data assumptions used later in modeling
- Establishes **design constraints** for candidate generation

---

### 3.2 `02_models_reco_candidates.ipynb` — Multi-Signal Candidate Generation

This notebook implements **independent candidate generators**, each producing candidates
without knowledge of the others.

#### Implemented Signals

1. **Association Rules**
   - Pair-based Apriori-style rules
   - Filtered by support, confidence, and lift
   - Captures strong conditional item relationships

2. **Basket Co-occurrence**
   - Frequency-based item–item co-occurrence
   - High-recall, popularity-weighted signal

3. **Category-Level Expansion**
   - Category hierarchy–driven fallback candidates
   - Reduces cold-start and sparsity issues

4. **Item Embedding Neighbors**
   - Item2Vec-style embeddings trained on basket sequences
   - Captures latent semantic similarity

#### Multi-Signal Blending

- Candidates are unioned into a single pool
- For each candidate, the system tracks:
  - blended score
  - number of supporting signals
  - exact source combination

This enables signal ablation and robustness analysis.

---

### 3.3 `03_models_ranking.ipynb` — Learning-to-Rank & Online Inference Simulation

This notebook completes the pipeline.

#### Learning-to-Rank (LightGBM LambdaRank)

- Training data constructed via **basket-level grouping**
- One held-out item per basket used as positive label
- Features used:
  - `blended_score`
  - `n_sources`
  - `basket_size`

The ranker learns to re-order candidates **within each basket context**.

#### Offline Ranking Evaluation

- NDCG@K and HitRate@K computed on validation baskets
- Comparison against baseline (heuristic blended score)
- Feature importance inspection

#### Online Inference Simulation

A production-style inference flow is simulated:

1. Candidate generation from live basket context
2. Feature construction (serving-safe)
3. Ranker scoring
4. Top-N recommendation output
5. Response serialization (JSON-compatible)

This validates:
- serving-time feature availability
- low-latency data flow
- stable system interface

---

## 4. Data Pipeline

Raw transactional data is transformed into two canonical tables:

- **`baskets.parquet`**
  - One row per basket
  - Basket-level metadata

- **`basket_items.parquet`**
  - One row per `(basket, item)`
  - Canonical basket–item relationship table

These tables are rebuilt using:

`src/data_processing/build_baskets_tables.py`

This keeps the repository lightweight while preserving full reproducibility.

---

## 5. Evaluation Methodology

### 5.1 Candidate-Level Diagnostics
- Candidate count distribution
- Empty candidate pool rate
- Source diversity per candidate
- Sensitivity to basket size

### 5.2 Hold-Out Recall@K
- One item removed from each basket
- Candidates generated from remaining context
- Hit recorded if held-out item appears in top-K

### 5.3 Ranking Metrics
- NDCG@K
- HitRate@K
- Baseline vs ranker comparison

---

## 6. Key Findings

- Candidate pools are **non-empty for nearly all baskets**
- Multi-signal blending outperforms any single signal
- Co-occurrence and rules provide strong recall
- Embeddings add semantic diversity
- Category expansion stabilizes sparse baskets
- LightGBM ranker improves ordering over heuristic blending
- Simple features already provide meaningful ranking gains

---

## 7. Why This Architecture Matters

This project mirrors how modern recommender systems are built in production:

- Candidate generation is **explicitly modeled and validated**
- Ranking is treated as a **refinement stage**, not a miracle fix
- Online constraints are considered from the start
- The system remains interpretable, debuggable, and extensible

---

## 8. Future Extensions

- Personalized ranking features
- User embeddings
- Feature store integration
- FastAPI deployment
- Real-time monitoring dashboards
- A/B testing framework

---

## 9. Analytics Engineering & BI Layer (dbt + BigQuery)

In addition to the recommendation and ranking pipeline, this project includes a
**production-grade analytics engineering layer** designed to support business
intelligence, monitoring, and decision-making use cases.

This layer transforms raw transactional data into **clean, tested, BI-ready marts**
using **BigQuery + dbt**.

### 9.1 Goals of the Analytics Layer

- Establish a **single source of truth** for KPIs
- Enable fast and reliable BI dashboards
- Make data quality issues explicit and measurable
- Separate analytics modeling from ML experimentation

---

### 9.2 Data Warehouse & Tooling

- **Data Warehouse:** Google BigQuery  
- **Transformation Framework:** dbt  
- **Modeling Style:** staging → intermediate → marts  
- **Testing:** dbt generic tests (`not_null`, `unique`)  
- **Version Control:** Git / GitHub  

---

### 9.3 dbt Model Architecture

#### Staging Layer
- `stg_baskets`
- `stg_basket_items`

Responsibilities:
- Type casting and normalization
- Explicit missing-value flags
- Canonical basket–item structure

---

#### Intermediate Layer
- `int_basket_summary`

Responsibilities:
- Basket-level aggregation
- Revenue, item counts, quality metrics
- Stable inputs for analytical marts

---

#### Mart Layer (BI-Ready Tables)

- **`mrt_daily_kpis`**
  - Daily basket count, customer count, revenue, AOV
  - Data quality indicators

- **`mrt_customer_summary`**
  - Customer frequency & monetary proxies
  - Recency signals and data quality flags

- **`mrt_top_items_daily`**
  - Daily Top-20 items by revenue
  - Ranking and coverage diagnostics

- **`mrt_category_daily_kpis`**
  - Daily KPIs by product category
  - Revenue, quantity, basket count

All marts are **fully tested and production-ready**.

---

### 9.4 Data Quality & Testing

- No unexpected nulls in critical dimensions
- Uniqueness enforced where required
- All dbt tests pass with zero failures

This ensures analytical reliability for downstream BI.

---

### 9.5 BI & Dashboard Readiness

Final marts are designed for direct consumption by **Looker Studio**:

- Clear grain and aggregation levels
- Stable date dimensions
- No hidden joins or implicit logic

Dashboard development is intentionally kept **outside the repository**, reflecting
real-world separation of concerns.

---

## 10. End-to-End Perspective

This repository intentionally combines:

- Recommendation system design
- Machine learning & ranking
- Analytics engineering
- BI-ready data modeling

Reflecting how modern data teams operate in production environments.

---

## Author

**Gizem Totkanlı**  
Data Scientist — Machine Learning / AI  

Portfolio Project:  
**Multi-Signal Basket Recommendation System**
