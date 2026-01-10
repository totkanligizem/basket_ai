# basket_ai — Multi-Signal Basket Recommendation System  
*(Candidate Generation → Learning-to-Rank)*

A production-style, end-to-end **basket-based recommendation pipeline** covering the full  
**pre-ranking → ranking → online inference simulation** lifecycle.

This project is intentionally designed to reflect **real-world recommender system architecture**,  
where system performance is driven not by a single sophisticated model, but by the **quality, diversity,  
and robustness of candidate generation**, followed by a **lightweight yet effective ranking stage**.

The repository demonstrates how multiple weak-to-moderate signals can be combined, diagnosed,  
and ranked into a stable, interpretable, and extensible recommendation system.

---

## System Architecture (High-Level)

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

In large-scale recommender systems, overall performance is often constrained **before** any  
advanced model is applied.

If relevant items are **not surfaced during candidate generation**, no downstream ranking model  
can recover them.

This project explicitly addresses the following questions:

- Can we reliably generate **non-empty and sufficiently rich candidate pools**?
- Do heterogeneous recommendation signals reinforce each other in practice?
- How stable is candidate generation across basket sizes and sparsity regimes?
- Does a simple learning-to-rank model outperform heuristic blending?
- Can the system operate under **online, low-latency serving constraints**?

Rather than treating recommendation as a single black-box model, this project models the  
**system layers** commonly used in production recommender pipelines.

---

## 2. Design Principles

### Separation of Concerns
- Candidate generation, ranking, and serving are treated as **independent layers**
- Each layer is validated before advancing to the next

### Reproducibility Over Convenience
- Large raw datasets are excluded from version control
- All processed tables and artifacts are **fully regenerable**
- Deterministic random seeds are used where applicable

### Multi-Signal Redundancy
- No single signal is assumed to be sufficient
- Agreement across signals acts as an implicit confidence signal

### Offline Validation First
- Candidate quality is evaluated **prior to ranking**
- Emphasis is placed on recall, coverage, and robustness — not accuracy alone

---

## 3. Repository Structure

basket_ai/
├── data/
│   ├── external/          # Raw external datasets (not versioned)
│   ├── generated/         # Rules, graphs, embeddings
│   └── processed/         # Final parquet tables
│
├── notebooks/
│   ├── 01_eda_baseline.ipynb
│   ├── 02_models_reco_candidates.ipynb
│   └── 03_models_ranking.ipynb
│
├── src/
│   ├── data_generation/  # Rules, graphs, embeddings
│   └── data_processing/  # Basket & transaction builders
│
├── basket_ai_dbt/         # Analytics Engineering (dbt + BigQuery)
├── dash_app/              # Dash-based model diagnostics UI
└── README.md

---

## 4. Notebook Overview

### 4.1 `01_eda_baseline.ipynb` — Data Understanding & Framing
- Exploratory analysis of basket and item distributions
- Basket size and sparsity diagnostics
- Validation of modeling assumptions
- Establishes design constraints for candidate generation

---

### 4.2 `02_models_reco_candidates.ipynb` — Multi-Signal Candidate Generation

Independent candidate generators are implemented without knowledge of each other.

**Implemented signals:**
- Association rules (Apriori-style; filtered by support, confidence, and lift)
- Basket co-occurrence (frequency-weighted, high-recall)
- Category-level expansion (hierarchy-driven fallback)
- Item embedding neighbors (Item2Vec-style)

Candidates are unioned into a single pool while tracking:
- blended score
- number of supporting signals
- exact source combinations

This enables robustness analysis and signal ablation.

---

### 4.3 `03_models_ranking.ipynb` — Learning-to-Rank & Online Simulation

**Learning-to-Rank**
- LightGBM LambdaRank
- Basket-level grouping
- One held-out item per basket used as the positive label
- Features: `blended_score`, `n_sources`, `basket_size`

**Evaluation**
- NDCG@K
- HitRate@K
- Baseline vs ranker comparison
- Feature importance inspection

**Online Inference Simulation**
1. Candidate generation from live basket context
2. Serving-safe feature construction
3. Ranker scoring
4. Top-N recommendation output
5. JSON-compatible response serialization

---

## 5. Data Sources & Dataset Documentation

### Core Data Sources
- **Kaggle public e-commerce datasets** for transactional and basket behavior
- **Synthetic data generation** for privacy-safe customer and behavioral enrichment
- **Domain research** reflecting Turkish e-commerce behavior
- **Design-level external signals** for demand and trend enrichment

> Platforms such as Trendyol, Migros, and Getir are **not accessed via API**,  
> but are used as **behavioral reference points** to guide realistic system design.

---

### Canonical Transaction Tables

**baskets.parquet**
- basket_id (STRING)
- customer_id (STRING)
- order_date (DATE)
- channel (STRING)
- basket_size (INT)

**basket_items.parquet**
- basket_id (STRING)
- item_id (STRING)
- quantity (INT)
- price (FLOAT)
- category (STRING)

These tables are rebuilt via:

src/data_processing/build_baskets_tables.py

---

### Synthetic Enrichment Tables

**synthetic_customers**
- customer_id
- age_group
- gender
- income_segment
- lifecycle_stage
- price_sensitivity

**synthetic_behavior_features**
- avg_basket_size
- avg_basket_value
- repeat_purchase_rate
- category_diversity_score

---

### Model & Evaluation Outputs
- ranking_metrics.csv (model, k, ndcg, hit_rate)
- feature_importance.csv (feature, importance)
- rules_top.csv
- cooc_top_pairs.csv

---

## 6. Analytics Engineering Layer (dbt + BigQuery)

The project includes a **production-grade analytics layer** supporting BI and monitoring use cases.

- **Modeling pattern:** staging → intermediate → marts  
- **Warehouse:** Google BigQuery  
- **Transformations:** dbt  
- **Testing:** `not_null`, `unique`  

### Staging
- stg_baskets
- stg_basket_items

### Intermediate
- int_basket_summary

### Marts (BI-ready)
- mrt_daily_kpis
- mrt_customer_summary
- mrt_top_items_daily
- mrt_category_daily_kpis

All marts are tested, documented, and Looker Studio–ready.

---

## 7. Evaluation Methodology

- Candidate pool coverage and sparsity analysis
- Recall@K using held-out basket items
- Ranking quality via NDCG@K and HitRate@K
- Baseline versus ranker comparison

---

## 8. Key Findings

- Candidate pools are non-empty for nearly all baskets
- Multi-signal blending outperforms single-signal approaches
- Co-occurrence and rules dominate recall
- Embeddings add semantic diversity
- Category expansion stabilizes sparse baskets
- LightGBM ranker improves ordering over heuristics
- Simple features already yield meaningful gains

---

## 9. Why This Architecture Matters

This project mirrors how modern recommender systems are built in production:
- Candidate generation is explicitly modeled and validated
- Ranking is treated as a refinement layer, not a miracle fix
- Online constraints are considered from the start
- The system remains interpretable, debuggable, and extensible

---

## 10. Future Extensions

- Personalized ranking features
- User embeddings
- Feature store integration
- FastAPI deployment
- Real-time monitoring dashboards
- A/B testing framework

---

## Author

**Gizem Totkanlı**  
Data Scientist — Machine Learning / AI  

Portfolio Project:  
**Multi-Signal Basket Recommendation System**



