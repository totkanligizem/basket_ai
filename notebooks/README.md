# Notebooks vs Production Workflow

Bu klasördeki notebooklar **exploration-only** kullanım içindir.

## Scope

- `01_eda_baseline.ipynb`: EDA ve veri davranışı keşfi
- `02_models_reco_candidates.ipynb`: aday üretim sinyallerini deneysel analiz
- `03_models_ranking.ipynb`: ranker yaklaşımı ve hata analizi

## Production Path (Recommended)

Notebook çıktılarından dashboard artifact üretmek yerine script tabanlı akışı kullan:

```bash
python src/data_processing/build_baskets_tables.py
python src/data_generation/build_synthetic_customers.py
python src/data_generation/build_product_embeddings_item2vec.py
python scripts/train_ranker_leakage_safe.py
python scripts/export_dashboard_artifacts.py
python dash_app/app.py
```

## Why

- Leakage-safe değerlendirme tek kanaldan garanti edilir.
- Dashboard artifact üretimi deterministic olur.
- Notebook hücre sırasına bağımlı hatalar azalır.
- CI ve otomasyon için daha stabil bir çalışma şekli sağlanır.
