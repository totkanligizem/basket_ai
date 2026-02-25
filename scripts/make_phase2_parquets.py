from pathlib import Path
import pandas as pd

FILES = [
    # Phase 2 – Synthetic customers
    (
        "data/generated/synthetic_customers/synthetic_customers.csv",
        "data/generated/synthetic_customers/synthetic_customers.parquet",
    ),

    # Phase 2 – Google Trends
    (
        "data/generated/google_trends/trends_weekly.csv",
        "data/generated/google_trends/trends_weekly.parquet",
    ),

    # Phase 2 – Product graph / embeddings
    (
        "data/generated/embeddings/product_neighbors_top20.csv",
        "data/generated/embeddings/product_neighbors_top20.parquet",
    ),

    # Phase 2 – Category hierarchy
    (
        "data/generated/category_trees/marketsales_category_tree.csv",
        "data/generated/category_trees/category_tree.parquet",
    ),
    (
        "data/generated/category_trees/marketsales_category_edges.csv",
        "data/generated/category_trees/category_edges.parquet",
    ),
]


def main():
    for src, dst in FILES:
        src_p = Path(src)
        dst_p = Path(dst)

        if dst_p.exists():
            print(f"ℹ️ EXISTS: {dst_p} (skip)")
            continue

        if not src_p.exists():
            print(f"❌ MISSING: {src_p}")
            continue

        dst_p.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(src_p, encoding="utf-8")
        df.to_parquet(dst_p, index=False)

        print(f"✅ {src_p.name} -> {dst_p.name} | shape={df.shape}")


if __name__ == "__main__":
    main()
    
