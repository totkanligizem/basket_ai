from __future__ import annotations

from pathlib import Path
import pandas as pd

MARKETSALES_PATH = Path("data/processed/transactions/marketsales.parquet")
OUT_DIR = Path("data/generated/category_trees")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "marketsales_category_tree.csv"

def norm(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    return s if s else None

def main():
    cols = ["CATEGORY_NAME1", "CATEGORY_NAME2", "CATEGORY_NAME3"]
    df = pd.read_parquet(MARKETSALES_PATH, columns=cols).copy()

    for c in cols:
        df[c] = df[c].map(norm)

    # unique category triples
    tree = df.drop_duplicates().sort_values(cols)

    # ek: parent-child edge list de üretelim (graph için altın değerinde)
    edges = []

    for _, r in tree.iterrows():
        l1, l2, l3 = r["CATEGORY_NAME1"], r["CATEGORY_NAME2"], r["CATEGORY_NAME3"]
        if l1 and l2:
            edges.append({"parent": l1, "child": l2, "level_parent": 1, "level_child": 2})
        if l2 and l3:
            edges.append({"parent": l2, "child": l3, "level_parent": 2, "level_child": 3})

    edges_df = pd.DataFrame(edges).drop_duplicates().sort_values(["level_parent","parent","child"])
    edges_path = OUT_DIR / "marketsales_category_edges.csv"

    tree.to_csv(OUT_PATH, index=False)
    edges_df.to_csv(edges_path, index=False)

    print(f"[done] wrote: {OUT_PATH} rows={len(tree)}")
    print(f"[done] wrote: {edges_path} rows={len(edges_df)}")

if __name__ == "__main__":
    main()
