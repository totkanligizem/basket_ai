from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

MARKETSALES_PATH = Path("data/processed/transactions/marketsales.parquet")
OUT_DIR = Path("data/generated/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_PATH = OUT_DIR / "product_embeddings.parquet"
NN_PATH = OUT_DIR / "product_neighbors_top20.csv"

RNG_SEED = 42

def main() -> None:
    df = pd.read_parquet(MARKETSALES_PATH, columns=["FICHENO", "ITEMCODE"]).dropna()
    df["ITEMCODE"] = df["ITEMCODE"].astype(int).astype(str)

    # Basket = sentence, item = token
    baskets = (
        df.groupby("FICHENO")["ITEMCODE"]
        .apply(lambda x: list(dict.fromkeys(x.tolist())))  # uniq + order
        .tolist()
    )
    baskets = [b for b in baskets if len(b) >= 2]
    print(f"[info] baskets used: {len(baskets)}")

    model = Word2Vec(
        sentences=baskets,
        vector_size=64,
        window=8,
        min_count=5,
        workers=4,
        sg=1,          # skip-gram
        negative=10,
        sample=1e-3,
        epochs=20,
        seed=RNG_SEED,
    )

    vocab = list(model.wv.index_to_key)
    vectors = np.vstack([model.wv[w] for w in vocab]).astype("float32")

    emb_df = pd.DataFrame(vectors)
    emb_df.insert(0, "itemcode", vocab)
    emb_df.to_parquet(EMB_PATH, index=False)
    print(f"[done] wrote embeddings: {EMB_PATH} rows={len(emb_df)} dims={vectors.shape[1]}")

    # Neighbor list (demo/dashboard için)
    rows = []
    for w in vocab[:500]:
        for neigh, score in model.wv.most_similar(w, topn=20):
            rows.append({"itemcode": w, "neighbor_itemcode": neigh, "similarity": float(score)})

    pd.DataFrame(rows).to_csv(NN_PATH, index=False)
    print(f"[done] wrote neighbors: {NN_PATH} rows={len(rows)}")

if __name__ == "__main__":
    main()
