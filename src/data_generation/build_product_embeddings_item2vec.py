from __future__ import annotations

import argparse
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Item2Vec-style product embeddings.")
    parser.add_argument("--source", type=Path, default=MARKETSALES_PATH, help="Input transactions parquet.")
    parser.add_argument("--emb-out", type=Path, default=EMB_PATH, help="Output embeddings parquet.")
    parser.add_argument("--neighbors-out", type=Path, default=NN_PATH, help="Output neighbors CSV.")
    parser.add_argument("--vector-size", type=int, default=64)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--sample", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--max-neighbors-per-item", type=int, default=20)
    parser.add_argument(
        "--max-anchor-items",
        type=int,
        default=0,
        help="0 means all items. Positive values cap neighbor generation anchor count.",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    return parser.parse_args()


def build_baskets(source: Path) -> list[list[str]]:
    df = pd.read_parquet(source, columns=["FICHENO", "ITEMCODE"]).dropna(subset=["FICHENO", "ITEMCODE"])
    df["ITEMCODE"] = pd.to_numeric(df["ITEMCODE"], errors="coerce")
    df = df.dropna(subset=["ITEMCODE"])
    df["ITEMCODE"] = df["ITEMCODE"].astype(int).astype(str)

    baskets = (
        df.groupby("FICHENO")["ITEMCODE"]
        .apply(lambda x: list(dict.fromkeys(x.tolist())))
        .tolist()
    )
    return [basket for basket in baskets if len(basket) >= 2]


def save_embeddings(model: Word2Vec, out_path: Path) -> pd.DataFrame:
    vocab = list(model.wv.index_to_key)
    vectors = np.vstack([model.wv[w] for w in vocab]).astype("float32")
    dim_cols = [f"emb_{i:03d}" for i in range(vectors.shape[1])]
    emb_df = pd.DataFrame(vectors, columns=dim_cols)
    emb_df.insert(0, "itemcode", vocab)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(out_path, index=False)
    return emb_df


def save_neighbors(
    model: Word2Vec,
    out_path: Path,
    max_neighbors_per_item: int,
    max_anchor_items: int,
) -> pd.DataFrame:
    vocab = list(model.wv.index_to_key)
    if not vocab or len(vocab) == 1:
        neighbors_df = pd.DataFrame(columns=["itemcode", "neighbor_itemcode", "similarity"])
        neighbors_df.to_csv(out_path, index=False)
        return neighbors_df

    anchor_items = vocab if max_anchor_items <= 0 else vocab[:max_anchor_items]
    topn = min(max_neighbors_per_item, len(vocab) - 1)

    rows: list[dict[str, object]] = []
    for item in anchor_items:
        for neigh, score in model.wv.most_similar(item, topn=topn):
            rows.append(
                {
                    "itemcode": item,
                    "neighbor_itemcode": neigh,
                    "similarity": float(score),
                }
            )

    neighbors_df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    neighbors_df.to_csv(out_path, index=False)
    return neighbors_df


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Missing source file: {args.source}")

    baskets = build_baskets(args.source)
    print(f"[info] baskets used: {len(baskets)}")
    if not baskets:
        raise RuntimeError("No baskets with at least 2 items were found.")

    model = Word2Vec(
        sentences=baskets,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=1,
        negative=args.negative,
        sample=args.sample,
        epochs=args.epochs,
        seed=args.seed,
    )

    emb_df = save_embeddings(model, args.emb_out)
    print(
        f"[done] wrote embeddings: {args.emb_out} rows={len(emb_df)} dims={len(emb_df.columns) - 1}"
    )

    neighbors_df = save_neighbors(
        model=model,
        out_path=args.neighbors_out,
        max_neighbors_per_item=args.max_neighbors_per_item,
        max_anchor_items=args.max_anchor_items,
    )
    print(f"[done] wrote neighbors: {args.neighbors_out} rows={len(neighbors_df)}")


if __name__ == "__main__":
    main()
