from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SRC = Path("data/processed/transactions/marketsales.parquet")
OUT_DIR = Path("data/processed/baskets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASKET_ITEMS_PATH = OUT_DIR / "basket_items.parquet"
BASKETS_PATH = OUT_DIR / "baskets.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build basket-level and basket-item tables from marketsales.")
    parser.add_argument("--source", type=Path, default=SRC, help="Input transactions parquet.")
    parser.add_argument("--basket-items-out", type=Path, default=BASKET_ITEMS_PATH, help="Output basket_items parquet.")
    parser.add_argument("--baskets-out", type=Path, default=BASKETS_PATH, help="Output baskets parquet.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Missing source file: {args.source}")

    use_cols = [
        "FICHENO",
        "CLIENTCODE",
        "DATE_",
        "ITEMCODE",
        "AMOUNT",
        "PRICE",
        "LINENETTOTAL",
        "CATEGORY_NAME1",
        "CATEGORY_NAME2",
        "CATEGORY_NAME3",
        "CITY",
        "REGION",
        "GENDER",
    ]

    df = pd.read_parquet(args.source, columns=use_cols)
    df["DATE_"] = pd.to_datetime(df["DATE_"], errors="coerce")
    df = df.dropna(subset=["FICHENO", "DATE_"])
    df["FICHENO"] = pd.to_numeric(df["FICHENO"], errors="coerce")
    df = df.dropna(subset=["FICHENO"])
    df["FICHENO"] = df["FICHENO"].astype("int64")

    # -------------------------
    # basket_items
    # -------------------------
    basket_items = (
        df.rename(
            columns={
                "FICHENO": "basket_id",
                "CLIENTCODE": "customer_id",
                "DATE_": "basket_date",
                "ITEMCODE": "itemcode",
                "LINENETTOTAL": "item_total",
            }
        )
        .sort_values(["basket_id", "itemcode"])
    )
    basket_items["customer_id"] = basket_items["customer_id"].astype("string")
    basket_items["basket_date"] = pd.to_datetime(basket_items["basket_date"], errors="coerce")
    basket_items["item_total"] = pd.to_numeric(basket_items["item_total"], errors="coerce").fillna(0.0)
    basket_items["AMOUNT"] = pd.to_numeric(basket_items["AMOUNT"], errors="coerce").fillna(0.0)
    basket_items["PRICE"] = pd.to_numeric(basket_items["PRICE"], errors="coerce").fillna(0.0)

    args.basket_items_out.parent.mkdir(parents=True, exist_ok=True)
    basket_items.to_parquet(args.basket_items_out, index=False)
    print(f"[done] basket_items -> {args.basket_items_out} rows={len(basket_items)}")

    # -------------------------
    # baskets (basket-level)
    # -------------------------
    baskets = (
        basket_items.groupby("basket_id")
        .agg(
            customer_id=("customer_id", "first"),
            basket_date=("basket_date", "first"),
            total_items=("AMOUNT", "sum"),
            distinct_items=("itemcode", "nunique"),
            total_amount=("item_total", "sum"),
            category_count=("CATEGORY_NAME2", "nunique"),
            city=("CITY", "first"),
            region=("REGION", "first"),
            gender=("GENDER", "first"),
        )
        .reset_index()
    )
    baskets["customer_id"] = baskets["customer_id"].astype("string")
    baskets["basket_date"] = pd.to_datetime(baskets["basket_date"], errors="coerce")
    baskets["total_amount"] = pd.to_numeric(baskets["total_amount"], errors="coerce").fillna(0.0)
    baskets["total_items"] = pd.to_numeric(baskets["total_items"], errors="coerce").fillna(0.0)
    baskets["distinct_items"] = pd.to_numeric(baskets["distinct_items"], errors="coerce").fillna(0).astype("int64")
    baskets["category_count"] = pd.to_numeric(baskets["category_count"], errors="coerce").fillna(0).astype("int64")

    args.baskets_out.parent.mkdir(parents=True, exist_ok=True)
    baskets.to_parquet(args.baskets_out, index=False)
    print(f"[done] baskets -> {args.baskets_out} rows={len(baskets)}")
    print(f"[info] unique baskets: {baskets['basket_id'].nunique()} | unique customers: {baskets['customer_id'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
