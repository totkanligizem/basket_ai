from __future__ import annotations

from pathlib import Path
import pandas as pd

SRC = Path("data/processed/transactions/marketsales.parquet")
OUT_DIR = Path("data/processed/baskets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASKET_ITEMS_PATH = OUT_DIR / "basket_items.parquet"
BASKETS_PATH = OUT_DIR / "baskets.parquet"

def main() -> None:
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

    df = pd.read_parquet(SRC, columns=use_cols)
    df["DATE_"] = pd.to_datetime(df["DATE_"], errors="coerce")
    df = df.dropna(subset=["FICHENO", "DATE_"])

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

    basket_items.to_parquet(BASKET_ITEMS_PATH, index=False)
    print(f"[done] basket_items -> {BASKET_ITEMS_PATH} rows={len(basket_items)}")

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

    baskets.to_parquet(BASKETS_PATH, index=False)
    print(f"[done] baskets -> {BASKETS_PATH} rows={len(baskets)}")

if __name__ == "__main__":
    main()
