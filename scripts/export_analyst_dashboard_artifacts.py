from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export analyst-focused dashboard artifacts from canonical basket data."
    )
    parser.add_argument(
        "--baskets-path",
        type=Path,
        default=Path("data/processed/baskets/baskets.parquet"),
        help="Input baskets parquet.",
    )
    parser.add_argument(
        "--basket-items-path",
        type=Path,
        default=Path("data/processed/baskets/basket_items.parquet"),
        help="Input basket_items parquet.",
    )
    parser.add_argument(
        "--transactions-path",
        type=Path,
        default=Path("data/processed/transactions/marketsales.parquet"),
        help="Optional transactions parquet for item names and data quality rates.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dash_app/data/analyst"),
        help="Analyst dashboard output directory.",
    )
    parser.add_argument(
        "--top-items",
        type=int,
        default=80,
        help="How many top items to export in item summary.",
    )
    return parser.parse_args()


def _normalize_text(series: pd.Series, default: str = "__UNKNOWN__") -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    out = out.mask(out.eq(""), default)
    return out


def _normalize_itemcode(series: pd.Series) -> pd.Series:
    s = _normalize_text(series, default="__UNKNOWN__")
    # Remove trailing .0 introduced by float casting in some parquet readers.
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def _find_category_col(df: pd.DataFrame) -> str:
    for col in ["CATEGORY_NAME1", "CATEGORY_NAME2", "category"]:
        if col in df.columns:
            return col
    raise ValueError("Category column is missing. Expected CATEGORY_NAME1 or CATEGORY_NAME2.")


def _find_city_col(df: pd.DataFrame) -> str:
    for col in ["city", "CITY", "region", "REGION"]:
        if col in df.columns:
            return col
    raise ValueError("City/region column is missing. Expected city/CITY/region/REGION.")


def export_daily_metrics(baskets: pd.DataFrame, out_dir: Path) -> Path:
    df = baskets.copy()
    df["basket_date"] = pd.to_datetime(df.get("basket_date"), errors="coerce")
    df = df.dropna(subset=["basket_date"])
    df["date"] = df["basket_date"].dt.date

    for col in ["total_amount", "total_items", "distinct_items", "category_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out = (
        df.groupby("date", as_index=False)
        .agg(
            baskets=("basket_id", "nunique"),
            customers=("customer_id", "nunique"),
            revenue=("total_amount", "sum"),
            total_items=("total_items", "sum"),
            avg_distinct_items=("distinct_items", "mean"),
            avg_category_count=("category_count", "mean"),
        )
        .sort_values("date")
    )
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)
    out["avg_items_per_basket"] = out["total_items"] / out["baskets"].clip(lower=1)

    out_path = out_dir / "daily_metrics.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_category_daily_metrics(basket_items: pd.DataFrame, out_dir: Path) -> Path:
    df = basket_items.copy()
    df["basket_date"] = pd.to_datetime(df.get("basket_date"), errors="coerce")
    df = df.dropna(subset=["basket_date"])
    df["date"] = df["basket_date"].dt.date

    cat_col = _find_category_col(df)
    df["category"] = _normalize_text(df[cat_col])
    df["item_total"] = pd.to_numeric(df.get("item_total", 0.0), errors="coerce").fillna(0.0)
    df["AMOUNT"] = pd.to_numeric(df.get("AMOUNT", 0.0), errors="coerce").fillna(0.0)

    out = (
        df.groupby(["date", "category"], as_index=False)
        .agg(
            revenue=("item_total", "sum"),
            baskets=("basket_id", "nunique"),
            quantity=("AMOUNT", "sum"),
            line_count=("basket_id", "size"),
        )
        .sort_values(["date", "revenue"], ascending=[True, False])
    )
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)
    out["items_per_basket"] = out["quantity"] / out["baskets"].clip(lower=1)

    out_path = out_dir / "category_daily_metrics.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_category_city_daily_metrics(basket_items: pd.DataFrame, out_dir: Path) -> Path:
    df = basket_items.copy()
    df["basket_date"] = pd.to_datetime(df.get("basket_date"), errors="coerce")
    df = df.dropna(subset=["basket_date"])
    df["date"] = df["basket_date"].dt.date

    cat_col = _find_category_col(df)
    city_col = _find_city_col(df)
    df["category"] = _normalize_text(df[cat_col])
    df["city"] = _normalize_text(df[city_col])
    df["item_total"] = pd.to_numeric(df.get("item_total", 0.0), errors="coerce").fillna(0.0)
    df["AMOUNT"] = pd.to_numeric(df.get("AMOUNT", 0.0), errors="coerce").fillna(0.0)

    out = (
        df.groupby(["date", "city", "category"], as_index=False)
        .agg(
            revenue=("item_total", "sum"),
            baskets=("basket_id", "nunique"),
            quantity=("AMOUNT", "sum"),
            line_count=("basket_id", "size"),
        )
        .sort_values(["date", "revenue"], ascending=[True, False])
    )
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)
    out["items_per_basket"] = out["quantity"] / out["baskets"].clip(lower=1)

    out_path = out_dir / "category_city_daily_metrics.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_city_daily_metrics(baskets: pd.DataFrame, out_dir: Path) -> Path:
    df = baskets.copy()
    df["basket_date"] = pd.to_datetime(df.get("basket_date"), errors="coerce")
    df = df.dropna(subset=["basket_date"])
    df["date"] = df["basket_date"].dt.date

    city_col = _find_city_col(df)
    df["city"] = _normalize_text(df[city_col])
    df["total_amount"] = pd.to_numeric(df.get("total_amount", 0.0), errors="coerce").fillna(0.0)

    out = (
        df.groupby(["date", "city"], as_index=False)
        .agg(
            baskets=("basket_id", "nunique"),
            customers=("customer_id", "nunique"),
            revenue=("total_amount", "sum"),
        )
        .sort_values(["date", "revenue"], ascending=[True, False])
    )
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)

    out_path = out_dir / "city_daily_metrics.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_basket_scope(baskets: pd.DataFrame, out_dir: Path) -> Path:
    df = baskets.copy()
    df["basket_date"] = pd.to_datetime(df.get("basket_date"), errors="coerce")
    df = df.dropna(subset=["basket_date"])
    df["date"] = df["basket_date"].dt.date

    city_col = _find_city_col(df)
    df["city"] = _normalize_text(df[city_col])
    df["revenue"] = pd.to_numeric(df.get("total_amount", 0.0), errors="coerce").fillna(0.0)
    df["total_items"] = pd.to_numeric(df.get("total_items", 0.0), errors="coerce").fillna(0.0)
    df["distinct_items"] = pd.to_numeric(df.get("distinct_items", 0.0), errors="coerce").fillna(0.0)
    df["customer_id"] = _normalize_text(df.get("customer_id"), default="__UNKNOWN__")

    keep_cols = ["basket_id", "date", "city", "customer_id", "revenue", "total_items", "distinct_items"]
    out = df[keep_cols].drop_duplicates(subset=["basket_id"])
    out_path = out_dir / "basket_scope.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_basket_category_bridge(basket_items: pd.DataFrame, out_dir: Path) -> Path:
    df = basket_items.copy()
    df["basket_date"] = pd.to_datetime(df.get("basket_date"), errors="coerce")
    df = df.dropna(subset=["basket_date"])
    df["date"] = df["basket_date"].dt.date

    cat_col = _find_category_col(df)
    city_col = _find_city_col(df)
    df["category"] = _normalize_text(df[cat_col])
    df["city"] = _normalize_text(df[city_col])

    out = df[["basket_id", "date", "city", "category"]].drop_duplicates()
    out_path = out_dir / "basket_category_bridge.csv"
    out.to_csv(out_path, index=False)
    return out_path


def _item_name_lookup(transactions: pd.DataFrame | None) -> pd.Series | None:
    if transactions is None or not {"ITEMCODE", "ITEMNAME"}.issubset(transactions.columns):
        return None

    ref = transactions[["ITEMCODE", "ITEMNAME"]].copy()
    ref["ITEMCODE"] = _normalize_itemcode(ref["ITEMCODE"])
    ref["ITEMNAME"] = _normalize_text(ref["ITEMNAME"], default="")
    ref = ref[(ref["ITEMCODE"] != "__UNKNOWN__") & (ref["ITEMNAME"] != "")]
    if ref.empty:
        return None

    name_map = (
        ref.groupby("ITEMCODE")["ITEMNAME"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .rename("item_name")
    )
    return name_map


def export_top_items(
    basket_items: pd.DataFrame,
    transactions: pd.DataFrame | None,
    out_dir: Path,
    top_n: int,
) -> Path:
    df = basket_items.copy()
    item_col = "itemcode" if "itemcode" in df.columns else "ITEMCODE"
    if item_col not in df.columns:
        raise ValueError("Item code column missing. Expected itemcode or ITEMCODE.")

    cat_col = _find_category_col(df)
    df["itemcode"] = _normalize_itemcode(df[item_col])
    df["category"] = _normalize_text(df[cat_col])
    df["item_total"] = pd.to_numeric(df.get("item_total", 0.0), errors="coerce").fillna(0.0)
    df["AMOUNT"] = pd.to_numeric(df.get("AMOUNT", 0.0), errors="coerce").fillna(0.0)

    out = (
        df.groupby(["itemcode", "category"], as_index=False)
        .agg(
            revenue=("item_total", "sum"),
            baskets=("basket_id", "nunique"),
            quantity=("AMOUNT", "sum"),
            line_count=("basket_id", "size"),
        )
        .sort_values("revenue", ascending=False)
    )
    out["avg_unit_price"] = out["revenue"] / out["quantity"].where(out["quantity"] > 0)

    name_map = _item_name_lookup(transactions)
    if name_map is not None:
        out = out.join(name_map, on="itemcode")
    else:
        out["item_name"] = ""

    keep_cols = [
        "itemcode",
        "item_name",
        "category",
        "revenue",
        "baskets",
        "quantity",
        "avg_unit_price",
        "line_count",
    ]
    out = out[keep_cols].head(max(top_n, 1))

    out_path = out_dir / "top_items.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_top_items_daily(
    basket_items: pd.DataFrame,
    transactions: pd.DataFrame | None,
    out_dir: Path,
) -> Path:
    df = basket_items.copy()
    df["basket_date"] = pd.to_datetime(df.get("basket_date"), errors="coerce")
    df = df.dropna(subset=["basket_date"])
    df["date"] = df["basket_date"].dt.date

    item_col = "itemcode" if "itemcode" in df.columns else "ITEMCODE"
    if item_col not in df.columns:
        raise ValueError("Item code column missing. Expected itemcode or ITEMCODE.")

    cat_col = _find_category_col(df)
    df["itemcode"] = _normalize_itemcode(df[item_col])
    df["category"] = _normalize_text(df[cat_col])
    df["item_total"] = pd.to_numeric(df.get("item_total", 0.0), errors="coerce").fillna(0.0)
    df["AMOUNT"] = pd.to_numeric(df.get("AMOUNT", 0.0), errors="coerce").fillna(0.0)

    out = (
        df.groupby(["date", "itemcode", "category"], as_index=False)
        .agg(
            revenue=("item_total", "sum"),
            baskets=("basket_id", "nunique"),
            quantity=("AMOUNT", "sum"),
            line_count=("basket_id", "size"),
        )
        .sort_values(["date", "revenue"], ascending=[True, False])
    )
    out["avg_unit_price"] = out["revenue"] / out["quantity"].where(out["quantity"] > 0)

    name_map = _item_name_lookup(transactions)
    if name_map is not None:
        out = out.join(name_map, on="itemcode")
    else:
        out["item_name"] = ""

    keep_cols = [
        "date",
        "itemcode",
        "item_name",
        "category",
        "revenue",
        "baskets",
        "quantity",
        "avg_unit_price",
        "line_count",
    ]
    out = out[keep_cols]

    out_path = out_dir / "top_items_daily.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_quality_daily(
    basket_items: pd.DataFrame,
    transactions: pd.DataFrame | None,
    out_dir: Path,
) -> Path:
    if transactions is not None and "DATE_" in transactions.columns:
        src = transactions.copy()
        src["date"] = pd.to_datetime(src["DATE_"], errors="coerce").dt.date
        item_col = "ITEMCODE" if "ITEMCODE" in src.columns else None
        cat_col = _find_category_col(src) if any(c in src.columns for c in ["CATEGORY_NAME1", "CATEGORY_NAME2"]) else None
        price_col = "PRICE" if "PRICE" in src.columns else None
        amount_col = "AMOUNT" if "AMOUNT" in src.columns else None
    else:
        src = basket_items.copy()
        src["date"] = pd.to_datetime(src.get("basket_date"), errors="coerce").dt.date
        item_col = "itemcode" if "itemcode" in src.columns else None
        cat_col = _find_category_col(src) if any(c in src.columns for c in ["CATEGORY_NAME1", "CATEGORY_NAME2", "category"]) else None
        price_col = "PRICE" if "PRICE" in src.columns else None
        amount_col = "AMOUNT" if "AMOUNT" in src.columns else None

    src = src.dropna(subset=["date"])
    if src.empty:
        raise ValueError("No valid dated rows for quality export.")

    if item_col is not None:
        item = src[item_col]
        src["missing_item_code"] = item.isna() | item.astype(str).str.strip().eq("")
    else:
        src["missing_item_code"] = False

    if cat_col is not None:
        category = _normalize_text(src[cat_col], default="__UNKNOWN__")
        src["missing_category"] = category.eq("__UNKNOWN__")
    else:
        src["missing_category"] = False

    if price_col is not None:
        price = pd.to_numeric(src[price_col], errors="coerce")
        src["missing_price"] = price.isna() | price.le(0)
    else:
        src["missing_price"] = False

    if amount_col is not None:
        amount = pd.to_numeric(src[amount_col], errors="coerce")
        src["missing_amount"] = amount.isna() | amount.le(0)
    else:
        src["missing_amount"] = False

    out = (
        src.groupby("date", as_index=False)
        .agg(
            missing_item_code_rate=("missing_item_code", "mean"),
            missing_category_rate=("missing_category", "mean"),
            missing_price_rate=("missing_price", "mean"),
            missing_amount_rate=("missing_amount", "mean"),
        )
        .sort_values("date")
    )
    out[[
        "missing_item_code_rate",
        "missing_category_rate",
        "missing_price_rate",
        "missing_amount_rate",
    ]] = out[[
        "missing_item_code_rate",
        "missing_category_rate",
        "missing_price_rate",
        "missing_amount_rate",
    ]].fillna(0.0)

    out_path = out_dir / "quality_daily.csv"
    out.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    args = parse_args()

    if not args.baskets_path.exists():
        raise FileNotFoundError(f"Missing baskets parquet: {args.baskets_path}")
    if not args.basket_items_path.exists():
        raise FileNotFoundError(f"Missing basket_items parquet: {args.basket_items_path}")

    baskets = pd.read_parquet(args.baskets_path)
    basket_items = pd.read_parquet(args.basket_items_path)
    transactions = pd.read_parquet(args.transactions_path) if args.transactions_path.exists() else None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    daily_path = export_daily_metrics(baskets=baskets, out_dir=args.out_dir)
    cat_daily_path = export_category_daily_metrics(basket_items=basket_items, out_dir=args.out_dir)
    cat_city_daily_path = export_category_city_daily_metrics(basket_items=basket_items, out_dir=args.out_dir)
    city_daily_path = export_city_daily_metrics(baskets=baskets, out_dir=args.out_dir)
    basket_scope_path = export_basket_scope(baskets=baskets, out_dir=args.out_dir)
    basket_category_bridge_path = export_basket_category_bridge(basket_items=basket_items, out_dir=args.out_dir)
    quality_path = export_quality_daily(
        basket_items=basket_items,
        transactions=transactions,
        out_dir=args.out_dir,
    )
    top_items_path = export_top_items(
        basket_items=basket_items,
        transactions=transactions,
        out_dir=args.out_dir,
        top_n=args.top_items,
    )
    top_items_daily_path = export_top_items_daily(
        basket_items=basket_items,
        transactions=transactions,
        out_dir=args.out_dir,
    )

    print(f"[done] wrote {daily_path}")
    print(f"[done] wrote {cat_daily_path}")
    print(f"[done] wrote {cat_city_daily_path}")
    print(f"[done] wrote {city_daily_path}")
    print(f"[done] wrote {basket_scope_path}")
    print(f"[done] wrote {basket_category_bridge_path}")
    print(f"[done] wrote {quality_path}")
    print(f"[done] wrote {top_items_path}")
    print(f"[done] wrote {top_items_daily_path}")


if __name__ == "__main__":
    main()
