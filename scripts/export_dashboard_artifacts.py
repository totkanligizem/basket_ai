from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export dashboard-friendly CSV artifacts from canonical parquet sources."
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
        "--rules-path",
        type=Path,
        default=Path("data/processed/baskets/rules.parquet"),
        help="Input rules parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dash_app/data"),
        help="Dashboard data output directory.",
    )
    return parser.parse_args()


def export_eda_timeseries(baskets: pd.DataFrame, out_dir: Path) -> Path:
    df = baskets.copy()
    df["basket_date"] = pd.to_datetime(df["basket_date"], errors="coerce")
    df = df.dropna(subset=["basket_date"])
    df["date"] = df["basket_date"].dt.date

    rows = (
        df.groupby("date", as_index=False)
        .agg(
            baskets=("basket_id", "nunique"),
            distinct_customers=("customer_id", "nunique"),
            revenue=("total_amount", "sum"),
        )
        .sort_values("date")
    )
    rows["aov"] = rows["revenue"] / rows["baskets"].clip(lower=1)

    out_path = out_dir / "eda_timeseries.csv"
    rows.to_csv(out_path, index=False)
    return out_path


def export_top_categories(basket_items: pd.DataFrame, out_dir: Path) -> Path:
    df = basket_items.copy()
    cat_col = "CATEGORY_NAME1" if "CATEGORY_NAME1" in df.columns else "CATEGORY_NAME2"
    if cat_col not in df.columns:
        raise ValueError("Category columns are missing. Expected CATEGORY_NAME1 or CATEGORY_NAME2.")

    df[cat_col] = df[cat_col].fillna("__UNKNOWN__")
    df["item_total"] = pd.to_numeric(df.get("item_total", 0), errors="coerce").fillna(0.0)

    out = (
        df.groupby(cat_col, as_index=False)
        .agg(orders=("basket_id", "nunique"), revenue=("item_total", "sum"))
        .rename(columns={cat_col: "category"})
        .sort_values("orders", ascending=False)
        .head(20)
    )

    out_path = out_dir / "top_categories.csv"
    out.to_csv(out_path, index=False)
    return out_path


def export_rules_and_pairs(rules: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    rules_df = rules.copy()
    keep_rules = [c for c in ["antecedent", "consequent", "support", "confidence", "lift"] if c in rules_df.columns]
    if not {"antecedent", "consequent"}.issubset(keep_rules):
        raise ValueError("Rules input must contain antecedent and consequent columns.")

    sort_cols = [c for c in ["lift", "confidence", "support"] if c in rules_df.columns]
    rules_top = rules_df.sort_values(sort_cols, ascending=False).head(30)[keep_rules]
    rules_path = out_dir / "rules_top.csv"
    rules_top.to_csv(rules_path, index=False)

    pairs = rules_df.rename(columns={"antecedent": "item_a", "consequent": "item_b"}).copy()
    pair_cols = [c for c in ["item_a", "item_b", "pair_count", "lift", "confidence", "support"] if c in pairs.columns]
    sort_pair_cols = [c for c in ["pair_count", "lift", "confidence", "support"] if c in pairs.columns]
    pairs_top = pairs.sort_values(sort_pair_cols, ascending=False).head(30)[pair_cols]
    pairs_path = out_dir / "cooc_top_pairs.csv"
    pairs_top.to_csv(pairs_path, index=False)

    return rules_path, pairs_path


def main() -> None:
    args = parse_args()

    if not args.baskets_path.exists():
        raise FileNotFoundError(f"Missing baskets file: {args.baskets_path}")
    if not args.basket_items_path.exists():
        raise FileNotFoundError(f"Missing basket_items file: {args.basket_items_path}")
    if not args.rules_path.exists():
        raise FileNotFoundError(f"Missing rules file: {args.rules_path}")

    baskets = pd.read_parquet(args.baskets_path)
    basket_items = pd.read_parquet(args.basket_items_path)
    rules = pd.read_parquet(args.rules_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    eda_path = export_eda_timeseries(baskets=baskets, out_dir=args.out_dir)
    top_cat_path = export_top_categories(basket_items=basket_items, out_dir=args.out_dir)
    rules_path, pairs_path = export_rules_and_pairs(rules=rules, out_dir=args.out_dir)

    print(f"[done] wrote {eda_path}")
    print(f"[done] wrote {top_cat_path}")
    print(f"[done] wrote {rules_path}")
    print(f"[done] wrote {pairs_path}")


if __name__ == "__main__":
    main()
