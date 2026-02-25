from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

MARKETSALES_PATH = Path("data/processed/transactions/marketsales.parquet")
OUT_DIR = Path("data/generated/synthetic_customers")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV_PATH = OUT_DIR / "synthetic_customers.csv"
OUT_PARQUET_PATH = OUT_DIR / "synthetic_customers.parquet"

RNG_SEED = 42
ID_SALT = "basket_ai_synthetic_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build privacy-safe synthetic customer features from transaction history."
    )
    parser.add_argument("--source", type=Path, default=MARKETSALES_PATH, help="Input marketsales parquet path.")
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV_PATH, help="Output CSV path.")
    parser.add_argument(
        "--out-parquet",
        type=Path,
        default=OUT_PARQUET_PATH,
        help="Output parquet path.",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed for synthetic fields.")
    parser.add_argument(
        "--id-salt",
        type=str,
        default=ID_SALT,
        help="Salt used for deterministic customer ID anonymization.",
    )
    parser.add_argument(
        "--keep-raw-customer-id",
        action="store_true",
        help="Also keep raw customer code column (not recommended for privacy-safe exports).",
    )
    return parser.parse_args()


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def anonymize_customer_id(raw_id: str, salt: str) -> str:
    text = f"{salt}:{raw_id}".encode("utf-8")
    return hashlib.sha256(text).hexdigest()[:20]


def assign_income_band(monthly_spend: float) -> str:
    if monthly_spend < 250:
        return "low"
    if monthly_spend < 600:
        return "lower_mid"
    if monthly_spend < 1200:
        return "mid"
    if monthly_spend < 2500:
        return "upper_mid"
    return "high"


def assign_age_band(freq_m: float, spend_m: float, rng: np.random.Generator) -> str:
    score = 0.55 * clip01(freq_m / 12) + 0.45 * clip01(spend_m / 1500)
    s = score + rng.normal(0, 0.08)
    if s < 0.25:
        return "18-24"
    if s < 0.50:
        return "25-34"
    if s < 0.75:
        return "35-44"
    return "45+"


def assign_household_size(age_band: str, rng: np.random.Generator) -> int:
    base = {"18-24": 1.8, "25-34": 2.6, "35-44": 3.2, "45+": 2.8}[age_band]
    val = int(round(max(1, rng.normal(base, 0.7))))
    return min(val, 6)


def assign_persona(freq_m: float, spend_m: float, diversity: float) -> str:
    if freq_m >= 8 and spend_m >= 1200:
        return "power_shopper"
    if freq_m >= 8 and spend_m < 1200:
        return "frequent_budget"
    if freq_m < 4 and spend_m >= 1200:
        return "bulk_buyer"
    if diversity >= 0.6:
        return "variety_seeker"
    return "routine_shopper"


def build_synthetic_customers(
    source: Path,
    out_csv: Path,
    out_parquet: Path,
    seed: int,
    id_salt: str,
    keep_raw_customer_id: bool,
) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"Missing source file: {source}")

    rng = np.random.default_rng(seed)

    cols = ["CLIENTCODE", "GENDER", "DATE_", "FICHENO", "LINENETTOTAL", "CATEGORY_NAME2"]
    df = pd.read_parquet(source, columns=cols).copy()
    df["DATE_"] = pd.to_datetime(df["DATE_"], errors="coerce")
    df = df.dropna(subset=["CLIENTCODE", "DATE_", "FICHENO"])

    df["CLIENTCODE"] = df["CLIENTCODE"].astype(str).str.strip()
    df = df[df["CLIENTCODE"] != ""]
    df["month"] = df["DATE_"].dt.to_period("M").astype(str)

    basket = (
        df.groupby(["CLIENTCODE", "month"], as_index=False)["FICHENO"]
        .nunique()
        .rename(columns={"FICHENO": "baskets_in_month"})
    )
    spend = (
        df.groupby(["CLIENTCODE", "month"], as_index=False)["LINENETTOTAL"]
        .sum()
        .rename(columns={"LINENETTOTAL": "spend_in_month"})
    )
    monthly = basket.merge(spend, on=["CLIENTCODE", "month"], how="left")

    agg = (
        monthly.groupby("CLIENTCODE", as_index=False)
        .agg(
            avg_baskets_per_month=("baskets_in_month", "mean"),
            avg_spend_per_month=("spend_in_month", "mean"),
            active_months=("month", "nunique"),
        )
    )

    cat_div = (
        df.groupby("CLIENTCODE")["CATEGORY_NAME2"]
        .agg(total=("count"), uniq=("nunique"))
        .reset_index()
    )
    cat_div["category_diversity"] = (cat_div["uniq"] / cat_div["total"]).fillna(0.0)
    agg = agg.merge(cat_div[["CLIENTCODE", "category_diversity"]], on="CLIENTCODE", how="left")

    gender = (
        df.dropna(subset=["GENDER"])
        .groupby("CLIENTCODE")["GENDER"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index(name="gender")
    )
    out = agg.merge(gender, on="CLIENTCODE", how="left")

    out["income_band"] = out["avg_spend_per_month"].map(assign_income_band)
    out["age_band"] = [
        assign_age_band(f, s, rng)
        for f, s in zip(out["avg_baskets_per_month"].fillna(0), out["avg_spend_per_month"].fillna(0))
    ]
    out["household_size"] = [assign_household_size(a, rng) for a in out["age_band"]]
    out["persona"] = [
        assign_persona(f, s, d)
        for f, s, d in zip(
            out["avg_baskets_per_month"].fillna(0),
            out["avg_spend_per_month"].fillna(0),
            out["category_diversity"].fillna(0),
        )
    ]

    out["customer_id_raw"] = out["CLIENTCODE"]
    out["customer_id"] = out["customer_id_raw"].map(lambda x: anonymize_customer_id(str(x), id_salt))
    out["is_synthetic_profile"] = True
    out["synthetic_profile_note"] = (
        "Generated from transactional behavior. IDs are anonymized and demographic attributes are synthetic."
    )

    columns = [
        "customer_id",
        "avg_baskets_per_month",
        "avg_spend_per_month",
        "active_months",
        "category_diversity",
        "gender",
        "income_band",
        "age_band",
        "household_size",
        "persona",
        "is_synthetic_profile",
        "synthetic_profile_note",
    ]
    if keep_raw_customer_id:
        columns.append("customer_id_raw")

    out = out[columns].sort_values("customer_id").reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    out.to_parquet(out_parquet, index=False)

    return out


def main() -> None:
    args = parse_args()
    out = build_synthetic_customers(
        source=args.source,
        out_csv=args.out_csv,
        out_parquet=args.out_parquet,
        seed=args.seed,
        id_salt=args.id_salt,
        keep_raw_customer_id=args.keep_raw_customer_id,
    )
    print(f"[done] wrote: {args.out_csv} rows={len(out)} cols={out.shape[1]}")
    print(f"[done] wrote: {args.out_parquet}")


if __name__ == "__main__":
    main()
