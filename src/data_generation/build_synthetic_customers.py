from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

MARKETSALES_PATH = Path("data/processed/transactions/marketsales.parquet")
OUT_DIR = Path("data/generated/synthetic_customers")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "synthetic_customers.csv"

RNG_SEED = 42

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def assign_income_band(monthly_spend: float) -> str:
    # Harcama arttıkça gelir bandı artar (sentetik)
    if monthly_spend < 250:   return "low"
    if monthly_spend < 600:   return "lower_mid"
    if monthly_spend < 1200:  return "mid"
    if monthly_spend < 2500:  return "upper_mid"
    return "high"

def assign_age_band(freq_m: float, spend_m: float, rng: np.random.Generator) -> str:
    # Daha yüksek frekans + orta harcama genelde aile/çalışan; tamamen sentetik kural
    score = 0.55 * clip01(freq_m / 12) + 0.45 * clip01(spend_m / 1500)
    noise = rng.normal(0, 0.08)
    s = score + noise
    if s < 0.25: return "18-24"
    if s < 0.50: return "25-34"
    if s < 0.75: return "35-44"
    return "45+"

def assign_household_size(age_band: str, rng: np.random.Generator) -> int:
    # sentetik: yaş arttıkça hane büyüklüğü hafif artabilir
    base = {"18-24": 1.8, "25-34": 2.6, "35-44": 3.2, "45+": 2.8}[age_band]
    val = int(round(max(1, rng.normal(base, 0.7))))
    return min(val, 6)

def assign_persona(freq_m: float, spend_m: float, diversity: float) -> str:
    # Basit ama açıklanabilir persona: davranıştan
    if freq_m >= 8 and spend_m >= 1200:
        return "power_shopper"
    if freq_m >= 8 and spend_m < 1200:
        return "frequent_budget"
    if freq_m < 4 and spend_m >= 1200:
        return "bulk_buyer"
    if diversity >= 0.6:
        return "variety_seeker"
    return "routine_shopper"

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    cols = ["CLIENTCODE", "CLIENTNAME", "GENDER", "DATE_", "FICHENO", "LINENETTOTAL", "CATEGORY_NAME2"]
    df = pd.read_parquet(MARKETSALES_PATH, columns=cols).copy()

    df["DATE_"] = pd.to_datetime(df["DATE_"], errors="coerce")
    df = df.dropna(subset=["DATE_"])
    df["month"] = df["DATE_"].dt.to_period("M").astype(str)

    # müşteri bazında sepet metrikleri
    basket = (
        df.groupby(["CLIENTCODE", "month"])["FICHENO"]
          .nunique()
          .reset_index(name="baskets_in_month")
    )
    spend = (
        df.groupby(["CLIENTCODE", "month"])["LINENETTOTAL"]
          .sum()
          .reset_index(name="spend_in_month")
    )
    tmp = basket.merge(spend, on=["CLIENTCODE", "month"], how="left")

    # müşteri bazında ortalama aylık frekans/harcama
    agg = tmp.groupby("CLIENTCODE").agg(
        avg_baskets_per_month=("baskets_in_month", "mean"),
        avg_spend_per_month=("spend_in_month", "mean"),
        active_months=("month", "nunique"),
    ).reset_index()

    # kategori çeşitliliği (unique category2 / total category2)
    cat_div = (
        df.groupby("CLIENTCODE")["CATEGORY_NAME2"]
          .agg(total=("count"), uniq=("nunique"))
          .reset_index()
    )
    cat_div["category_diversity"] = (cat_div["uniq"] / cat_div["total"]).fillna(0.0)
    agg = agg.merge(cat_div[["CLIENTCODE", "category_diversity"]], on="CLIENTCODE", how="left")

    # gender: en sık görüleni al
    gender = (
        df.dropna(subset=["GENDER"])
          .groupby("CLIENTCODE")["GENDER"]
          .agg(lambda x: x.value_counts().index[0])
          .reset_index(name="gender")
    )
    names = (
        df.groupby("CLIENTCODE")[["CLIENTNAME"]]
          .agg(lambda x: x.dropna().iloc[0] if len(x.dropna()) else None)
          .reset_index()
    )

    out = agg.merge(gender, on="CLIENTCODE", how="left").merge(names, on="CLIENTCODE", how="left")

    # sentetik alanlar
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

    # etik etiket
    out["is_synthetic_profile"] = True
    out["synthetic_profile_note"] = "Derived from transactional behavior; demographics are synthetic for modeling/visualization."

    # düzen
    out = out.rename(columns={"CLIENTCODE": "customer_id", "CLIENTNAME": "customer_name"})
    out = out.sort_values("customer_id")

    out.to_csv(OUT_PATH, index=False)
    print(f"[done] wrote: {OUT_PATH} rows={len(out)} cols={out.shape[1]}")

if __name__ == "__main__":
    main()
