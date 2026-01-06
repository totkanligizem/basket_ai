from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

# ----------------------------
# Ayarlar
# ----------------------------
MARKETSALES_PATH = Path("data/processed/transactions/marketsales.parquet")
OUT_DIR = Path("data/generated/google_trends")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GEO = "TR"
HL = "tr-TR"
TZ = 180  # Istanbul UTC+3
SLEEP_SEC = 2.0  # Google rate-limit yememek için

# Keyword stratejisi:
# - CATEGORY_NAME2 ağırlıklı (daha stabil)
# - Top-N seç
TOP_N_CAT2 = 15

# Bazı "çok genel/işe yaramaz" kelimeler için filtre
STOPWORDS = {
    "DIGER", "DİGER", "DİĞER", "GENEL", "MALZEME", "ÇEŞİTLİ", "CESITLI",
    "ÜRÜNLER", "URUNLER", "ÜRÜN", "URUN"
}

def normalize_tr_keyword(text: str) -> str:
    """
    Google Trends için Türkçe keyword normalize eder.
    - Unicode birleşik karakterleri düzeltir (i̇ → i)
    - Boşlukları sadeleştirir
    """
    if not isinstance(text, str):
        return str(text)

    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text

def clean_kw(s: str) -> str:
    s = str(s).strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)

    if len(s) < 3:
        return ""
    if s.upper() in STOPWORDS:
        return ""
    return s.lower()

def main() -> None:
    if not MARKETSALES_PATH.exists():
        raise FileNotFoundError(f"Missing: {MARKETSALES_PATH}")

    df = pd.read_parquet(MARKETSALES_PATH, columns=["DATE_", "CATEGORY_NAME2"])
    df["DATE_"] = pd.to_datetime(df["DATE_"], errors="coerce")
    df = df.dropna(subset=["DATE_"])

    start = df["DATE_"].min().date()
    end = df["DATE_"].max().date()
    timeframe = f"{start} {end}"
    print(f"[info] timeframe: {timeframe}")

    # Top kategori2 çıkar
    cat2 = (
        df["CATEGORY_NAME2"]
        .dropna()
        .astype(str)
        .map(clean_kw)
    )
    cat2 = cat2[cat2 != ""]
    top_cat2 = cat2.value_counts().head(TOP_N_CAT2).index.tolist()

    # Keywords (normalize ederek)
    keywords = [normalize_tr_keyword(kw.strip()) for kw in top_cat2]
    print("[info] selected keywords:", keywords)

    pytrends = TrendReq(hl=HL, tz=TZ)

    rows = []
    for kw in keywords:
        kw = normalize_tr_keyword(kw)  # build_payload öncesi garanti
        try:
            pytrends.build_payload([kw], timeframe=timeframe, geo=GEO)
            tdf = pytrends.interest_over_time()

            if tdf is None or tdf.empty:
                print(f"[warn] empty trends for: {kw}")
                time.sleep(SLEEP_SEC)
                continue

            tdf = tdf.reset_index()

            # pytrends sütunu keyword adıyla gelir
            value_col = kw if kw in tdf.columns else None
            if value_col is None:
                # bazen farklı isimlendirme olabilir, ilk numeric kolonu bul
                num_cols = [c for c in tdf.columns if c not in ("date", "isPartial")]
                value_col = num_cols[0] if num_cols else None

            if value_col is None:
                print(f"[warn] no value column for: {kw}")
                time.sleep(SLEEP_SEC)
                continue

            tdf = tdf.rename(columns={value_col: "trend_index"})
            tdf["keyword"] = kw
            tdf["geo"] = GEO
            tdf["timeframe_start"] = str(start)
            tdf["timeframe_end"] = str(end)

            keep = ["date", "keyword", "trend_index", "isPartial", "geo", "timeframe_start", "timeframe_end"]
            rows.append(tdf[keep])

            print(f"[ok] fetched: {kw} -> {len(tdf)} rows")
            time.sleep(SLEEP_SEC)

        except Exception as e:
            print(f"[error] {kw}: {e}")
            time.sleep(SLEEP_SEC)

    if not rows:
        raise RuntimeError("No trends data fetched. (Possible rate limit or blocked requests)")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_DIR / "trends_weekly.csv", index=False)
    print(f"[done] wrote: {OUT_DIR / 'trends_weekly.csv'}  rows={len(out)}  keywords={out['keyword'].nunique()}")

if __name__ == "__main__":
    main()
