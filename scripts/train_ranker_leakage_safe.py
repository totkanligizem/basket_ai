from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a leakage-safe basket ranker, then export dashboard metrics."
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
        "--neighbors-path",
        type=Path,
        default=Path("data/generated/embeddings/product_neighbors_top20.csv"),
        help="Embedding neighbors CSV (optional signal).",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Time-based split ratio.")
    parser.add_argument("--train-holdout-size", type=int, default=1200)
    parser.add_argument("--valid-holdout-size", type=int, default=400)
    parser.add_argument("--top-k-candidates", type=int, default=50)
    parser.add_argument("--min-pair-count", type=int, default=10)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--min-lift", type=float, default=1.05)
    parser.add_argument("--max-items-per-basket", type=int, default=50)
    parser.add_argument("--max-rules-per-item", type=int, default=100)
    parser.add_argument("--max-cooc-per-item", type=int, default=100)
    parser.add_argument("--max-category-items", type=int, default=200)
    parser.add_argument("--max-embedding-neighbors", type=int, default=80)
    parser.add_argument("--disable-embedding", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("dash_app/data/metrics/ranking_metrics.csv"),
        help="Output ranking metrics CSV.",
    )
    parser.add_argument(
        "--feature-importance-out",
        type=Path,
        default=Path("dash_app/data/feature_importance.csv"),
        help="Output feature importance CSV.",
    )
    parser.add_argument(
        "--rules-top-out",
        type=Path,
        default=Path("dash_app/data/rules_top.csv"),
        help="Output top rules CSV.",
    )
    parser.add_argument(
        "--pairs-top-out",
        type=Path,
        default=Path("dash_app/data/cooc_top_pairs.csv"),
        help="Output top co-occurrence pairs CSV.",
    )
    parser.add_argument(
        "--valid-predictions-out",
        type=Path,
        default=Path("dash_app/data/metrics/valid_predictions.csv"),
        help="Output validation predictions CSV.",
    )
    return parser.parse_args()


def load_inputs(baskets_path: Path, basket_items_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not baskets_path.exists():
        raise FileNotFoundError(f"Missing baskets file: {baskets_path}")
    if not basket_items_path.exists():
        raise FileNotFoundError(f"Missing basket_items file: {basket_items_path}")

    baskets = pd.read_parquet(
        baskets_path,
        columns=["basket_id", "basket_date", "total_amount", "customer_id"],
    )
    basket_items = pd.read_parquet(
        basket_items_path,
        columns=[
            "basket_id",
            "basket_date",
            "itemcode",
            "item_total",
            "CATEGORY_NAME1",
            "CATEGORY_NAME2",
            "CATEGORY_NAME3",
        ],
    )

    baskets["basket_id"] = pd.to_numeric(baskets["basket_id"], errors="coerce")
    baskets["basket_date"] = pd.to_datetime(baskets["basket_date"], errors="coerce")
    baskets = baskets.dropna(subset=["basket_id", "basket_date"]).copy()
    baskets["basket_id"] = baskets["basket_id"].astype(int)

    basket_items["basket_id"] = pd.to_numeric(basket_items["basket_id"], errors="coerce")
    basket_items["itemcode"] = pd.to_numeric(basket_items["itemcode"], errors="coerce")
    basket_items["basket_date"] = pd.to_datetime(basket_items["basket_date"], errors="coerce")
    basket_items = basket_items.dropna(subset=["basket_id", "itemcode"]).copy()
    basket_items["basket_id"] = basket_items["basket_id"].astype(int)
    basket_items["item_id"] = basket_items["itemcode"].astype(int)
    basket_items["item_total"] = pd.to_numeric(basket_items["item_total"], errors="coerce").fillna(0.0)

    if basket_items["basket_date"].isna().all():
        date_map = baskets[["basket_id", "basket_date"]].drop_duplicates()
        basket_items = basket_items.drop(columns=["basket_date"]).merge(date_map, on="basket_id", how="left")

    return baskets, basket_items


def split_baskets_by_time(
    baskets: pd.DataFrame,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    dates = sorted(pd.to_datetime(baskets["basket_date"], errors="coerce").dropna().unique())
    if len(dates) < 2:
        ids = baskets["basket_id"].drop_duplicates().sample(frac=1.0, random_state=seed).to_numpy()
        cut = max(1, int(len(ids) * train_ratio))
        return ids[:cut], ids[cut:], pd.Timestamp.min

    cutoff_idx = max(0, min(len(dates) - 2, int(len(dates) * train_ratio) - 1))
    cutoff_date = pd.Timestamp(dates[cutoff_idx])

    train_ids = baskets.loc[baskets["basket_date"] <= cutoff_date, "basket_id"].drop_duplicates().to_numpy()
    valid_ids = baskets.loc[baskets["basket_date"] > cutoff_date, "basket_id"].drop_duplicates().to_numpy()

    if len(train_ids) == 0 or len(valid_ids) == 0:
        ids = baskets["basket_id"].drop_duplicates().sample(frac=1.0, random_state=seed).to_numpy()
        cut = max(1, int(len(ids) * train_ratio))
        train_ids, valid_ids = ids[:cut], ids[cut:]

    return train_ids, valid_ids, cutoff_date


def build_basket_to_items(items_df: pd.DataFrame, basket_ids: np.ndarray) -> pd.Series:
    subset = items_df[items_df["basket_id"].isin(set(map(int, basket_ids)))]
    basket_to_items = (
        subset.groupby("basket_id")["item_id"]
        .apply(lambda s: sorted(set(int(x) for x in s.tolist())))
    )
    return basket_to_items


def build_rules_and_cooc_indices(
    basket_to_items: pd.Series,
    min_pair_count: int,
    min_confidence: float,
    min_lift: float,
    max_items_per_basket: int,
    max_rules_per_item: int,
    max_cooc_per_item: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float]]]]:
    item_counter: Counter[int] = Counter()
    pair_counter: Counter[tuple[int, int]] = Counter()

    for items in basket_to_items.values:
        uniq = list(dict.fromkeys(int(x) for x in items))
        if not uniq:
            continue
        if len(uniq) > max_items_per_basket:
            uniq = uniq[:max_items_per_basket]

        item_counter.update(uniq)
        if len(uniq) < 2:
            continue
        for a, b in combinations(sorted(uniq), 2):
            pair_counter[(a, b)] += 1

    n_baskets = max(1, int(len(basket_to_items)))
    pair_rows = [
        {"item_a": a, "item_b": b, "pair_count": cnt}
        for (a, b), cnt in pair_counter.items()
        if cnt >= min_pair_count
    ]
    if not pair_rows:
        empty_rules = pd.DataFrame(
            columns=["antecedent", "consequent", "support", "confidence", "lift", "pair_count", "a_count", "b_count"]
        )
        empty_cooc = pd.DataFrame(columns=["item_a", "item_b", "pair_count", "cooc_score"])
        return empty_rules, empty_cooc, {}, {}

    pairs_df = pd.DataFrame(pair_rows)
    pairs_df["a_count"] = pairs_df["item_a"].map(item_counter).astype(float)
    pairs_df["b_count"] = pairs_df["item_b"].map(item_counter).astype(float)
    pairs_df["support_ab"] = pairs_df["pair_count"] / n_baskets
    pairs_df["support_a"] = pairs_df["a_count"] / n_baskets
    pairs_df["support_b"] = pairs_df["b_count"] / n_baskets

    rules_rows: list[dict[str, float | int]] = []
    for row in pairs_df.itertuples(index=False):
        conf_ab = float(row.pair_count / row.a_count) if row.a_count > 0 else 0.0
        conf_ba = float(row.pair_count / row.b_count) if row.b_count > 0 else 0.0
        lift_ab = float(conf_ab / row.support_b) if row.support_b > 0 else 0.0
        lift_ba = float(conf_ba / row.support_a) if row.support_a > 0 else 0.0

        rules_rows.append(
            {
                "antecedent": int(row.item_a),
                "consequent": int(row.item_b),
                "support": float(row.support_ab),
                "confidence": conf_ab,
                "lift": lift_ab,
                "pair_count": int(row.pair_count),
                "a_count": float(row.a_count),
                "b_count": float(row.b_count),
            }
        )
        rules_rows.append(
            {
                "antecedent": int(row.item_b),
                "consequent": int(row.item_a),
                "support": float(row.support_ab),
                "confidence": conf_ba,
                "lift": lift_ba,
                "pair_count": int(row.pair_count),
                "a_count": float(row.b_count),
                "b_count": float(row.a_count),
            }
        )

    rules_df = pd.DataFrame(rules_rows)
    rules_df = rules_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["confidence", "lift"])
    rules_df = rules_df[
        (rules_df["confidence"] >= min_confidence) & (rules_df["lift"] >= min_lift)
    ].copy()

    if rules_df.empty:
        rules_index: dict[int, list[tuple[int, float]]] = {}
    else:
        rules_df["lift_norm"] = rules_df["lift"].clip(lower=0) / (rules_df["lift"].clip(lower=0).max() + 1e-9)
        rules_df["rule_score"] = rules_df["confidence"] * 0.7 + rules_df["lift_norm"] * 0.3
        rules_df = rules_df.sort_values(["rule_score", "pair_count"], ascending=False)
        rules_index = {}
        for antecedent, group in rules_df.groupby("antecedent"):
            grp = group.head(max_rules_per_item)
            rules_index[int(antecedent)] = [
                (int(r.consequent), float(r.rule_score)) for r in grp.itertuples(index=False)
            ]

    cooc_dir_rows = []
    for row in pairs_df.itertuples(index=False):
        cooc_dir_rows.append({"item_a": int(row.item_a), "item_b": int(row.item_b), "pair_count": int(row.pair_count)})
        cooc_dir_rows.append({"item_a": int(row.item_b), "item_b": int(row.item_a), "pair_count": int(row.pair_count)})

    cooc_df = pd.DataFrame(cooc_dir_rows)
    cooc_df["cooc_score"] = cooc_df["pair_count"] / cooc_df.groupby("item_a")["pair_count"].transform("max").clip(lower=1)
    cooc_df = cooc_df.sort_values(["item_a", "cooc_score"], ascending=[True, False])

    cooc_index: dict[int, list[tuple[int, float]]] = {}
    for item_a, group in cooc_df.groupby("item_a"):
        grp = group.head(max_cooc_per_item)
        cooc_index[int(item_a)] = [(int(r.item_b), float(r.cooc_score)) for r in grp.itertuples(index=False)]

    return rules_df, cooc_df, rules_index, cooc_index


def build_category_indices(
    train_items: pd.DataFrame,
    max_category_items: int,
) -> tuple[dict[str, list[tuple[int, float]]], dict[int, str]]:
    cat_col = "CATEGORY_NAME2" if "CATEGORY_NAME2" in train_items.columns else "CATEGORY_NAME1"
    if cat_col not in train_items.columns:
        return {}, {}

    cat_items = train_items[["item_id", cat_col]].dropna().copy()
    cat_items[cat_col] = cat_items[cat_col].astype(str)

    item_to_cat = (
        cat_items.drop_duplicates(subset=["item_id"])
        .set_index("item_id")[cat_col]
        .to_dict()
    )
    category_counts = (
        cat_items.groupby([cat_col, "item_id"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["cnt"], ascending=False)
    )

    cat_index: dict[str, list[tuple[int, float]]] = {}
    for category, group in category_counts.groupby(cat_col):
        grp = group.head(max_category_items).copy()
        max_cnt = float(grp["cnt"].max())
        grp["score"] = grp["cnt"] / (max_cnt + 1e-9)
        cat_index[str(category)] = [(int(r.item_id), float(r.score)) for r in grp.itertuples(index=False)]

    return cat_index, {int(k): str(v) for k, v in item_to_cat.items()}


def infer_neighbor_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = set(df.columns)
    candidates = [
        ("itemcode", "neighbor_itemcode", "similarity"),
        ("item_id", "neighbor_id", "similarity"),
        ("src", "dst", "score"),
    ]
    for src, nbr, score in candidates:
        if {src, nbr, score}.issubset(cols):
            return src, nbr, score
    raise ValueError(f"Cannot infer neighbor columns from {list(df.columns)}")


def build_embedding_index(
    neighbors_path: Path,
    train_item_set: set[int],
    max_embedding_neighbors: int,
) -> dict[int, list[tuple[int, float]]]:
    if not neighbors_path.exists():
        return {}

    neighbors = pd.read_csv(neighbors_path)
    if neighbors.empty:
        return {}

    src_col, nbr_col, score_col = infer_neighbor_columns(neighbors)
    n = neighbors[[src_col, nbr_col, score_col]].copy()
    n[src_col] = pd.to_numeric(n[src_col], errors="coerce")
    n[nbr_col] = pd.to_numeric(n[nbr_col], errors="coerce")
    n[score_col] = pd.to_numeric(n[score_col], errors="coerce")
    n = n.dropna(subset=[src_col, nbr_col, score_col]).copy()
    n[src_col] = n[src_col].astype(int)
    n[nbr_col] = n[nbr_col].astype(int)

    n = n[n[src_col].isin(train_item_set)]
    if n.empty:
        return {}

    out: dict[int, list[tuple[int, float]]] = {}
    for src, group in n.groupby(src_col):
        grp = group.sort_values(score_col, ascending=False).head(max_embedding_neighbors)
        out[int(src)] = [
            (int(getattr(row, nbr_col)), float(getattr(row, score_col)))
            for row in grp.itertuples(index=False)
        ]
    return out


def make_holdout_table(
    basket_to_items: pd.Series,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    eligible = basket_to_items[basket_to_items.apply(lambda x: isinstance(x, list) and len(x) >= 2)]
    if eligible.empty:
        return pd.DataFrame(columns=["basket_id", "held_out", "context_items", "basket_size"])

    if sample_size > 0 and sample_size < len(eligible):
        sampled_ids = rng.choice(eligible.index.to_numpy(), size=sample_size, replace=False)
        eligible = eligible.loc[sampled_ids]

    rows = []
    for basket_id, items in eligible.items():
        item_list = list(dict.fromkeys(int(x) for x in items))
        held_out = int(rng.choice(item_list))
        context = [int(x) for x in item_list if int(x) != held_out]
        if not context:
            continue
        rows.append(
            {
                "basket_id": int(basket_id),
                "held_out": held_out,
                "context_items": context,
                "basket_size": len(item_list),
            }
        )
    return pd.DataFrame(rows)


def generate_candidates(
    context_items: list[int],
    top_k: int,
    rules_index: dict[int, list[tuple[int, float]]],
    cooc_index: dict[int, list[tuple[int, float]]],
    cat_index: dict[str, list[tuple[int, float]]],
    item_to_cat: dict[int, str],
    emb_index: dict[int, list[tuple[int, float]]],
) -> pd.DataFrame:
    if not context_items:
        return pd.DataFrame(columns=["candidate", "blended_score", "n_sources", "sources"])

    weights = {"rules": 1.0, "cooc": 0.8, "category": 0.45, "embedding": 0.75}
    ctx = [int(x) for x in context_items]
    ctx_set = set(ctx)

    scores: defaultdict[int, float] = defaultdict(float)
    sources: dict[int, set[str]] = defaultdict(set)

    def add_signal(candidate: int, score: float, source: str) -> None:
        if candidate in ctx_set:
            return
        scores[candidate] += float(score) * weights[source]
        sources[candidate].add(source)

    for item in ctx:
        for cand, score in rules_index.get(item, []):
            add_signal(cand, score, "rules")
        for cand, score in cooc_index.get(item, []):
            add_signal(cand, score, "cooc")
        for cand, score in emb_index.get(item, []):
            add_signal(cand, score, "embedding")

    ctx_cats = {item_to_cat.get(item) for item in ctx if item_to_cat.get(item)}
    for category in ctx_cats:
        for cand, score in cat_index.get(category, []):
            add_signal(cand, score, "category")

    if not scores:
        return pd.DataFrame(columns=["candidate", "blended_score", "n_sources", "sources"])

    cand_df = pd.DataFrame(
        {
            "candidate": list(scores.keys()),
            "blended_score": list(scores.values()),
        }
    )
    cand_df["sources"] = cand_df["candidate"].map(lambda x: ",".join(sorted(sources[int(x)])))
    cand_df["n_sources"] = cand_df["sources"].str.split(",").map(lambda parts: len([p for p in parts if p]))

    cand_df = cand_df.sort_values(["blended_score", "n_sources"], ascending=False).head(top_k).reset_index(drop=True)
    cand_df["candidate"] = cand_df["candidate"].astype(int)
    cand_df["n_sources"] = cand_df["n_sources"].astype(int)
    cand_df["blended_score"] = cand_df["blended_score"].astype(float)
    return cand_df


def build_ranking_table(
    holdout_df: pd.DataFrame,
    top_k_candidates: int,
    rules_index: dict[int, list[tuple[int, float]]],
    cooc_index: dict[int, list[tuple[int, float]]],
    cat_index: dict[str, list[tuple[int, float]]],
    item_to_cat: dict[int, str],
    emb_index: dict[int, list[tuple[int, float]]],
) -> tuple[pd.DataFrame, int]:
    rows = []
    skipped_empty = 0

    for row in holdout_df.itertuples(index=False):
        candidates = generate_candidates(
            context_items=list(row.context_items),
            top_k=top_k_candidates,
            rules_index=rules_index,
            cooc_index=cooc_index,
            cat_index=cat_index,
            item_to_cat=item_to_cat,
            emb_index=emb_index,
        )
        if candidates.empty:
            skipped_empty += 1
            continue

        candidates = candidates.copy()
        candidates["label"] = (candidates["candidate"].astype(int) == int(row.held_out)).astype(int)
        candidates["basket_id"] = int(row.basket_id)
        candidates["held_out"] = int(row.held_out)
        candidates["basket_size"] = int(row.basket_size)
        candidates["rank_blended"] = np.arange(1, len(candidates) + 1)
        rows.append(candidates)

    if not rows:
        return pd.DataFrame(), skipped_empty

    rank_df = pd.concat(rows, ignore_index=True)
    return rank_df, skipped_empty


def add_features(rank_df: pd.DataFrame) -> pd.DataFrame:
    df = rank_df.copy()
    src = df["sources"].fillna("")
    df["src_rules"] = src.str.contains("rules").astype(int)
    df["src_cooc"] = src.str.contains("cooc").astype(int)
    df["src_category"] = src.str.contains("category").astype(int)
    df["src_embedding"] = src.str.contains("embedding").astype(int)
    df["is_multi_source"] = (df["n_sources"] >= 2).astype(int)
    df["score_x_n_sources"] = df["blended_score"] * df["n_sources"]
    df["inv_rank_blended"] = 1.0 / df["rank_blended"].clip(lower=1)
    return df


def group_slices(group_sizes: list[int]) -> list[tuple[int, int]]:
    out = []
    start = 0
    for size in group_sizes:
        end = start + int(size)
        out.append((start, end))
        start = end
    return out


def groupwise_ndcg_at_k(labels: np.ndarray, scores: np.ndarray, group_sizes: list[int], k: int) -> float:
    values: list[float] = []
    for start, end in group_slices(group_sizes):
        y_true = labels[start:end]
        y_score = scores[start:end]
        if len(y_true) == 0:
            continue
        values.append(float(ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1), k=k)))
    return float(np.mean(values)) if values else 0.0


def groupwise_hitrate_at_k(labels: np.ndarray, scores: np.ndarray, group_sizes: list[int], k: int) -> float:
    hits: list[int] = []
    for start, end in group_slices(group_sizes):
        y_true = labels[start:end]
        y_score = scores[start:end]
        if len(y_true) == 0 or y_true.sum() == 0:
            continue
        top_idx = np.argsort(-y_score)[: min(k, len(y_score))]
        hits.append(int(y_true[top_idx].sum() > 0))
    return float(np.mean(hits)) if hits else 0.0


def compute_metrics_and_outputs(
    train_rank_df: pd.DataFrame,
    valid_rank_df: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = add_features(train_rank_df)
    valid_df = add_features(valid_rank_df)

    train_df = train_df.sort_values(["basket_id", "rank_blended"]).reset_index(drop=True)
    valid_df = valid_df.sort_values(["basket_id", "rank_blended"]).reset_index(drop=True)

    feature_cols = [
        "blended_score",
        "n_sources",
        "basket_size",
        "src_rules",
        "src_cooc",
        "src_category",
        "src_embedding",
        "is_multi_source",
        "score_x_n_sources",
        "rank_blended",
        "inv_rank_blended",
    ]
    for col in feature_cols + ["label"]:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)
        valid_df[col] = pd.to_numeric(valid_df[col], errors="coerce").fillna(0)

    X_train = train_df[feature_cols]
    y_train = train_df["label"].astype(int)
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["label"].astype(int)

    train_group = train_df.groupby("basket_id").size().to_list()
    valid_group = valid_df.groupby("basket_id").size().to_list()

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
    )
    ranker.fit(
        X_train,
        y_train,
        group=train_group,
        eval_set=[(X_valid, y_valid)],
        eval_group=[valid_group],
        eval_at=[5, 10, 20, 50],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    valid_pred = ranker.predict(X_valid)
    baseline_pred = valid_df["blended_score"].to_numpy()
    labels = y_valid.to_numpy()

    metric_rows = []
    for k in [5, 10, 20, 50]:
        metric_rows.append(
            {
                "k": k,
                "model": "LeakageSafe LightGBM",
                "ndcg": groupwise_ndcg_at_k(labels, valid_pred, valid_group, k),
                "hit_rate": groupwise_hitrate_at_k(labels, valid_pred, valid_group, k),
            }
        )
        metric_rows.append(
            {
                "k": k,
                "model": "Baseline blended_score",
                "ndcg": groupwise_ndcg_at_k(labels, baseline_pred, valid_group, k),
                "hit_rate": groupwise_hitrate_at_k(labels, baseline_pred, valid_group, k),
            }
        )
    metrics_df = pd.DataFrame(metric_rows)

    feature_importance_df = pd.DataFrame(
        {"feature": feature_cols, "importance": ranker.feature_importances_}
    ).sort_values("importance", ascending=False)

    valid_out = valid_df.copy()
    valid_out["pred"] = valid_pred

    return metrics_df, feature_importance_df, valid_out


def export_rules_pairs(rules_df: pd.DataFrame, cooc_df: pd.DataFrame, rules_top_out: Path, pairs_top_out: Path) -> None:
    rules_top_out.parent.mkdir(parents=True, exist_ok=True)
    pairs_top_out.parent.mkdir(parents=True, exist_ok=True)

    if rules_df.empty:
        pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"]).to_csv(
            rules_top_out, index=False
        )
    else:
        cols = [c for c in ["antecedent", "consequent", "support", "confidence", "lift"] if c in rules_df.columns]
        rules_df.sort_values(["lift", "confidence"], ascending=False).head(30)[cols].to_csv(rules_top_out, index=False)

    if cooc_df.empty:
        pd.DataFrame(columns=["item_a", "item_b", "pair_count", "cooc_score"]).to_csv(pairs_top_out, index=False)
    else:
        cols = [c for c in ["item_a", "item_b", "pair_count", "cooc_score"] if c in cooc_df.columns]
        cooc_df.sort_values(["pair_count", "cooc_score"], ascending=False).head(30)[cols].to_csv(
            pairs_top_out, index=False
        )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    baskets, basket_items = load_inputs(args.baskets_path, args.basket_items_path)
    train_ids, valid_ids, cutoff_date = split_baskets_by_time(
        baskets=baskets, train_ratio=args.train_ratio, seed=args.seed
    )
    print(f"[info] cutoff_date={cutoff_date.date() if cutoff_date != pd.Timestamp.min else 'random_split'}")
    print(f"[info] train baskets={len(train_ids)} | valid baskets={len(valid_ids)}")

    train_b2i = build_basket_to_items(basket_items, train_ids)
    valid_b2i = build_basket_to_items(basket_items, valid_ids)
    print(f"[info] train basket_to_items={len(train_b2i)} | valid basket_to_items={len(valid_b2i)}")

    rules_df, cooc_df, rules_index, cooc_index = build_rules_and_cooc_indices(
        basket_to_items=train_b2i,
        min_pair_count=args.min_pair_count,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_items_per_basket=args.max_items_per_basket,
        max_rules_per_item=args.max_rules_per_item,
        max_cooc_per_item=args.max_cooc_per_item,
    )
    print(f"[info] rules={len(rules_df)} | cooc_edges={len(cooc_df)}")

    train_items = basket_items[basket_items["basket_id"].isin(set(train_ids))]
    cat_index, item_to_cat = build_category_indices(train_items=train_items, max_category_items=args.max_category_items)
    print(f"[info] category nodes={len(cat_index)}")

    emb_index: dict[int, list[tuple[int, float]]] = {}
    if not args.disable_embedding:
        train_item_set = set(map(int, train_items["item_id"].dropna().astype(int).unique().tolist()))
        emb_index = build_embedding_index(
            neighbors_path=args.neighbors_path,
            train_item_set=train_item_set,
            max_embedding_neighbors=args.max_embedding_neighbors,
        )
    print(f"[info] embedding anchors={len(emb_index)}")

    holdout_train = make_holdout_table(train_b2i, sample_size=args.train_holdout_size, seed=args.seed)
    holdout_valid = make_holdout_table(valid_b2i, sample_size=args.valid_holdout_size, seed=args.seed + 1)
    print(f"[info] holdout train={len(holdout_train)} | holdout valid={len(holdout_valid)}")

    train_rank_df, skipped_train = build_ranking_table(
        holdout_df=holdout_train,
        top_k_candidates=args.top_k_candidates,
        rules_index=rules_index,
        cooc_index=cooc_index,
        cat_index=cat_index,
        item_to_cat=item_to_cat,
        emb_index=emb_index,
    )
    valid_rank_df, skipped_valid = build_ranking_table(
        holdout_df=holdout_valid,
        top_k_candidates=args.top_k_candidates,
        rules_index=rules_index,
        cooc_index=cooc_index,
        cat_index=cat_index,
        item_to_cat=item_to_cat,
        emb_index=emb_index,
    )
    print(f"[info] ranking train rows={len(train_rank_df)} (empty={skipped_train})")
    print(f"[info] ranking valid rows={len(valid_rank_df)} (empty={skipped_valid})")

    if train_rank_df.empty or valid_rank_df.empty:
        raise RuntimeError("Ranking tables are empty. Candidate generation did not produce training/validation rows.")
    if train_rank_df["label"].sum() == 0 or valid_rank_df["label"].sum() == 0:
        raise RuntimeError("No positive labels found in ranking tables.")

    metrics_df, feature_importance_df, valid_out_df = compute_metrics_and_outputs(
        train_rank_df=train_rank_df, valid_rank_df=valid_rank_df, seed=args.seed
    )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.feature_importance_out.parent.mkdir(parents=True, exist_ok=True)
    args.valid_predictions_out.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(args.metrics_out, index=False)
    feature_importance_df.to_csv(args.feature_importance_out, index=False)
    valid_out_df.to_csv(args.valid_predictions_out, index=False)
    export_rules_pairs(
        rules_df=rules_df,
        cooc_df=cooc_df,
        rules_top_out=args.rules_top_out,
        pairs_top_out=args.pairs_top_out,
    )

    print(f"[done] wrote {args.metrics_out}")
    print(f"[done] wrote {args.feature_importance_out}")
    print(f"[done] wrote {args.valid_predictions_out}")
    print(f"[done] wrote {args.rules_top_out}")
    print(f"[done] wrote {args.pairs_top_out}")
    print("[done] leakage-safe training pipeline completed")


if __name__ == "__main__":
    main()
