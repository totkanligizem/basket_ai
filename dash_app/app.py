from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
METRICS_DIR = DATA_DIR / "metrics"

RANKING_CSV = METRICS_DIR / "ranking_metrics.csv"
VALID_PRED_CSV = METRICS_DIR / "valid_predictions.csv"
EDA_TS_CSV = DATA_DIR / "eda_timeseries.csv"
TOP_CAT_CSV = DATA_DIR / "top_categories.csv"
RULES_CSV = DATA_DIR / "rules_top.csv"
PAIRS_CSV = DATA_DIR / "cooc_top_pairs.csv"
FEAT_IMP_CSV = DATA_DIR / "feature_importance.csv"

ALL_MODELS = "__all__"
UI_REV = "dash_v2"
GRAPH_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}
MODEL_COLORS = ["#0b6efd", "#0ea5a4", "#f59e0b", "#d6336c", "#198754", "#6c757d"]


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        return None
    return None


def _load_data() -> dict[str, pd.DataFrame | None]:
    data = {
        "ranking": _read_csv(RANKING_CSV),
        "valid_pred": _read_csv(VALID_PRED_CSV),
        "eda_ts": _read_csv(EDA_TS_CSV),
        "top_cat": _read_csv(TOP_CAT_CSV),
        "rules": _read_csv(RULES_CSV),
        "pairs": _read_csv(PAIRS_CSV),
        "feat_imp": _read_csv(FEAT_IMP_CSV),
    }

    ranking = data["ranking"]
    if ranking is not None and {"k", "model", "ndcg", "hit_rate"}.issubset(ranking.columns):
        ranking = ranking.copy()
        ranking["k"] = pd.to_numeric(ranking["k"], errors="coerce")
        ranking["ndcg"] = pd.to_numeric(ranking["ndcg"], errors="coerce")
        ranking["hit_rate"] = pd.to_numeric(ranking["hit_rate"], errors="coerce")
        ranking["model"] = ranking["model"].astype(str)
        ranking = ranking.dropna(subset=["k", "ndcg", "hit_rate"]).sort_values(["model", "k"])
        data["ranking"] = ranking
    else:
        data["ranking"] = None

    eda = data["eda_ts"]
    if eda is not None and {"date", "baskets"}.issubset(eda.columns):
        eda = eda.copy()
        eda["date"] = pd.to_datetime(eda["date"], errors="coerce")
        for col in ["baskets", "distinct_customers", "revenue", "aov"]:
            if col in eda.columns:
                eda[col] = pd.to_numeric(eda[col], errors="coerce")
        data["eda_ts"] = eda.dropna(subset=["date"]).sort_values("date")
    else:
        data["eda_ts"] = None

    top_cat = data["top_cat"]
    if top_cat is not None and {"category", "orders"}.issubset(top_cat.columns):
        top_cat = top_cat.copy()
        top_cat["orders"] = pd.to_numeric(top_cat["orders"], errors="coerce")
        if "revenue" in top_cat.columns:
            top_cat["revenue"] = pd.to_numeric(top_cat["revenue"], errors="coerce")
        data["top_cat"] = top_cat.dropna(subset=["orders"])
    else:
        data["top_cat"] = None

    if data["rules"] is not None:
        rules = data["rules"].copy()
        for col in ["lift", "confidence", "support"]:
            if col in rules.columns:
                rules[col] = pd.to_numeric(rules[col], errors="coerce")
        data["rules"] = rules

    if data["pairs"] is not None:
        pairs = data["pairs"].copy()
        for col in ["pair_count", "lift", "confidence", "support", "cooc_score"]:
            if col in pairs.columns:
                pairs[col] = pd.to_numeric(pairs[col], errors="coerce")
        data["pairs"] = pairs

    feat = data["feat_imp"]
    if feat is not None and {"feature", "importance"}.issubset(feat.columns):
        feat = feat.copy()
        feat["importance"] = pd.to_numeric(feat["importance"], errors="coerce")
        data["feat_imp"] = feat.dropna(subset=["importance"]).sort_values("importance", ascending=False)
    else:
        data["feat_imp"] = None

    return data


DATA = _load_data()


def _baseline_model() -> str | None:
    ranking = DATA.get("ranking")
    if ranking is None or ranking.empty:
        return None
    models = ranking["model"].dropna().astype(str).unique().tolist()
    baseline = [m for m in models if "baseline" in m.lower()]
    return baseline[0] if baseline else None


def _default_model() -> str:
    ranking = DATA.get("ranking")
    if ranking is None or ranking.empty:
        return ALL_MODELS
    models = ranking["model"].dropna().astype(str).unique().tolist()
    baseline = _baseline_model()
    preferred = [m for m in models if m != baseline]
    return preferred[0] if preferred else models[0]


def _model_options() -> list[dict[str, str]]:
    ranking = DATA.get("ranking")
    if ranking is None or ranking.empty:
        return [{"label": "No ranking metrics", "value": ALL_MODELS}]
    options = [{"label": "All Models", "value": ALL_MODELS}]
    options += [{"label": m, "value": m} for m in ranking["model"].dropna().astype(str).unique().tolist()]
    return options


def _source_status() -> str:
    labels = {
        "ranking": "ranking_metrics.csv",
        "eda_ts": "eda_timeseries.csv",
        "top_cat": "top_categories.csv",
        "rules": "rules_top.csv",
        "pairs": "cooc_top_pairs.csv",
        "feat_imp": "feature_importance.csv",
        "valid_pred": "valid_predictions.csv",
    }
    parts = []
    for key, label in labels.items():
        df = DATA.get(key)
        if df is None:
            parts.append(f"{label}: missing")
        else:
            parts.append(f"{label}: {len(df):,} rows")
    return " | ".join(parts)


def _fmt(value: float | None, precision: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{precision}f}"


def _metric_at_k(ranking: pd.DataFrame, model: str, metric: str, k_target: int = 10) -> tuple[float | None, int | None]:
    model_df = ranking[ranking["model"] == model]
    if model_df.empty:
        return None, None
    if (model_df["k"] == k_target).any():
        row = model_df[model_df["k"] == k_target].iloc[0]
        return float(row[metric]), int(row["k"])
    nearest = model_df.assign(dist=(model_df["k"] - k_target).abs()).sort_values("dist").iloc[0]
    return float(nearest[metric]), int(nearest["k"])


def _apply_figure_style(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=height,
        margin=dict(l=56, r=20, t=62, b=50),
        title=dict(text=title, x=0.01, xanchor="left"),
        font=dict(family="IBM Plex Sans, Segoe UI, Arial, sans-serif", size=12, color="#0f172a"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0)",
        ),
        uirevision=UI_REV,
        colorway=MODEL_COLORS,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
        linecolor="rgba(15, 23, 42, 0.2)",
        ticks="outside",
        tickfont=dict(color="#334155"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(15, 23, 42, 0.08)",
        zeroline=False,
        linecolor="rgba(15, 23, 42, 0.2)",
        ticks="outside",
        tickfont=dict(color="#334155"),
    )
    return fig


def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#64748b"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _apply_figure_style(fig, title=title, height=300)


def _ranking_figure(metric: str, selected_model: str) -> go.Figure:
    ranking = DATA.get("ranking")
    if ranking is None or ranking.empty:
        return _empty_figure("Ranking", "ranking_metrics.csv not found")

    title = "NDCG@K by Model" if metric == "ndcg" else "HitRate@K by Model"
    y_label = "NDCG@K" if metric == "ndcg" else "HitRate@K"

    plot_df = ranking.copy()
    baseline = _baseline_model()
    if selected_model != ALL_MODELS:
        models = [selected_model]
        if baseline and baseline != selected_model:
            models.append(baseline)
        plot_df = plot_df[plot_df["model"].isin(models)]

    fig = px.line(
        plot_df.sort_values("k"),
        x="k",
        y=metric,
        color="model",
        markers=True,
        labels={"k": "Top-K", metric: y_label, "model": "Model"},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=7))
    return _apply_figure_style(fig, title=title)


def _delta_figure(selected_model: str) -> go.Figure:
    ranking = DATA.get("ranking")
    baseline = _baseline_model()
    if (
        ranking is None
        or ranking.empty
        or selected_model in [ALL_MODELS, baseline, None]
        or baseline is None
        or selected_model not in ranking["model"].unique()
    ):
        return _empty_figure("Delta vs Baseline", "Select a non-baseline model to compare")

    left = ranking[ranking["model"] == selected_model][["k", "ndcg", "hit_rate"]].rename(
        columns={"ndcg": "ndcg_model", "hit_rate": "hit_model"}
    )
    right = ranking[ranking["model"] == baseline][["k", "ndcg", "hit_rate"]].rename(
        columns={"ndcg": "ndcg_base", "hit_rate": "hit_base"}
    )
    merged = left.merge(right, on="k", how="inner")
    if merged.empty:
        return _empty_figure("Delta vs Baseline", "No common K values between models")

    merged["delta_ndcg"] = merged["ndcg_model"] - merged["ndcg_base"]
    merged["delta_hit"] = merged["hit_model"] - merged["hit_base"]
    fig = go.Figure()
    fig.add_bar(x=merged["k"], y=merged["delta_ndcg"], name="Delta NDCG", marker_color="#0b6efd")
    fig.add_bar(x=merged["k"], y=merged["delta_hit"], name="Delta HitRate", marker_color="#0ea5a4")
    fig.update_layout(barmode="group")
    return _apply_figure_style(fig, title=f"{selected_model} vs {baseline} (delta)")


def _eda_volume_figure() -> go.Figure:
    ts = DATA.get("eda_ts")
    if ts is None:
        return _empty_figure("Basket Volume", "eda_timeseries.csv not found")
    fig = px.line(ts, x="date", y="baskets", markers=False, labels={"date": "Date", "baskets": "Baskets"})
    fig.update_traces(line=dict(width=3, color="#0b6efd"))
    return _apply_figure_style(fig, title="Daily Basket Volume")


def _eda_revenue_figure() -> go.Figure:
    ts = DATA.get("eda_ts")
    if ts is None or "revenue" not in ts.columns:
        return _empty_figure("Revenue & AOV", "Revenue columns missing")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts["date"],
            y=ts["revenue"],
            mode="lines",
            name="Revenue",
            line=dict(color="#0ea5a4", width=3),
        )
    )
    if "aov" in ts.columns and ts["aov"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=ts["date"],
                y=ts["aov"],
                mode="lines",
                name="AOV",
                line=dict(color="#f59e0b", width=2),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title="AOV",
                overlaying="y",
                side="right",
                showgrid=False,
                zeroline=False,
            )
        )
    return _apply_figure_style(fig, title="Revenue Trend and AOV")


def _top_category_figure() -> go.Figure:
    top_cat = DATA.get("top_cat")
    if top_cat is None:
        return _empty_figure("Top Categories", "top_categories.csv not found")

    df = top_cat.sort_values("orders", ascending=True).tail(15)
    fig = px.bar(
        df,
        x="orders",
        y="category",
        orientation="h",
        labels={"orders": "Unique baskets", "category": "Category"},
    )
    fig.update_traces(marker_color="#0b6efd")
    return _apply_figure_style(fig, title="Top Categories by Basket Participation")


def _rules_lift_figure() -> go.Figure:
    rules = DATA.get("rules")
    if rules is None or "lift" not in rules.columns:
        return _empty_figure("Rules Lift", "rules_top.csv missing lift column")
    df = rules.dropna(subset=["lift"]).sort_values("lift", ascending=False).head(12).copy()
    if df.empty:
        return _empty_figure("Rules Lift", "No non-null lift values")
    df["rule"] = df["antecedent"].astype(str) + " -> " + df["consequent"].astype(str)
    fig = px.bar(df.sort_values("lift"), x="lift", y="rule", orientation="h")
    fig.update_traces(marker_color="#0ea5a4")
    return _apply_figure_style(fig, title="Top Association Rules by Lift")


def _pairs_figure() -> go.Figure:
    pairs = DATA.get("pairs")
    if pairs is None or "pair_count" not in pairs.columns:
        return _empty_figure("Top Pairs", "cooc_top_pairs.csv missing pair_count")
    df = pairs.dropna(subset=["pair_count"]).sort_values("pair_count", ascending=False).head(12).copy()
    if df.empty:
        return _empty_figure("Top Pairs", "No pair_count values found")
    df["pair"] = df["item_a"].astype(str) + " | " + df["item_b"].astype(str)
    fig = px.bar(df.sort_values("pair_count"), x="pair_count", y="pair", orientation="h")
    fig.update_traces(marker_color="#f59e0b")
    return _apply_figure_style(fig, title="Top Co-occurring Item Pairs")


def _feature_importance_figure() -> go.Figure:
    feat = DATA.get("feat_imp")
    if feat is None:
        return _empty_figure("Feature Importance", "feature_importance.csv not found")
    df = feat.sort_values("importance", ascending=True).tail(15)
    fig = px.bar(df, x="importance", y="feature", orientation="h")
    fig.update_traces(marker_color="#0b6efd")
    return _apply_figure_style(fig, title="Ranker Feature Importance")


def kpi_card(label: str, value: str, subtext: str, delta: str | None = None) -> html.Div:
    return html.Div(
        className="kpi",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(subtext, className="kpi-sub"),
            html.Div(delta, className="kpi-delta") if delta else None,
        ],
    )


def card(title: str, body: list, note: str | None = None) -> html.Div:
    children = [html.Div(title, className="card-title"), html.Div(body, className="card-body")]
    if note:
        children.append(html.Div(note, className="note"))
    return html.Div(children, className="card")


def preview_table(df: pd.DataFrame, max_rows: int = 12) -> html.Div:
    view = df.head(max_rows).copy()
    return html.Div(
        [
            html.Div(
                html.Table(
                    [
                        html.Thead(html.Tr([html.Th(c) for c in view.columns])),
                        html.Tbody(
                            [
                                html.Tr([html.Td(str(view.iloc[i, j])) for j in range(view.shape[1])])
                                for i in range(view.shape[0])
                            ]
                        ),
                    ],
                    className="table",
                ),
                className="table-wrap",
            ),
            html.Div(f"Showing {min(max_rows, len(df))} of {len(df)} rows", className="table-note"),
        ]
    )


def page_overview(selected_model: str) -> html.Div:
    ranking = DATA.get("ranking")
    if ranking is None:
        return card("Overview", [html.Div("ranking_metrics.csv bulunamadi.", className="note")])

    model = selected_model if selected_model in ranking["model"].unique() else _default_model()
    baseline = _baseline_model()

    ndcg_v, ndcg_k = _metric_at_k(ranking, model, "ndcg", 10)
    hit_v, hit_k = _metric_at_k(ranking, model, "hit_rate", 10)

    delta_ndcg = None
    delta_hit = None
    if baseline and baseline != model:
        base_ndcg, _ = _metric_at_k(ranking, baseline, "ndcg", ndcg_k or 10)
        base_hit, _ = _metric_at_k(ranking, baseline, "hit_rate", hit_k or 10)
        if base_ndcg is not None and ndcg_v is not None:
            delta_ndcg = f"Delta vs baseline: {ndcg_v - base_ndcg:+.3f}"
        if base_hit is not None and hit_v is not None:
            delta_hit = f"Delta vs baseline: {hit_v - base_hit:+.3f}"

    rules_rows = len(DATA["rules"]) if DATA["rules"] is not None else 0
    pairs_rows = len(DATA["pairs"]) if DATA["pairs"] is not None else 0

    kpis = html.Div(
        className="kpi-grid",
        children=[
            kpi_card(f"NDCG@{ndcg_k or 10}", _fmt(ndcg_v), model, delta_ndcg),
            kpi_card(f"HitRate@{hit_k or 10}", _fmt(hit_v), model, delta_hit),
            kpi_card("Rules", f"{rules_rows:,}", "Available signal rows"),
            kpi_card("Cooc Pairs", f"{pairs_rows:,}", "Available pair rows"),
        ],
    )

    inventory = card(
        "Artifact Health",
        [html.Div(_source_status(), className="artifact-line")],
        note="Dashboard reads local CSV artifacts. Missing files are shown as placeholders.",
    )

    charts = html.Div(
        className="stack",
        children=[
            dcc.Graph(figure=_ranking_figure("ndcg", model), config=GRAPH_CONFIG, className="graph"),
            dcc.Graph(figure=_ranking_figure("hit_rate", model), config=GRAPH_CONFIG, className="graph"),
        ],
    )

    left = card("Validation KPIs", [kpis], note="Metrics are query-level aggregates on held-out items.")
    right = card("Ranking Curves", [charts])

    return html.Div(className="stack", children=[html.Div(className="grid", children=[left, right]), inventory])


def page_eda() -> html.Div:
    return html.Div(
        className="grid",
        children=[
            card(
                "Demand Volume",
                [dcc.Graph(figure=_eda_volume_figure(), config=GRAPH_CONFIG, className="graph")],
                note="Use this chart to watch seasonality and sudden shifts.",
            ),
            card(
                "Revenue and AOV",
                [dcc.Graph(figure=_eda_revenue_figure(), config=GRAPH_CONFIG, className="graph")],
                note="If revenue columns are missing, this panel will show a placeholder.",
            ),
            card(
                "Category Mix",
                [dcc.Graph(figure=_top_category_figure(), config=GRAPH_CONFIG, className="graph")],
                note="Top categories by unique basket participation.",
            ),
        ],
    )


def page_signals() -> html.Div:
    rules = DATA.get("rules")
    pairs = DATA.get("pairs")
    rules_table = preview_table(rules, max_rows=12) if rules is not None else html.Div("rules_top.csv missing", className="note")
    pairs_table = preview_table(pairs, max_rows=12) if pairs is not None else html.Div("cooc_top_pairs.csv missing", className="note")

    return html.Div(
        className="stack",
        children=[
            html.Div(
                className="grid",
                children=[
                    card("Rules Lift Distribution", [dcc.Graph(figure=_rules_lift_figure(), config=GRAPH_CONFIG, className="graph")]),
                    card("Co-occurrence Strength", [dcc.Graph(figure=_pairs_figure(), config=GRAPH_CONFIG, className="graph")]),
                ],
            ),
            html.Div(
                className="grid",
                children=[
                    card("Top Rules Table", [rules_table]),
                    card("Top Pair Table", [pairs_table]),
                ],
            ),
        ],
    )


def page_model(selected_model: str) -> html.Div:
    ranking = DATA.get("ranking")
    model = selected_model
    if ranking is not None and selected_model not in ranking["model"].unique():
        model = _default_model()

    metric_table = preview_table(ranking.sort_values(["model", "k"]), max_rows=20) if ranking is not None else html.Div("No ranking data", className="note")
    valid_pred = DATA.get("valid_pred")
    pred_table = preview_table(valid_pred, max_rows=12) if valid_pred is not None else html.Div("valid_predictions.csv missing", className="note")

    return html.Div(
        className="stack",
        children=[
            html.Div(
                className="grid",
                children=[
                    card(
                        "Feature Importance",
                        [dcc.Graph(figure=_feature_importance_figure(), config=GRAPH_CONFIG, className="graph")],
                    ),
                    card(
                        "Model Delta vs Baseline",
                        [dcc.Graph(figure=_delta_figure(model), config=GRAPH_CONFIG, className="graph")],
                        note="Shows absolute metric improvement by K.",
                    ),
                ],
            ),
            html.Div(
                className="grid",
                children=[
                    card("Ranking Metric Table", [metric_table]),
                    card("Validation Predictions Sample", [pred_table]),
                ],
            ),
        ],
    )


app = Dash(__name__, assets_folder=str(APP_DIR / "assets"), title="Basket AI Dashboard")

VIEW_OPTIONS = [
    {"label": "Overview", "value": "overview"},
    {"label": "Data & EDA", "value": "eda"},
    {"label": "Candidate Signals", "value": "signals"},
    {"label": "Model Diagnostics", "value": "model"},
]

app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            className="header",
            children=[
                html.Div(
                    [
                        html.Div("BASKET AI", className="eyebrow"),
                        html.H1("Recommendation System Performance Dashboard"),
                        html.Div(
                            "Leakage-safe ranking evaluation with candidate signal diagnostics.",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    className="controls",
                    children=[
                        html.Div("View", className="control-label"),
                        dcc.Dropdown(
                            id="view",
                            options=VIEW_OPTIONS,
                            value="overview",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                        html.Div("Model", className="control-label control-label-spaced"),
                        dcc.Dropdown(
                            id="model",
                            options=_model_options(),
                            value=_default_model(),
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),
            ],
        ),
        html.Div(_source_status(), className="status-line"),
        html.Div(id="content", className="content content-lock"),
        html.Div("Dash + Plotly • Stable local artifacts • Professional view", className="footer"),
    ],
)


@app.callback(Output("content", "children"), Input("view", "value"), Input("model", "value"))
def render(view: str, model: str):
    if view == "eda":
        return page_eda()
    if view == "signals":
        return page_signals()
    if view == "model":
        return page_model(model)
    return page_overview(model)


if __name__ == "__main__":
    app.run(debug=True)
