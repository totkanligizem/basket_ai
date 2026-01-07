from __future__ import annotations

from pathlib import Path
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import plotly.express as px


# -----------------------------
# Paths
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
METRICS_DIR = DATA_DIR / "metrics"

RANKING_CSV = METRICS_DIR / "ranking_metrics.csv"
EDA_TS_CSV = DATA_DIR / "eda_timeseries.csv"
TOP_CAT_CSV = DATA_DIR / "top_categories.csv"
RULES_CSV = DATA_DIR / "rules_top.csv"
PAIRS_CSV = DATA_DIR / "cooc_top_pairs.csv"
FEAT_IMP_CSV = DATA_DIR / "feature_importance.csv"


# -----------------------------
# Loaders (robust + non-fatal)
# -----------------------------
def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists() and path.is_file():
        try:
            df = pd.read_csv(path)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            return None
    return None


def load_all() -> dict[str, pd.DataFrame | None]:
    data = {
        "ranking": _read_csv(RANKING_CSV),
        "eda_ts": _read_csv(EDA_TS_CSV),
        "top_cat": _read_csv(TOP_CAT_CSV),
        "rules": _read_csv(RULES_CSV),
        "pairs": _read_csv(PAIRS_CSV),
        "feat_imp": _read_csv(FEAT_IMP_CSV),
    }
    return data


DATA = load_all()


# -----------------------------
# Helpers
# -----------------------------
def pill(text: str) -> html.Div:
    return html.Div(text, className="pill")


def card(children, title: str | None = None) -> html.Div:
    return html.Div(
        [
            html.Div(title, className="card-title") if title else None,
            html.Div(children, className="card-body"),
        ],
        className="card",
    )


def kpi_card(label: str, value: str, sub: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub"),
        ],
        className="kpi",
    )


def df_preview_table(df: pd.DataFrame, max_rows: int = 10) -> html.Div:
    view = df.head(max_rows).copy()
    return html.Div(
        [
            html.Table(
                [
                    html.Thead(
                        html.Tr([html.Th(c) for c in view.columns])
                    ),
                    html.Tbody(
                        [
                            html.Tr([html.Td(str(view.iloc[i, j])) for j in range(view.shape[1])])
                            for i in range(view.shape[0])
                        ]
                    ),
                ],
                className="table",
            ),
            html.Div(f"Showing top {min(max_rows, len(df))} rows of {len(df)}", className="table-note"),
        ]
    )


def safe_note(text: str) -> html.Div:
    return html.Div(text, className="note")


def data_source_line() -> str:
    parts = []
    if DATA.get("ranking") is not None:
        parts.append(f"ranking_metrics.csv ({len(DATA['ranking'])} rows)")
    if DATA.get("eda_ts") is not None:
        parts.append(f"eda_timeseries.csv ({len(DATA['eda_ts'])} rows)")
    if DATA.get("top_cat") is not None:
        parts.append(f"top_categories.csv ({len(DATA['top_cat'])} rows)")
    if DATA.get("rules") is not None:
        parts.append(f"rules_top.csv ({len(DATA['rules'])} rows)")
    if DATA.get("pairs") is not None:
        parts.append(f"cooc_top_pairs.csv ({len(DATA['pairs'])} rows)")
    if DATA.get("feat_imp") is not None:
        parts.append(f"feature_importance.csv ({len(DATA['feat_imp'])} rows)")
    return " • ".join(parts) if parts else "No CSV found yet (run notebook exports)"


# -----------------------------
# Figures
# -----------------------------
def fig_ranking(df: pd.DataFrame, metric: str):
    # Expect columns: k, model, ndcg, hit_rate
    y_map = {"ndcg": "ndcg", "hit_rate": "hit_rate"}
    y = y_map[metric]

    title = "NDCG@K — Validation (Ranking quality, higher is better)" if metric == "ndcg" else \
            "HitRate@K — Validation (Coverage / recall proxy, higher is better)"

    y_label = "NDCG@K (ranking quality)" if metric == "ndcg" else "HitRate@K (hit ratio)"
    x_label = "K (Top-K cutoff)"

    fig = px.line(
        df.sort_values("k"),
        x="k",
        y=y,
        color="model" if "model" in df.columns else None,
        markers=True,
        title=title,
        labels={"k": x_label, y: y_label, "model": "Model"},
    )
    fig.update_layout(
        margin=dict(l=30, r=20, t=60, b=40),
        height=420,
        legend_title_text="",
    )
    return fig


def fig_eda_timeseries(df: pd.DataFrame):
    # Expect: date, baskets, (optional revenue, aov)
    if "date" in df.columns:
        df2 = df.copy()
        df2["date"] = pd.to_datetime(df2["date"])
    else:
        df2 = df.copy()

    fig = px.line(
        df2.sort_values("date"),
        x="date",
        y="baskets" if "baskets" in df2.columns else df2.columns[1],
        markers=False,
        title="Basket volume over time",
        labels={"date": "Date", "baskets": "Number of baskets"},
    )
    fig.update_layout(margin=dict(l=30, r=20, t=60, b=40), height=360)
    return fig


def fig_top_categories(df: pd.DataFrame):
    # Expect: category, orders, (optional revenue)
    x = "orders" if "orders" in df.columns else df.columns[1]
    fig = px.bar(
        df.sort_values(x, ascending=True).tail(15),
        x=x,
        y="category" if "category" in df.columns else df.columns[0],
        orientation="h",
        title="Top categories by basket participation",
        labels={x: "Basket count (unique baskets)", "category": "Category"},
    )
    fig.update_layout(margin=dict(l=30, r=20, t=60, b=40), height=420)
    return fig


def fig_feature_importance(df: pd.DataFrame):
    # Expect: feature, importance
    df2 = df.copy()
    if "importance" in df2.columns:
        df2 = df2.sort_values("importance", ascending=True).tail(12)
        fig = px.bar(
            df2,
            x="importance",
            y="feature",
            orientation="h",
            title="Ranker feature importance (LightGBM)",
            labels={"importance": "Importance (gain/split proxy)", "feature": "Feature"},
        )
    else:
        fig = px.bar(df2.head(12), title="Feature importance")
    fig.update_layout(margin=dict(l=30, r=20, t=60, b=40), height=420)
    return fig


# -----------------------------
# App
# -----------------------------
app = Dash(
    __name__,
    assets_folder=str(APP_DIR / "assets"),
    title="Basket AI — Executive Recommendation Dashboard",
)

VIEW_OPTIONS = [
    {"label": "Executive Summary", "value": "summary"},
    {"label": "Data & EDA", "value": "eda"},
    {"label": "Basket Insights (Rules)", "value": "insights"},
    {"label": "Model Diagnostics (Ranker)", "value": "model"},
]

app.layout = html.Div(
    className="page",
    children=[
        html.Header(
            className="header",
            children=[
                html.Div(
                    [
                        html.H1("Basket AI — Executive Recommendation Dashboard"),
                        html.Div(
                            "Multi-signal candidate generation + Learning-to-Rank (LightGBM) — offline evaluation view",
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
                            value="summary",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                    ],
                ),
            ],
        ),

        html.Div(pill(f"Data source: {data_source_line()}"), className="topbar"),

        html.Div(id="content", className="content"),
    ],
)


# -----------------------------
# Page builders
# -----------------------------
def page_summary() -> html.Div:
    # KPIs from ranking (prefer K=10)
    kpi_ndcg = "—"
    kpi_hit = "—"
    kpi_note = "Snapshot computed from: ranking_metrics.csv (offline holdout evaluation)."

    df = DATA.get("ranking")
    if df is not None and {"k", "ndcg", "hit_rate"}.issubset(df.columns):
        df2 = df.copy()
        try:
            df2["k"] = df2["k"].astype(int)
        except Exception:
            pass
        row10 = df2[df2["k"] == 10]
        if len(row10) == 1:
            kpi_ndcg = f"{float(row10['ndcg'].iloc[0]):.3f}"
            kpi_hit = f"{float(row10['hit_rate'].iloc[0]):.3f}"
        else:
            # fallback to smallest K
            row = df2.sort_values("k").head(1)
            kpi_ndcg = f"{float(row['ndcg'].iloc[0]):.3f}"
            kpi_hit = f"{float(row['hit_rate'].iloc[0]):.3f}"

    left = card(
        [
            html.Div(
                "We built an offline evaluation dashboard for a basket-based recommender. "
                "The pipeline generates candidate items from multiple signals, then a LightGBM ranker orders them. "
                "This view answers: “Does the held-out true item land near the top (quality) and does it appear at all (coverage)?”",
                className="p",
            ),
            html.Ul(
                [
                    html.Li("NDCG@K → position-aware ranking quality (Top-5 / Top-10 matter most for UX)."),
                    html.Li("HitRate@K → recall/coverage proxy (does the true item appear within Top-K?)."),
                    html.Li("Metrics are computed per basket_id and averaged across validation."),
                ],
                className="bullets",
            ),
            safe_note(
                "Next: add baseline comparison, segment slices (category/price band), "
                "and lightweight ROI proxies (CTR uplift × margin) for a business-readable story."
            ),
        ],
        title="Executive Summary",
    )

    right = card(
        [
            html.Div(
                [
                    html.Div("Key KPIs (Validation)", className="card-title-mini"),
                    html.Div(
                        [
                            kpi_card("NDCG@10", kpi_ndcg, "Ranking quality (higher is better)"),
                            kpi_card("HitRate@10", kpi_hit, "Coverage / recall proxy (higher is better)"),
                        ],
                        className="kpi-grid",
                    ),
                    html.Div(kpi_note, className="kpi-foot"),
                ]
            )
        ],
        title=None,
    )

    return html.Div(className="grid", children=[left, right])


def page_eda() -> html.Div:
    ts = DATA.get("eda_ts")
    cats = DATA.get("top_cat")

    left_children = []
    if ts is not None and {"date"}.issubset(ts.columns):
        left_children += [
            dcc.Graph(figure=fig_eda_timeseries(ts), config={"displayModeBar": False}),
            safe_note("Interpretation: basket volume trend helps detect seasonality, campaign spikes, and data drift."),
        ]
    else:
        left_children += [safe_note("eda_timeseries.csv not found or missing expected columns.")]

    right_children = []
    if cats is not None and {"category"}.issubset(cats.columns):
        right_children += [
            dcc.Graph(figure=fig_top_categories(cats), config={"displayModeBar": False}),
            safe_note("Interpretation: category concentration hints where cross-sell rules and ranker uplift may come from."),
        ]
    else:
        right_children += [safe_note("top_categories.csv not found or missing expected columns.")]

    return html.Div(
        className="grid",
        children=[
            card(left_children, title="Data & EDA — Volume"),
            card(right_children, title="Data & EDA — Category mix"),
        ],
    )


def page_insights() -> html.Div:
    rules = DATA.get("rules")
    pairs = DATA.get("pairs")

    left_children = []
    if rules is not None and {"antecedent", "consequent"}.issubset(rules.columns):
        show_cols = [c for c in ["antecedent", "consequent", "lift", "confidence", "support"] if c in rules.columns]
        left_children += [
            df_preview_table(rules[show_cols], max_rows=12),
            safe_note("Interpretation: high lift + good confidence rules suggest strong co-purchase structure."),
        ]
    else:
        left_children += [safe_note("rules_top.csv not found or missing expected columns.")]

    right_children = []
    if pairs is not None and {"item_a", "item_b"}.issubset(pairs.columns):
        show_cols = [c for c in ["item_a", "item_b", "pair_count", "lift", "confidence", "support"] if c in pairs.columns]
        right_children += [
            df_preview_table(pairs[show_cols], max_rows=12),
            safe_note("Interpretation: top pairs help merchandising narratives (bundles, placements, campaign planning)."),
        ]
    else:
        right_children += [safe_note("cooc_top_pairs.csv not found or missing expected columns.")]

    return html.Div(
        className="grid",
        children=[
            card(left_children, title="Basket Insights — Association Rules (Top)"),
            card(right_children, title="Basket Insights — Top co-occurring pairs"),
        ],
    )


def page_model() -> html.Div:
    ranking = DATA.get("ranking")
    feat = DATA.get("feat_imp")

    top = []
    if feat is not None and {"feature", "importance"}.issubset(feat.columns):
        top += [
            dcc.Graph(figure=fig_feature_importance(feat), config={"displayModeBar": False}),
            safe_note("Interpretation: importance shows which signals the ranker relies on; monitor drift & stability over time."),
        ]
    else:
        top += [safe_note("feature_importance.csv not found or missing expected columns.")]

    bottom = []
    if ranking is not None and {"k", "ndcg", "hit_rate"}.issubset(ranking.columns):
        bottom += [
            dcc.Graph(figure=fig_ranking(ranking, "ndcg"), config={"displayModeBar": False}),
            safe_note(
                "Interpretation: NDCG@K rewards placing the true (held-out) item near the top. "
                "Because it is position-aware, Top-5 / Top-10 improvements typically matter most for UX."
            ),
            dcc.Graph(figure=fig_ranking(ranking, "hit_rate"), config={"displayModeBar": False}),
            safe_note(
                "Interpretation: HitRate@K measures whether the true item appears anywhere in Top-K. "
                "It usually increases with K and helps validate candidate+ranker recall behavior."
            ),
        ]
    else:
        bottom += [safe_note("ranking_metrics.csv not found or missing expected columns.")]

    return html.Div(
        className="stack",
        children=[
            card(top, title="Model Diagnostics — Ranker behavior"),
            card(bottom, title="Ranking Quality (NDCG / HitRate)"),
        ],
    )


@app.callback(Output("content", "children"), Input("view", "value"))
def render(view):
    if view == "eda":
        return page_eda()
    if view == "insights":
        return page_insights()
    if view == "model":
        return page_model()
    return page_summary()


if __name__ == "__main__":
    app.run_server(debug=True)
    