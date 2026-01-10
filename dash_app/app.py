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
    return {
        "ranking": _read_csv(RANKING_CSV),
        "eda_ts": _read_csv(EDA_TS_CSV),
        "top_cat": _read_csv(TOP_CAT_CSV),
        "rules": _read_csv(RULES_CSV),
        "pairs": _read_csv(PAIRS_CSV),
        "feat_imp": _read_csv(FEAT_IMP_CSV),
    }


DATA = load_all()


# -----------------------------
# UI constants (stable layout)
# -----------------------------
# Fixed heights prevent "page sliding" while Plotly is re-rendering.
GRAPH_H_SM = 360
GRAPH_H_MD = 420

GRAPH_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
    "responsive": True,  # keep responsive, but we fix container heights + uirevision
}

# Prevent Plotly from resetting view on tab switches / rerenders
UI_REV = "stable_v1"


# -----------------------------
# Plotly theme (dark, modern, subtle grids)
# -----------------------------
def apply_plotly_theme(fig, title: str):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",  # not harsh white
        height=GRAPH_H_MD,
        margin=dict(l=52, r=28, t=64, b=54),
        title=dict(text=title, x=0.02, xanchor="left", y=0.98),
        font=dict(
            family="-apple-system, BlinkMacSystemFont, Inter, Segoe UI, Roboto, Helvetica Neue, Arial",
            size=13,
            color="rgba(230,232,239,0.92)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=12),
        ),
        uirevision=UI_REV,
        autosize=True,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        ticks="outside",
        ticklen=6,
        tickcolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="rgba(230,232,239,0.86)", size=12),
        title_font=dict(color="rgba(230,232,239,0.92)", size=13),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        ticks="outside",
        ticklen=6,
        tickcolor="rgba(255,255,255,0.12)",
        tickfont=dict(color="rgba(230,232,239,0.86)", size=12),
        title_font=dict(color="rgba(230,232,239,0.92)", size=13),
    )

    # Better hover + line width for line charts (safe for bars too; bars ignore line.width)
    fig.update_traces(
        hoverlabel=dict(bgcolor="rgba(17,22,42,0.96)", font_color="#e6e8ef", bordercolor="rgba(255,255,255,0.10)"),
    )

    return fig


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


def safe_note(text: str) -> html.Div:
    return html.Div(text, className="note")


def df_preview_table(df: pd.DataFrame, max_rows: int = 12) -> html.Div:
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
            html.Div(f"Showing top {min(max_rows, len(df))} rows of {len(df)}", className="table-note"),
        ]
    )


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
    return " • ".join(parts) if parts else "No CSV files found yet (run notebook exports)."


# -----------------------------
# Figures
# -----------------------------
def fig_eda_timeseries(df: pd.DataFrame):
    df2 = df.copy()
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")

    ycol = "baskets" if "baskets" in df2.columns else df2.columns[1]
    fig = px.line(
        df2.sort_values("date"),
        x="date",
        y=ycol,
        markers=False,
        labels={"date": "Date", ycol: "Number of baskets"},
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(height=GRAPH_H_SM)
    return apply_plotly_theme(fig, title="Basket volume over time")


def fig_top_categories(df: pd.DataFrame):
    x = "orders" if "orders" in df.columns else df.columns[1]
    y = "category" if "category" in df.columns else df.columns[0]
    fig = px.bar(
        df.sort_values(x, ascending=True).tail(15),
        x=x,
        y=y,
        orientation="h",
        labels={x: "Unique baskets", y: "Category"},
    )
    fig.update_layout(height=GRAPH_H_MD)
    return apply_plotly_theme(fig, title="Top categories by basket participation")


def fig_feature_importance(df: pd.DataFrame):
    df2 = df.copy()
    if {"feature", "importance"}.issubset(df2.columns):
        df2 = df2.sort_values("importance", ascending=True).tail(12)
        fig = px.bar(
            df2,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance (gain/split proxy)", "feature": "Feature"},
        )
        return apply_plotly_theme(fig, title="Ranker feature importance (LightGBM)")

    fig = px.bar(df2.head(12))
    return apply_plotly_theme(fig, title="Feature importance")


def fig_ranking(df: pd.DataFrame, metric: str):
    y_map = {"ndcg": "ndcg", "hit_rate": "hit_rate"}
    y = y_map[metric]

    title = (
        "NDCG@K — Validation (higher is better)"
        if metric == "ndcg"
        else "HitRate@K — Validation (higher is better)"
    )

    x_label = "K (Top-K cutoff)"
    y_label = "NDCG@K" if metric == "ndcg" else "HitRate@K"

    df2 = df.copy()
    if "k" in df2.columns:
        df2["k"] = pd.to_numeric(df2["k"], errors="coerce")

    fig = px.line(
        df2.sort_values("k"),
        x="k",
        y=y,
        color="model" if "model" in df2.columns else None,
        markers=True,
        labels={"k": x_label, y: y_label, "model": "Model"},
    )
    fig.update_traces(line=dict(width=3))
    return apply_plotly_theme(fig, title=title)


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
                        html.Div("BASKET AI", className="eyebrow"),
                        html.H1("Executive Recommendation Dashboard"),
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

        html.Div(pill(f"Data sources: {data_source_line()}"), className="topbar"),

        # Lock content area min height to prevent "page jumping" when switching views
        html.Div(id="content", className="content content-lock"),

        html.Div("Local demo • Dash + Plotly • Dark Glass UI", className="footer"),
    ],
)


# -----------------------------
# Pages
# -----------------------------
def page_summary() -> html.Div:
    kpi_ndcg = "—"
    kpi_hit = "—"

    df = DATA.get("ranking")
    if df is not None and {"k", "ndcg", "hit_rate"}.issubset(df.columns):
        df2 = df.copy()
        df2["k"] = pd.to_numeric(df2["k"], errors="coerce")
        # Prefer K=10; otherwise smallest available K
        row10 = df2[df2["k"] == 10]
        if len(row10) >= 1:
            kpi_ndcg = f"{float(row10['ndcg'].iloc[0]):.3f}"
            kpi_hit = f"{float(row10['hit_rate'].iloc[0]):.3f}"
        else:
            row = df2.sort_values("k").head(1)
            kpi_ndcg = f"{float(row['ndcg'].iloc[0]):.3f}"
            kpi_hit = f"{float(row['hit_rate'].iloc[0]):.3f}"

    left = card(
        [
            html.Div(
                "This dashboard summarizes an offline evaluation for a basket-based recommender. "
                "The pipeline generates candidate items from multiple signals, then a LightGBM ranker orders them. "
                "We evaluate whether the true held-out item appears in the recommended list and how high it ranks.",
                className="p",
            ),
            html.Ul(
                [
                    html.Li(
                        html.Span(
                            [
                                html.B("NDCG@K"),
                                " measures ranking quality. It rewards placing the true item near the top (position-aware).",
                            ]
                        )
                    ),
                    html.Li(
                        html.Span(
                            [
                                html.B("HitRate@K"),
                                " measures coverage/recall. It checks whether the true item appears anywhere within Top-K.",
                            ]
                        )
                    ),
                    html.Li("Metrics are computed per basket_id and averaged across the validation split."),
                ],
                className="bullets",
            ),
            safe_note(
                "Next upgrades: baseline vs ranker deltas, segment slices (category/price band), "
                "and business proxies (CTR uplift × margin) to translate model gains into expected value."
            ),
        ],
        title="Executive Summary",
    )

    right = card(
        [
            html.Div("Key KPIs (Validation)", className="card-title-mini"),
            html.Div(
                [
                    kpi_card("NDCG@10", kpi_ndcg, "Ranking quality (higher is better)"),
                    kpi_card("HitRate@10", kpi_hit, "Coverage / recall proxy (higher is better)"),
                ],
                className="kpi-grid",
            ),
            html.Div(
                "Interpretation: NDCG@10 answers “How well are we ordering the list?”, while HitRate@10 answers "
                "“Did we include the right item at all?”",
                className="kpi-foot",
            ),
        ],
        title=None,
    )

    return html.Div(className="grid", children=[left, right])


def page_eda() -> html.Div:
    ts = DATA.get("eda_ts")
    cats = DATA.get("top_cat")

    left_children = []
    if ts is not None and ("date" in ts.columns):
        fig = fig_eda_timeseries(ts)
        fig.update_layout(height=GRAPH_H_SM)
        left_children += [
            dcc.Graph(figure=fig, config=GRAPH_CONFIG, className="graph"),
            safe_note("Use this to spot seasonality, campaign spikes, and potential data drift."),
        ]
    else:
        left_children += [safe_note("eda_timeseries.csv not found or missing expected columns (date, baskets).")]

    right_children = []
    if cats is not None and (("category" in cats.columns) or (cats.shape[1] >= 2)):
        right_children += [
            dcc.Graph(figure=fig_top_categories(cats), config=GRAPH_CONFIG, className="graph"),
            safe_note("Concentration indicates where cross-sell and uplift are likely to come from."),
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
            safe_note("High lift + good confidence suggests strong co-purchase structure worth merchandising."),
        ]
    else:
        left_children += [safe_note("rules_top.csv not found or missing expected columns (antecedent, consequent).")]

    right_children = []
    if pairs is not None and {"item_a", "item_b"}.issubset(pairs.columns):
        show_cols = [c for c in ["item_a", "item_b", "pair_count", "lift", "confidence", "support"] if c in pairs.columns]
        right_children += [
            df_preview_table(pairs[show_cols], max_rows=12),
            safe_note("Top co-occurring pairs support bundle ideas, placement decisions, and campaign planning."),
        ]
    else:
        right_children += [safe_note("cooc_top_pairs.csv not found or missing expected columns (item_a, item_b).")]

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

    top_children = []
    if feat is not None and {"feature", "importance"}.issubset(feat.columns):
        top_children += [
            dcc.Graph(figure=fig_feature_importance(feat), config=GRAPH_CONFIG, className="graph"),
            safe_note("Use this to understand which signals the ranker relies on and to monitor drift over time."),
        ]
    else:
        top_children += [safe_note("feature_importance.csv not found or missing expected columns (feature, importance).")]

    bottom_children = []
    if ranking is not None and {"k", "ndcg", "hit_rate"}.issubset(ranking.columns):
        bottom_children += [
            dcc.Graph(figure=fig_ranking(ranking, "ndcg"), config=GRAPH_CONFIG, className="graph"),
            safe_note("NDCG@K is position-aware: improvements at Top-5/Top-10 usually matter most for UX."),
            dcc.Graph(figure=fig_ranking(ranking, "hit_rate"), config=GRAPH_CONFIG, className="graph"),
            safe_note("HitRate@K captures whether the true item appears anywhere in Top-K (coverage/recall proxy)."),
        ]
    else:
        bottom_children += [safe_note("ranking_metrics.csv not found or missing expected columns (k, ndcg, hit_rate).")]

    return html.Div(
        className="stack",
        children=[
            card(top_children, title="Model Diagnostics — Ranker behavior"),
            card(bottom_children, title="Ranking Quality — NDCG & HitRate"),
        ],
    )


@app.callback(Output("content", "children"), Input("view", "value"))
def render(view: str):
    if view == "eda":
        return page_eda()
    if view == "insights":
        return page_insights()
    if view == "model":
        return page_model()
    return page_summary()


if __name__ == "__main__":
    # Dash 3: run_server -> run
    app.run(debug=True)
    