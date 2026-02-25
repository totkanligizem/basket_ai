from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

APP_DIR = Path(__file__).resolve().parent
ANALYST_DATA_DIR = APP_DIR / "data" / "analyst"

DAILY_CSV = ANALYST_DATA_DIR / "daily_metrics.csv"
CATEGORY_DAILY_CSV = ANALYST_DATA_DIR / "category_daily_metrics.csv"
CATEGORY_CITY_DAILY_CSV = ANALYST_DATA_DIR / "category_city_daily_metrics.csv"
CITY_DAILY_CSV = ANALYST_DATA_DIR / "city_daily_metrics.csv"
QUALITY_CSV = ANALYST_DATA_DIR / "quality_daily.csv"
TOP_ITEMS_CSV = ANALYST_DATA_DIR / "top_items.csv"
TOP_ITEMS_DAILY_CSV = ANALYST_DATA_DIR / "top_items_daily.csv"
BASKET_SCOPE_CSV = ANALYST_DATA_DIR / "basket_scope.csv"
BASKET_CATEGORY_BRIDGE_CSV = ANALYST_DATA_DIR / "basket_category_bridge.csv"

UI_REV = "analyst_dash_v2"
GRAPH_CONFIG = {"displayModeBar": False, "scrollZoom": False, "responsive": False}
COLORS = ["#0b6efd", "#0ea5a4", "#f59e0b", "#d6336c", "#198754", "#6c757d"]

GRAPH_HEIGHT_SM = 320
GRAPH_HEIGHT_MD = 380
GRAPH_HEIGHT_LG = 430

VIEW_OPTIONS = [
    {"label": "Executive Overview", "value": "exec"},
    {"label": "Category & Item Performance", "value": "category"},
    {"label": "Data Quality & Health", "value": "quality"},
]

QUALITY_LABELS = {
    "missing_item_code_rate": "Missing Item Code",
    "missing_category_rate": "Missing Category",
    "missing_price_rate": "Missing Price",
    "missing_amount_rate": "Missing Quantity",
}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if isinstance(df, pd.DataFrame) and not df.empty else None


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _normalize_text(series: pd.Series, default: str = "__UNKNOWN__") -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    out = out.mask(out.eq(""), default)
    return out


def _normalize_id(series: pd.Series) -> pd.Series:
    s = _normalize_text(series, default="")
    return s.str.replace(r"\.0$", "", regex=True)


def _load_data() -> dict[str, pd.DataFrame | None]:
    data = {
        "daily": _read_csv(DAILY_CSV),
        "category_daily": _read_csv(CATEGORY_DAILY_CSV),
        "category_city_daily": _read_csv(CATEGORY_CITY_DAILY_CSV),
        "city_daily": _read_csv(CITY_DAILY_CSV),
        "quality": _read_csv(QUALITY_CSV),
        "items": _read_csv(TOP_ITEMS_CSV),
        "items_daily": _read_csv(TOP_ITEMS_DAILY_CSV),
        "basket_scope": _read_csv(BASKET_SCOPE_CSV),
        "basket_bridge": _read_csv(BASKET_CATEGORY_BRIDGE_CSV),
    }

    daily = data["daily"]
    if daily is not None and {"date", "baskets", "revenue"}.issubset(daily.columns):
        daily = daily.copy()
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
        daily = _to_numeric(
            daily,
            [
                "baskets",
                "customers",
                "revenue",
                "total_items",
                "avg_distinct_items",
                "avg_category_count",
                "aov",
                "avg_items_per_basket",
            ],
        )
        data["daily"] = daily.dropna(subset=["date"]).sort_values("date")
    else:
        data["daily"] = None

    category_daily = data["category_daily"]
    if category_daily is not None and {"date", "category", "revenue"}.issubset(category_daily.columns):
        category_daily = category_daily.copy()
        category_daily["date"] = pd.to_datetime(category_daily["date"], errors="coerce")
        category_daily["category"] = _normalize_text(category_daily["category"])
        category_daily = _to_numeric(
            category_daily,
            ["revenue", "baskets", "quantity", "line_count", "aov", "items_per_basket"],
        )
        data["category_daily"] = category_daily.dropna(subset=["date"]).sort_values(["date", "revenue"], ascending=[True, False])
    else:
        data["category_daily"] = None

    category_city_daily = data["category_city_daily"]
    if category_city_daily is not None and {"date", "city", "category", "revenue"}.issubset(category_city_daily.columns):
        category_city_daily = category_city_daily.copy()
        category_city_daily["date"] = pd.to_datetime(category_city_daily["date"], errors="coerce")
        category_city_daily["city"] = _normalize_text(category_city_daily["city"])
        category_city_daily["category"] = _normalize_text(category_city_daily["category"])
        category_city_daily = _to_numeric(
            category_city_daily,
            ["revenue", "baskets", "quantity", "line_count", "aov", "items_per_basket"],
        )
        data["category_city_daily"] = category_city_daily.dropna(subset=["date"]).sort_values(["date", "revenue"], ascending=[True, False])
    else:
        data["category_city_daily"] = None

    city_daily = data["city_daily"]
    if city_daily is not None and {"date", "city", "revenue"}.issubset(city_daily.columns):
        city_daily = city_daily.copy()
        city_daily["date"] = pd.to_datetime(city_daily["date"], errors="coerce")
        city_daily["city"] = _normalize_text(city_daily["city"])
        city_daily = _to_numeric(city_daily, ["baskets", "customers", "revenue", "aov"])
        data["city_daily"] = city_daily.dropna(subset=["date"]).sort_values(["date", "revenue"], ascending=[True, False])
    else:
        data["city_daily"] = None

    quality = data["quality"]
    if quality is not None and "date" in quality.columns:
        quality = quality.copy()
        quality["date"] = pd.to_datetime(quality["date"], errors="coerce")
        quality = _to_numeric(
            quality,
            [
                "missing_item_code_rate",
                "missing_category_rate",
                "missing_price_rate",
                "missing_amount_rate",
            ],
        )
        data["quality"] = quality.dropna(subset=["date"]).sort_values("date")
    else:
        data["quality"] = None

    items = data["items"]
    if items is not None and {"itemcode", "revenue"}.issubset(items.columns):
        items = items.copy()
        items["itemcode"] = _normalize_id(items["itemcode"])
        if "category" in items.columns:
            items["category"] = _normalize_text(items["category"])
        if "item_name" in items.columns:
            items["item_name"] = _normalize_text(items["item_name"], default="")
        items = _to_numeric(items, ["revenue", "baskets", "quantity", "avg_unit_price", "line_count"])
        data["items"] = items.sort_values("revenue", ascending=False)
    else:
        data["items"] = None

    items_daily = data["items_daily"]
    if items_daily is not None and {"date", "itemcode", "revenue"}.issubset(items_daily.columns):
        items_daily = items_daily.copy()
        items_daily["date"] = pd.to_datetime(items_daily["date"], errors="coerce")
        items_daily["itemcode"] = _normalize_id(items_daily["itemcode"])
        if "city" in items_daily.columns:
            items_daily["city"] = _normalize_text(items_daily["city"])
        if "category" in items_daily.columns:
            items_daily["category"] = _normalize_text(items_daily["category"])
        if "item_name" in items_daily.columns:
            items_daily["item_name"] = _normalize_text(items_daily["item_name"], default="")
        items_daily = _to_numeric(items_daily, ["revenue", "baskets", "quantity", "avg_unit_price", "line_count"])
        data["items_daily"] = items_daily.dropna(subset=["date"]).sort_values(["date", "revenue"], ascending=[True, False])
    else:
        data["items_daily"] = None

    basket_scope = data["basket_scope"]
    if basket_scope is not None and {"basket_id", "date", "city", "revenue"}.issubset(basket_scope.columns):
        basket_scope = basket_scope.copy()
        basket_scope["basket_id"] = _normalize_id(basket_scope["basket_id"])
        basket_scope["date"] = pd.to_datetime(basket_scope["date"], errors="coerce")
        basket_scope["city"] = _normalize_text(basket_scope["city"])
        basket_scope["customer_id"] = _normalize_text(basket_scope.get("customer_id", pd.Series(["" for _ in range(len(basket_scope))])), default="__UNKNOWN__")
        basket_scope = _to_numeric(basket_scope, ["revenue", "total_items", "distinct_items"])
        data["basket_scope"] = basket_scope.dropna(subset=["date"]).sort_values(["date", "basket_id"])
    else:
        data["basket_scope"] = None

    basket_bridge = data["basket_bridge"]
    if basket_bridge is not None and {"basket_id", "date", "city", "category"}.issubset(basket_bridge.columns):
        basket_bridge = basket_bridge.copy()
        basket_bridge["basket_id"] = _normalize_id(basket_bridge["basket_id"])
        basket_bridge["date"] = pd.to_datetime(basket_bridge["date"], errors="coerce")
        basket_bridge["city"] = _normalize_text(basket_bridge["city"])
        basket_bridge["category"] = _normalize_text(basket_bridge["category"])
        data["basket_bridge"] = basket_bridge.dropna(subset=["date"]).drop_duplicates(["basket_id", "date", "city", "category"])
    else:
        data["basket_bridge"] = None

    return data


DATA = _load_data()


def _date_bounds() -> tuple[str | None, str | None]:
    daily = DATA.get("daily")
    if daily is None or daily.empty:
        return None, None
    start = daily["date"].min()
    end = daily["date"].max()
    if pd.isna(start) or pd.isna(end):
        return None, None
    return start.date().isoformat(), end.date().isoformat()


def _options(df: pd.DataFrame | None, col: str) -> list[dict[str, str]]:
    if df is None or col not in df.columns:
        return []
    values = sorted(v for v in df[col].dropna().astype(str).unique().tolist() if v.strip())
    return [{"label": v, "value": v} for v in values]


def _first_df(*frames: pd.DataFrame | None) -> pd.DataFrame | None:
    for frame in frames:
        if frame is not None:
            return frame
    return None


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _date_slice(
    df: pd.DataFrame | None,
    start_date: str | None,
    end_date: str | None,
    date_col: str = "date",
) -> pd.DataFrame | None:
    if df is None or df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    if start_date:
        start = pd.to_datetime(start_date, errors="coerce")
        if not pd.isna(start):
            out = out[out[date_col] >= start]
    if end_date:
        end = pd.to_datetime(end_date, errors="coerce")
        if not pd.isna(end):
            out = out[out[date_col] <= end]
    return out


def _daily_from_scope(scope: pd.DataFrame | None) -> pd.DataFrame | None:
    if scope is None or scope.empty:
        return None
    out = (
        scope.groupby("date", as_index=False)
        .agg(
            baskets=("basket_id", "nunique"),
            customers=("customer_id", "nunique"),
            revenue=("revenue", "sum"),
            total_items=("total_items", "sum"),
            avg_distinct_items=("distinct_items", "mean"),
        )
        .sort_values("date")
    )
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)
    out["avg_items_per_basket"] = out["total_items"] / out["baskets"].clip(lower=1)
    return out


def _city_daily_from_scope(scope: pd.DataFrame | None) -> pd.DataFrame | None:
    if scope is None or scope.empty:
        return None
    out = (
        scope.groupby(["date", "city"], as_index=False)
        .agg(
            baskets=("basket_id", "nunique"),
            customers=("customer_id", "nunique"),
            revenue=("revenue", "sum"),
        )
        .sort_values(["date", "revenue"], ascending=[True, False])
    )
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)
    return out


def _category_daily_from_cc(
    category_city_daily: pd.DataFrame | None,
    categories: list[str],
    cities: list[str],
    valid_dates: set[pd.Timestamp],
) -> pd.DataFrame | None:
    if category_city_daily is None or category_city_daily.empty:
        return None

    df = category_city_daily.copy()
    if categories:
        df = df[df["category"].isin(categories)]
    if cities:
        df = df[df["city"].isin(cities)]
    if valid_dates:
        df = df[df["date"].dt.normalize().isin(valid_dates)]
    if df.empty:
        return df

    out = (
        df.groupby(["date", "category"], as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            baskets=("baskets", "sum"),
            quantity=("quantity", "sum"),
            line_count=("line_count", "sum"),
        )
        .sort_values(["date", "revenue"], ascending=[True, False])
    )
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)
    out["items_per_basket"] = out["quantity"] / out["baskets"].clip(lower=1)
    return out


def _items_from_daily(
    items_daily: pd.DataFrame | None,
    items_fallback: pd.DataFrame | None,
    start_date: str | None,
    end_date: str | None,
    categories: list[str],
    cities: list[str],
    valid_dates: set[pd.Timestamp],
) -> pd.DataFrame | None:
    if items_daily is not None and not items_daily.empty:
        df = _date_slice(items_daily, start_date, end_date)
        if df is None or df.empty:
            return None
        if categories and "category" in df.columns:
            df = df[df["category"].isin(categories)]
        if cities and "city" in df.columns:
            df = df[df["city"].isin(cities)]
        if valid_dates:
            df = df[df["date"].dt.normalize().isin(valid_dates)]
        if df.empty:
            return df

        out = (
            df.groupby(["itemcode", "item_name", "category"], as_index=False)
            .agg(
                revenue=("revenue", "sum"),
                baskets=("baskets", "sum"),
                quantity=("quantity", "sum"),
                line_count=("line_count", "sum"),
            )
            .sort_values("revenue", ascending=False)
        )
        out["avg_unit_price"] = out["revenue"] / out["quantity"].where(out["quantity"] > 0)
        return out

    items = items_fallback.copy() if items_fallback is not None else None
    if items is None or items.empty:
        return items
    if categories and "category" in items.columns:
        items = items[items["category"].isin(categories)]
    return items


def _filtered_frames(
    start_date: str | None,
    end_date: str | None,
    categories: list[str],
    cities: list[str],
    min_daily_baskets: int | float | None,
) -> dict[str, pd.DataFrame | None]:
    basket_scope = _date_slice(DATA.get("basket_scope"), start_date, end_date)
    bridge = _date_slice(DATA.get("basket_bridge"), start_date, end_date)

    if basket_scope is not None and cities:
        basket_scope = basket_scope[basket_scope["city"].isin(cities)]
    if bridge is not None and cities:
        bridge = bridge[bridge["city"].isin(cities)]

    if categories and basket_scope is not None and bridge is not None:
        basket_ids = set(bridge[bridge["category"].isin(categories)]["basket_id"].astype(str).tolist())
        basket_scope = basket_scope[basket_scope["basket_id"].astype(str).isin(basket_ids)]

    daily = _daily_from_scope(basket_scope)
    if daily is None:
        daily = _date_slice(DATA.get("daily"), start_date, end_date)

    if daily is not None and "baskets" in daily.columns and min_daily_baskets is not None:
        daily = daily[daily["baskets"].fillna(0) >= float(min_daily_baskets)]

    valid_dates: set[pd.Timestamp] = set()
    if daily is not None and not daily.empty:
        valid_dates = set(pd.to_datetime(daily["date"], errors="coerce").dt.normalize().dropna().tolist())

    # Keep basket scope aligned to min basket threshold dates.
    if basket_scope is not None and valid_dates:
        basket_scope = basket_scope[basket_scope["date"].dt.normalize().isin(valid_dates)]

    city_daily = _city_daily_from_scope(basket_scope)
    if city_daily is None:
        city_daily = _date_slice(DATA.get("city_daily"), start_date, end_date)
        if city_daily is not None and cities:
            city_daily = city_daily[city_daily["city"].isin(cities)]
    if city_daily is not None and valid_dates:
        city_daily = city_daily[city_daily["date"].dt.normalize().isin(valid_dates)]

    category_daily = _category_daily_from_cc(
        category_city_daily=_date_slice(DATA.get("category_city_daily"), start_date, end_date),
        categories=categories,
        cities=cities,
        valid_dates=valid_dates,
    )
    if category_daily is None:
        category_daily = _date_slice(DATA.get("category_daily"), start_date, end_date)
        if category_daily is not None and categories:
            category_daily = category_daily[category_daily["category"].isin(categories)]
        if category_daily is not None and valid_dates:
            category_daily = category_daily[category_daily["date"].dt.normalize().isin(valid_dates)]

    quality = _date_slice(DATA.get("quality"), start_date, end_date)
    if quality is not None and valid_dates:
        quality = quality[quality["date"].dt.normalize().isin(valid_dates)]

    items = _items_from_daily(
        items_daily=DATA.get("items_daily"),
        items_fallback=DATA.get("items"),
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        cities=cities,
        valid_dates=valid_dates,
    )

    # Keep all panels in sync when min basket threshold removes all active days.
    if daily is not None and daily.empty:
        if category_daily is not None:
            category_daily = category_daily.iloc[0:0]
        if city_daily is not None:
            city_daily = city_daily.iloc[0:0]
        if quality is not None:
            quality = quality.iloc[0:0]
        if items is not None:
            items = items.iloc[0:0]

    return {
        "daily": daily,
        "category_daily": category_daily,
        "city_daily": city_daily,
        "quality": quality,
        "items": items,
    }


def _category_totals(category_daily: pd.DataFrame | None) -> pd.DataFrame | None:
    if category_daily is None or category_daily.empty:
        return None
    out = (
        category_daily.groupby("category", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            baskets=("baskets", "sum"),
            quantity=("quantity", "sum"),
            line_count=("line_count", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )
    total_revenue = out["revenue"].sum()
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)
    out["revenue_share"] = out["revenue"] / total_revenue if total_revenue > 0 else 0.0
    out["cum_revenue_share"] = out["revenue_share"].cumsum()
    return out


def _city_totals(city_daily: pd.DataFrame | None) -> pd.DataFrame | None:
    if city_daily is None or city_daily.empty:
        return None
    out = (
        city_daily.groupby("city", as_index=False)
        .agg(
            revenue=("revenue", "sum"),
            baskets=("baskets", "sum"),
            avg_daily_customers=("customers", "mean"),
            active_days=("date", "nunique"),
        )
        .sort_values("revenue", ascending=False)
    )
    out["aov"] = out["revenue"] / out["baskets"].clip(lower=1)
    return out


def _fmt_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{int(round(float(value))):,}"


def _fmt_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    num = float(value)
    if abs(num) >= 1_000_000:
        return f"TRY {num / 1_000_000:.2f}M"
    if abs(num) >= 1_000:
        return f"TRY {num / 1_000:.1f}K"
    return f"TRY {num:,.0f}"


def _fmt_float(value: float | int | None, precision: int = 2) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{precision}f}"


def _short_num(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _short_label(text: str, max_len: int = 16) -> str:
    t = str(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "..."


def _apply_figure_style(fig: go.Figure, title: str, height: int = GRAPH_HEIGHT_MD) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=height,
        margin=dict(l=56, r=20, t=62, b=52),
        title=dict(text=title, x=0.01, xanchor="left"),
        font=dict(family="IBM Plex Sans, Segoe UI, Arial, sans-serif", size=12, color="#0f172a"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        uirevision=UI_REV,
        autosize=False,
        transition={"duration": 0},
        colorway=COLORS,
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
    return _apply_figure_style(fig, title=title, height=GRAPH_HEIGHT_SM)


def graph_block(fig: go.Figure, size: str = "md") -> dcc.Graph:
    size_to_height = {
        "sm": GRAPH_HEIGHT_SM,
        "md": GRAPH_HEIGHT_MD,
        "lg": GRAPH_HEIGHT_LG,
    }
    height = size_to_height.get(size, GRAPH_HEIGHT_MD)
    return dcc.Graph(
        figure=fig,
        config=GRAPH_CONFIG,
        className=f"graph graph-{size}",
        style={"height": f"{height}px"},
    )


def _fig_revenue_vs_baskets(daily: pd.DataFrame | None) -> go.Figure:
    if daily is None or daily.empty:
        return _empty_figure("Revenue vs Basket Volume", "No daily metrics for selected filters")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["revenue"],
            mode="lines",
            name="Revenue",
            line=dict(width=3, color="#0ea5a4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["baskets"],
            mode="lines",
            name="Basket Count",
            line=dict(width=2.5, color="#0b6efd"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Revenue (TRY)", tickprefix="TRY ", separatethousands=True),
        yaxis2=dict(
            title="Baskets (#)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            separatethousands=True,
        ),
    )
    return _apply_figure_style(fig, title="Revenue vs Basket Volume Over Time")


def _fig_value_vs_diversity(daily: pd.DataFrame | None) -> go.Figure:
    if daily is None or daily.empty:
        return _empty_figure("Basket Value vs Diversity", "No daily metrics for selected filters")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["aov"],
            mode="lines",
            name="Average Order Value",
            line=dict(width=3, color="#f59e0b"),
        )
    )
    metric = "avg_distinct_items" if "avg_distinct_items" in daily.columns else "avg_items_per_basket"
    metric_label = "Avg Distinct Items per Basket" if metric == "avg_distinct_items" else "Avg Items per Basket"
    fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily[metric],
            mode="lines",
            name=metric_label,
            line=dict(width=2.5, color="#6f42c1"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="AOV (TRY)", tickprefix="TRY ", separatethousands=True),
        yaxis2=dict(
            title="Items per Basket (#)",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
    )
    return _apply_figure_style(fig, title="Basket Value vs Basket Diversity")


def _fig_category_pareto(category_totals: pd.DataFrame | None) -> go.Figure:
    if category_totals is None or category_totals.empty:
        return _empty_figure("Category Pareto", "No category rows for selected filters")

    df = category_totals.head(12).copy()
    df["cum_pct"] = (df["revenue"].cumsum() / df["revenue"].sum()).fillna(0.0)
    df["label"] = df["category"].map(lambda x: _short_label(x, max_len=18))

    fig = go.Figure()
    fig.add_bar(x=df["label"], y=df["revenue"], name="Revenue", marker_color="#0b6efd")
    fig.add_trace(
        go.Scatter(
            x=df["label"],
            y=df["cum_pct"],
            mode="lines+markers",
            name="Cumulative Revenue Share",
            line=dict(width=3, color="#d6336c"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Revenue (TRY)", tickprefix="TRY ", separatethousands=True),
        yaxis2=dict(
            title="Cumulative Share (%)",
            overlaying="y",
            side="right",
            tickformat=".0%",
            range=[0, 1.05],
            showgrid=False,
        ),
        xaxis=dict(tickangle=-18, title="Category"),
    )
    fig.add_hline(y=0.8, yref="y2", line_dash="dash", line_color="#64748b")
    return _apply_figure_style(fig, title="Revenue Concentration by Category (Pareto)")


def _fig_category_revenue(category_totals: pd.DataFrame | None) -> go.Figure:
    if category_totals is None or category_totals.empty:
        return _empty_figure("Category Revenue", "No category rows for selected filters")
    df = category_totals.sort_values("revenue", ascending=True).tail(12).copy()
    df["label"] = df["category"].map(lambda x: _short_label(x, max_len=26))
    fig = px.bar(
        df,
        x="revenue",
        y="label",
        orientation="h",
        labels={"revenue": "Revenue (TRY)", "label": "Category"},
    )
    fig.update_traces(marker_color="#0ea5a4")
    fig.update_xaxes(tickprefix="TRY ", separatethousands=True)
    return _apply_figure_style(fig, title="Top Categories by Revenue")


def _fig_top_items(items: pd.DataFrame | None) -> go.Figure:
    if items is None or items.empty:
        return _empty_figure("Top Items", "No item summary rows for selected filters")
    df = items.head(15).copy()
    labels = []
    for _, row in df.iterrows():
        item_name = str(row.get("item_name", "")).strip()
        item_code = str(row.get("itemcode", ""))
        base = item_name if item_name else item_code
        labels.append(_short_label(base, max_len=28))
    df["item_label"] = labels
    fig = px.bar(
        df.sort_values("revenue"),
        x="revenue",
        y="item_label",
        orientation="h",
        labels={"revenue": "Revenue (TRY)", "item_label": "Item"},
    )
    fig.update_traces(marker_color="#f59e0b")
    fig.update_xaxes(tickprefix="TRY ", separatethousands=True)
    return _apply_figure_style(fig, title="Top Items by Revenue")


def _fig_city_scatter(city_totals: pd.DataFrame | None) -> go.Figure:
    if city_totals is None or city_totals.empty:
        return _empty_figure("City Performance", "No city rows for selected filters")
    df = city_totals.head(30).copy()
    fig = px.scatter(
        df,
        x="baskets",
        y="revenue",
        size="avg_daily_customers",
        color="aov",
        hover_name="city",
        labels={
            "aov": "AOV (TRY)",
            "baskets": "Basket Count (#)",
            "revenue": "Revenue (TRY)",
            "avg_daily_customers": "Avg Daily Customers",
        },
        color_continuous_scale="Blues",
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="#e2e8f0")))
    fig.update_yaxes(tickprefix="TRY ", separatethousands=True)
    fig.update_xaxes(separatethousands=True)
    return _apply_figure_style(fig, title="City Revenue vs Basket Count")


def _quality_melt(quality: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in QUALITY_LABELS if c in quality.columns]
    non_zero_cols = [c for c in columns if quality[c].fillna(0).abs().sum() > 0]
    use_cols = non_zero_cols if non_zero_cols else columns
    melt = quality[["date"] + use_cols].melt(id_vars="date", var_name="metric", value_name="rate")
    melt["metric"] = melt["metric"].map(QUALITY_LABELS)
    melt["rate_pct"] = (melt["rate"] * 100).fillna(0.0)
    return melt


def _fig_quality_trend(quality: pd.DataFrame | None) -> go.Figure:
    if quality is None or quality.empty:
        return _empty_figure("Data Completeness", "No quality metrics for selected filters")

    melt = _quality_melt(quality)
    if melt.empty:
        return _empty_figure("Data Completeness", "Missing-rate columns are unavailable")

    fig = px.line(
        melt,
        x="date",
        y="rate_pct",
        color="metric",
        markers=False,
        labels={"date": "Date", "rate_pct": "Missing Rate (%)", "metric": "Metric"},
    )
    fig.update_layout(yaxis=dict(title="Missing Rate (%)"))
    return _apply_figure_style(fig, title="Data Completeness Trend")


def _fig_quality_latest(quality: pd.DataFrame | None) -> go.Figure:
    if quality is None or quality.empty:
        return _empty_figure("Latest Missing Rates", "No quality metrics for selected filters")

    tail = quality.sort_values("date").tail(14).copy()
    melt = _quality_melt(tail)
    if melt.empty:
        return _empty_figure("Latest Missing Rates", "Missing-rate columns are unavailable")

    fig = px.bar(
        melt,
        x="date",
        y="rate_pct",
        color="metric",
        barmode="group",
        labels={"date": "Date", "rate_pct": "Missing Rate (%)", "metric": "Metric"},
    )
    fig.update_layout(yaxis=dict(title="Missing Rate (%)"))
    return _apply_figure_style(fig, title="Last 14 Days Missing Rates")


def kpi_card(label: str, value: str, subtext: str) -> html.Div:
    return html.Div(
        className="kpi",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(subtext, className="kpi-sub"),
        ],
    )


def card(title: str, body: list, note: str | None = None) -> html.Div:
    children = [html.Div(title, className="card-title"), html.Div(body, className="card-body")]
    if note:
        children.append(html.Div(note, className="note"))
    return html.Div(children, className="card")


def _format_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, bool)):
        return f"{value:,}" if isinstance(value, int) else str(value)
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def preview_table(df: pd.DataFrame, max_rows: int = 12, max_height: int = 360) -> html.Div:
    view = df.head(max_rows).copy()
    return html.Div(
        [
            html.Div(
                html.Table(
                    [
                        html.Thead(html.Tr([html.Th(c) for c in view.columns])),
                        html.Tbody(
                            [
                                html.Tr([html.Td(_format_cell(view.iloc[i, j])) for j in range(view.shape[1])])
                                for i in range(view.shape[0])
                            ]
                        ),
                    ],
                    className="table",
                ),
                className="table-wrap table-wrap-scroll",
                style={"maxHeight": f"{max_height}px"},
            ),
            html.Div(f"Showing {min(max_rows, len(df))} of {len(df)} rows", className="table-note"),
        ]
    )


def page_exec(filtered: dict[str, pd.DataFrame | None]) -> html.Div:
    daily = filtered.get("daily")
    category_daily = filtered.get("category_daily")
    quality = filtered.get("quality")

    if daily is None or daily.empty:
        return card(
            "Executive Overview",
            [html.Div("No daily metrics found. Run analyst artifact export first.", className="note")],
        )

    total_revenue = daily["revenue"].sum()
    total_baskets = daily["baskets"].sum()
    aov = total_revenue / total_baskets if total_baskets > 0 else None

    weighted_distinct = None
    if "avg_distinct_items" in daily.columns:
        weighted_distinct = (daily["avg_distinct_items"] * daily["baskets"]).sum()
    avg_distinct = weighted_distinct / total_baskets if total_baskets > 0 and weighted_distinct is not None else None

    active_days = len(daily)
    category_totals = _category_totals(category_daily)

    kpis = html.Div(
        className="kpi-grid kpi-grid-4",
        children=[
            kpi_card("Total Revenue", _fmt_money(total_revenue), "Filtered period"),
            kpi_card("Total Baskets", _fmt_int(total_baskets), "Filtered period"),
            kpi_card("Average Order Value", _fmt_money(aov), "Revenue / baskets"),
            kpi_card("Avg Distinct Items / Basket", _fmt_float(avg_distinct, 2), f"{active_days} active days"),
        ],
    )

    top_row = html.Div(
        className="grid",
        children=[
            card("Revenue & Basket Volume", [graph_block(_fig_revenue_vs_baskets(daily), size="md")]),
            card("Basket Value & Diversity", [graph_block(_fig_value_vs_diversity(daily), size="md")]),
        ],
    )

    bottom_row = html.Div(
        className="grid",
        children=[
            card(
                "Revenue Concentration",
                [graph_block(_fig_category_pareto(category_totals), size="md")],
                note="Pareto line marks concentration and helps track category dependency.",
            ),
            card(
                "Data Completeness",
                [graph_block(_fig_quality_trend(quality), size="md")],
                note="Missing-rate trend highlights ingestion and mapping quality drift.",
            ),
        ],
    )

    return html.Div(className="stack", children=[kpis, top_row, bottom_row])


def page_category(filtered: dict[str, pd.DataFrame | None]) -> html.Div:
    category_daily = filtered.get("category_daily")
    items = filtered.get("items")
    city_daily = filtered.get("city_daily")

    category_totals = _category_totals(category_daily)
    city_totals = _city_totals(city_daily)

    if category_totals is None or category_totals.empty:
        return card(
            "Category & Item Performance",
            [html.Div("No category metrics found for selected filters.", className="note")],
        )

    category_table = category_totals.copy()
    for col in ["revenue", "aov", "revenue_share", "cum_revenue_share"]:
        if col in category_table.columns:
            category_table[col] = category_table[col].round(4)

    items_table = items.copy() if items is not None else pd.DataFrame()
    if not items_table.empty:
        for col in ["revenue", "avg_unit_price"]:
            if col in items_table.columns:
                items_table[col] = items_table[col].round(4)

    row_one = html.Div(
        className="grid",
        children=[
            card("Category Pareto", [graph_block(_fig_category_pareto(category_totals), size="md")]),
            card("Top Item Intelligence", [graph_block(_fig_top_items(items), size="md")]),
        ],
    )

    row_two = html.Div(
        className="grid",
        children=[
            card("Category Revenue View", [graph_block(_fig_category_revenue(category_totals), size="md")]),
            card(
                "City Performance",
                [graph_block(_fig_city_scatter(city_totals), size="md")],
                note="Bubble size represents average daily customers.",
            ),
        ],
    )

    row_three = html.Div(
        className="grid grid-start",
        children=[
            card("Category Summary Table", [preview_table(category_table, max_rows=14, max_height=360)]),
            card(
                "Top Items Table",
                [preview_table(items_table, max_rows=14, max_height=360)]
                if not items_table.empty
                else [html.Div("top_items data is missing", className="note")],
            ),
        ],
    )

    return html.Div(className="stack", children=[row_one, row_two, row_three])


def page_quality(filtered: dict[str, pd.DataFrame | None]) -> html.Div:
    quality = filtered.get("quality")
    daily = filtered.get("daily")

    if quality is None or quality.empty:
        return card(
            "Data Quality & Health",
            [html.Div("No quality metrics found for selected filters.", className="note")],
        )

    avg_missing_item = quality.get("missing_item_code_rate", pd.Series(dtype=float)).mean()
    avg_missing_cat = quality.get("missing_category_rate", pd.Series(dtype=float)).mean()
    avg_missing_price = quality.get("missing_price_rate", pd.Series(dtype=float)).mean()
    avg_missing_amt = quality.get("missing_amount_rate", pd.Series(dtype=float)).mean()

    kpis = html.Div(
        className="kpi-grid kpi-grid-4",
        children=[
            kpi_card("Avg Missing Item Code", f"{avg_missing_item * 100:.2f}%", "Daily mean"),
            kpi_card("Avg Missing Category", f"{avg_missing_cat * 100:.2f}%", "Daily mean"),
            kpi_card("Avg Missing Price", f"{avg_missing_price * 100:.2f}%", "Daily mean"),
            kpi_card("Avg Missing Quantity", f"{avg_missing_amt * 100:.2f}%", "Daily mean"),
        ],
    )

    chart_row = html.Div(
        className="grid",
        children=[
            card("Quality Trend", [graph_block(_fig_quality_trend(quality), size="md")]),
            card("Recent Quality Snapshot", [graph_block(_fig_quality_latest(quality), size="md")]),
        ],
    )

    quality_table = quality.sort_values("date", ascending=False).copy()
    for col in QUALITY_LABELS:
        if col in quality_table.columns:
            quality_table[col] = (quality_table[col] * 100).round(3)
    if "date" in quality_table.columns:
        quality_table["date"] = pd.to_datetime(quality_table["date"], errors="coerce").dt.date
    rename_map = {k: f"{v} (%)" for k, v in QUALITY_LABELS.items()}
    quality_table = quality_table.rename(columns=rename_map)

    table_card = card(
        "Quality Daily Table",
        [preview_table(quality_table, max_rows=20, max_height=420)],
        note="Rate columns are shown in percentage points.",
    )

    if daily is not None and not daily.empty:
        summary_text = f"Filtered days: {len(daily)} | Revenue sum: {_fmt_money(daily['revenue'].sum())}"
    else:
        summary_text = "Daily metric scope unavailable for current filters."

    summary = html.Div(summary_text, className="insight-strip")

    return html.Div(className="stack", children=[kpis, chart_row, table_card, summary])


def _source_status(filtered: dict[str, pd.DataFrame | None]) -> str:
    labels = {
        "daily": "daily_metrics",
        "category_daily": "category_daily",
        "city_daily": "city_daily",
        "quality": "quality_daily",
        "items": "top_items",
    }
    parts = []
    for key, label in labels.items():
        df = filtered.get(key)
        if df is None:
            parts.append(f"{label}: missing")
        else:
            parts.append(f"{label}: {len(df):,} rows")
    return " | ".join(parts)


def _filter_summary(
    start_date: str | None,
    end_date: str | None,
    categories: list[str],
    cities: list[str],
    min_daily_baskets: int | float | None,
) -> str:
    date_part = f"Date: {start_date or '-'} to {end_date or '-'}"
    category_part = "Categories: all" if not categories else f"Categories: {', '.join(categories[:3])}" + (" ..." if len(categories) > 3 else "")
    city_part = "Cities: all" if not cities else f"Cities: {', '.join(cities[:3])}" + (" ..." if len(cities) > 3 else "")
    min_part = f"Min daily baskets: {int(min_daily_baskets or 0):,}"
    return " | ".join([date_part, category_part, city_part, min_part])


DATE_START, DATE_END = _date_bounds()
CATEGORY_OPTIONS = _options(_first_df(DATA.get("category_city_daily"), DATA.get("category_daily")), "category")
CITY_OPTIONS = _options(_first_df(DATA.get("city_daily"), DATA.get("basket_scope")), "city")

daily_data = DATA.get("daily")
if daily_data is not None and "baskets" in daily_data.columns and not daily_data.empty:
    max_baskets = int(daily_data["baskets"].max())
    default_min_baskets = int(daily_data["baskets"].quantile(0.05))
else:
    max_baskets = 1
    default_min_baskets = 0

app = Dash(__name__, assets_folder=str(APP_DIR / "assets"), title="Basket AI Analyst Dashboard")

app.layout = html.Div(
    className="page analyst-page",
    children=[
        html.Header(
            className="header",
            children=[
                html.Div(
                    [
                        html.Div("BASKET AI", className="eyebrow"),
                        html.H1("Basket Intelligence Dashboard"),
                        html.Div(
                            "Business analytics layer for revenue, demand, category mix, and data quality.",
                            className="subtitle",
                        ),
                    ]
                ),
                html.Div(
                    className="controls controls-wide",
                    children=[
                        html.Div("View", className="control-label"),
                        dcc.Dropdown(
                            id="ana-view",
                            options=VIEW_OPTIONS,
                            value="exec",
                            clearable=False,
                            searchable=False,
                            className="dropdown",
                        ),
                        html.Div("Date Range", className="control-label control-label-spaced"),
                        dcc.DatePickerRange(
                            id="ana-date-range",
                            min_date_allowed=DATE_START,
                            max_date_allowed=DATE_END,
                            start_date=DATE_START,
                            end_date=DATE_END,
                            display_format="YYYY-MM-DD",
                            className="date-range",
                        ),
                        html.Div("Category Filter", className="control-label control-label-spaced"),
                        dcc.Dropdown(
                            id="ana-category-filter",
                            options=CATEGORY_OPTIONS,
                            value=[],
                            multi=True,
                            placeholder="All categories",
                            className="dropdown",
                        ),
                        html.Div("City Filter", className="control-label control-label-spaced"),
                        dcc.Dropdown(
                            id="ana-city-filter",
                            options=CITY_OPTIONS,
                            value=[],
                            multi=True,
                            placeholder="All cities",
                            className="dropdown",
                        ),
                        html.Div("Min Daily Baskets", className="control-label control-label-spaced"),
                        dcc.Slider(
                            id="ana-min-baskets",
                            min=0,
                            max=max(max_baskets, 1),
                            step=max(1, int(max(max_baskets, 1) / 50)),
                            value=default_min_baskets,
                            marks={0: "0", max(max_baskets, 1): _short_num(max(max_baskets, 1))},
                        ),
                    ],
                ),
            ],
        ),
        html.Div(id="ana-filter-summary", className="status-line filter-summary"),
        html.Div(id="ana-status", className="status-line"),
        html.Div(id="ana-content", className="content"),
        html.Div("Dash + Plotly | Analyst intelligence layer | Stable and consistent view", className="footer"),
    ],
)


@app.callback(
    Output("ana-content", "children"),
    Output("ana-status", "children"),
    Output("ana-filter-summary", "children"),
    Input("ana-view", "value"),
    Input("ana-date-range", "start_date"),
    Input("ana-date-range", "end_date"),
    Input("ana-category-filter", "value"),
    Input("ana-city-filter", "value"),
    Input("ana-min-baskets", "value"),
)
def render(
    view: str,
    start_date: str | None,
    end_date: str | None,
    category_filter: str | list[str] | None,
    city_filter: str | list[str] | None,
    min_daily_baskets: int | float | None,
):
    categories = _as_list(category_filter)
    cities = _as_list(city_filter)

    filtered = _filtered_frames(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        cities=cities,
        min_daily_baskets=min_daily_baskets,
    )

    if view == "category":
        content = page_category(filtered)
    elif view == "quality":
        content = page_quality(filtered)
    else:
        content = page_exec(filtered)

    status = _source_status(filtered)
    filter_summary = _filter_summary(
        start_date=start_date,
        end_date=end_date,
        categories=categories,
        cities=cities,
        min_daily_baskets=min_daily_baskets,
    )
    return content, status, filter_summary


if __name__ == "__main__":
    app.run(debug=False, dev_tools_hot_reload=False)
