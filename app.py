"""
Electric Vehicle (EV) Market Intelligence Dashboard
====================================================
A production-grade business intelligence application built with Streamlit,
Pandas, NumPy, and Plotly to transform EV market data into executive,
competitive, technological, geographic, and sustainability insights.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="EV Market Intelligence Dashboard",
    page_icon="\u26a1",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }
    .main-header {
        background: linear-gradient(135deg, #064E3B 0%, #047857 60%, #10B981 100%);
        color: #FFFFFF;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 12px rgba(6, 78, 59, 0.12);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        margin: 0 0 0.35rem 0;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #D1FAE5 !important;
        margin: 0;
        font-size: 0.98rem;
        line-height: 1.45;
    }
    .kpi-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #10B981;
        border-radius: 10px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        height: 100%;
        margin-bottom: 0.75rem;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #475569;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .kpi-unit {
        font-size: 0.8rem;
        font-weight: 500;
        color: #059669;
    }
    .kpi-desc {
        font-size: 0.75rem;
        color: #64748B;
        line-height: 1.3;
    }
    .insight-box {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-left: 5px solid #059669;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #064E3B;
    }
    .insight-box h4 {
        margin: 0 0 0.5rem 0;
        color: #065F46;
        font-size: 1rem;
        font-weight: 700;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.4rem;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

GREEN_PALETTE = [
    "#059669", "#10B981", "#0D9488", "#0284C7", "#2563EB",
    "#4F46E5", "#7C3AED", "#D97706", "#DC2626", "#475569",
]

BATTERY_COLORS = {
    "NMC": "#0284C7",
    "LFP": "#059669",
    "NCA": "#4F46E5",
    "Sodium-Ion": "#D97706",
    "Solid-State": "#7C3AED",
}

EXPECTED_COLUMNS = [
    "vehicle_id", "model_name", "manufacturer", "model_year",
    "manufacturing_country", "market_region", "category",
    "battery_capacity_kwh", "battery_type", "battery_cycle_life",
    "battery_degradation_pct_5yr", "range_km", "efficiency_wh_per_km",
    "motor_power_kw", "torque_nm", "num_motors", "drivetrain",
    "top_speed_kmph", "acceleration_0_100_sec", "max_dc_charging_kw",
    "charging_dc_fast_time_min", "charging_ac_time_hr", "num_charging_ports",
    "num_seats", "weight_kg", "cargo_capacity_liters", "price_usd",
    "insurance_cost_usd_yearly", "warranty_years", "resale_value_pct_3yr",
    "safety_rating_stars", "autonomous_level", "infotainment_screen_inch",
    "co2_saved_tons_per_year", "annual_sales_units", "customer_rating",
    "recommended_use_case",
]


def get_default_dataset_paths() -> List[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.join(script_dir, "ev_market_test.csv"),
        "ev_market_test.csv",
        os.path.join(os.path.dirname(script_dir), "ev_market_test.csv"),
    ]


@st.cache_data(show_spinner="Loading EV market dataset...")
def load_data(file_path_or_buffer: Any) -> pd.DataFrame:
    if isinstance(file_path_or_buffer, str):
        if file_path_or_buffer.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(file_path_or_buffer)
        return pd.read_csv(file_path_or_buffer)
    name = getattr(file_path_or_buffer, "name", "uploaded.csv").lower()
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(file_path_or_buffer)
    return pd.read_csv(file_path_or_buffer)


@st.cache_data(show_spinner="Validating and cleaning dataset...")
def clean_data(raw_df: pd.DataFrame, impute_missing: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = raw_df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    initial_rows = len(df)
    exact_duplicates = int(df.duplicated().sum())
    id_duplicates = int(df["vehicle_id"].duplicated().sum()) if "vehicle_id" in df.columns else 0
    missing_before = df.isnull().sum().to_dict()

    if exact_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    cat_cols = [
        "vehicle_id", "model_name", "manufacturer", "manufacturing_country",
        "market_region", "category", "battery_type", "drivetrain", "recommended_use_case",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    num_cols = [c for c in EXPECTED_COLUMNS if c not in cat_cols and c in df.columns]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    imputation_details: List[Dict[str, Any]] = []
    if impute_missing:
        group_imputations = {
            "resale_value_pct_3yr": ["category", "battery_type"],
            "battery_cycle_life": ["battery_type", "category"],
            "customer_rating": ["category"],
            "infotainment_screen_inch": ["category"],
            "annual_sales_units": ["category", "market_region"],
        }
        for col, group_keys in group_imputations.items():
            if col in df.columns:
                n_miss = int(df[col].isnull().sum())
                if n_miss > 0:
                    valid_keys = [k for k in group_keys if k in df.columns]
                    if valid_keys:
                        df[col] = df[col].fillna(df.groupby(valid_keys)[col].transform("median"))
                    df[col] = df[col].fillna(df[col].median())
                    imputation_details.append({
                        "column": col,
                        "missing_count": n_miss,
                        "missing_pct": round((n_miss / initial_rows) * 100, 2),
                        "strategy": f"Group median by {', '.join(valid_keys)} (fallback to global median)",
                    })
    else:
        for col in num_cols:
            n_miss = int(df[col].isnull().sum())
            if n_miss > 0:
                imputation_details.append({
                    "column": col,
                    "missing_count": n_miss,
                    "missing_pct": round((n_miss / initial_rows) * 100, 2),
                    "strategy": "Retained as NaN (excluded from aggregations)",
                })

    missing_after = df.isnull().sum().to_dict()

    if "price_usd" in df.columns and "range_km" in df.columns:
        df["price_per_km_range"] = np.where(df["range_km"] > 0, df["price_usd"] / df["range_km"], np.nan)
    if "price_usd" in df.columns and "battery_capacity_kwh" in df.columns:
        df["price_per_kwh"] = np.where(df["battery_capacity_kwh"] > 0, df["price_usd"] / df["battery_capacity_kwh"], np.nan)

    quality_report = {
        "initial_rows": initial_rows,
        "cleaned_rows": len(df),
        "total_columns": len(raw_df.columns),
        "exact_duplicates": exact_duplicates,
        "id_duplicates": id_duplicates,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "imputation_details": imputation_details,
        "impute_missing_enabled": impute_missing,
    }
    return df, quality_report


def filter_data(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    cat_filter_map = [
        ("model_years", "model_year"),
        ("market_regions", "market_region"),
        ("manufacturing_countries", "manufacturing_country"),
        ("categories", "category"),
        ("manufacturers", "manufacturer"),
        ("battery_types", "battery_type"),
        ("drivetrains", "drivetrain"),
        ("use_cases", "recommended_use_case"),
    ]
    for f_key, col in cat_filter_map:
        selected = filters.get(f_key)
        if selected is not None and col in df.columns:
            if len(selected) == 0:
                return df.iloc[0:0]
            mask &= df[col].isin(selected)

    range_filter_map = [
        ("price_range", "price_usd"),
        ("range_km_bounds", "range_km"),
        ("battery_kwh_bounds", "battery_capacity_kwh"),
        ("rating_bounds", "customer_rating"),
    ]
    for f_key, col in range_filter_map:
        bounds = filters.get(f_key)
        if bounds is not None and col in df.columns:
            low, high = bounds
            col_mask = (df[col] >= low) & (df[col] <= high)
            if not filters.get("impute_missing", True):
                col_mask |= df[col].isna()
            mask &= col_mask

    return df.loc[mask].copy()


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total_vehicles": 0, "total_manufacturers": 0, "total_annual_sales": 0.0,
            "avg_price": 0.0, "median_price": 0.0, "avg_range": 0.0,
            "avg_battery_kwh": 0.0, "avg_rating": 0.0, "avg_co2_saved": 0.0,
            "top_manufacturer": "N/A", "top_manufacturer_sales": 0.0,
            "top_category": "N/A", "top_category_sales": 0.0,
        }
    mfg_sales = df.groupby("manufacturer")["annual_sales_units"].sum()
    top_mfg = str(mfg_sales.idxmax()) if not mfg_sales.empty else "N/A"
    top_mfg_sales = float(mfg_sales.max()) if not mfg_sales.empty else 0.0

    cat_sales = df.groupby("category")["annual_sales_units"].sum()
    top_cat = str(cat_sales.idxmax()) if not cat_sales.empty else "N/A"
    top_cat_sales = float(cat_sales.max()) if not cat_sales.empty else 0.0

    return {
        "total_vehicles": int(len(df)),
        "total_manufacturers": int(df["manufacturer"].nunique()),
        "total_annual_sales": float(df["annual_sales_units"].sum()),
        "avg_price": float(df["price_usd"].mean()),
        "median_price": float(df["price_usd"].median()),
        "avg_range": float(df["range_km"].mean()),
        "avg_battery_kwh": float(df["battery_capacity_kwh"].mean()),
        "avg_rating": float(df["customer_rating"].mean()),
        "avg_co2_saved": float(df["co2_saved_tons_per_year"].mean()),
        "top_manufacturer": top_mfg,
        "top_manufacturer_sales": top_mfg_sales,
        "top_category": top_cat,
        "top_category_sales": top_cat_sales,
    }


def render_kpi_card(label: str, value: str, unit: str, desc: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value} <span class="kpi-unit">{unit}</span></div>
            <div class="kpi-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_fig(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=20, r=20, t=48, b=25),
        font=dict(family="Inter, sans-serif", size=12, color="#1E293B"),
        title_font=dict(size=15, color="#0F172A", family="Inter, sans-serif"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor="#CBD5E1")
    return fig


def render_executive_overview(df: pd.DataFrame, kpis: Dict[str, Any]) -> None:
    st.markdown('<div class="section-title">1. Executive Overview & Market Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi_card("Total Vehicles", f"{kpis['total_vehicles']:,}", "records", "Active EV configurations in filtered view")
    with c2:
        render_kpi_card("Manufacturers", f"{kpis['total_manufacturers']:,}", "brands", "Distinct automotive OEMs represented")
    with c3:
        render_kpi_card("Recorded Annual Sales", f"{kpis['total_annual_sales'] / 1e6:,.2f}M", "units", f"Exact: {kpis['total_annual_sales']:,.0f} units sold")
    with c4:
        render_kpi_card("Average Vehicle Price", f"${kpis['avg_price']:,.0f}", "USD", "Mean MSRP across filtered vehicles")
    with c5:
        render_kpi_card("Median Vehicle Price", f"${kpis['median_price']:,.0f}", "USD", "50th percentile price (robust to luxury/bus outliers)")

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        render_kpi_card("Avg Driving Range", f"{kpis['avg_range']:,.1f}", "km", "Mean single-charge WLTP/estimated range")
    with c7:
        render_kpi_card("Avg Battery Capacity", f"{kpis['avg_battery_kwh']:,.1f}", "kWh", "Mean gross battery pack energy capacity")
    with c8:
        render_kpi_card("Avg Customer Rating", f"{kpis['avg_rating']:.2f}", "/ 5.0 \u2605", "Mean verified owner satisfaction score")
    with c9:
        render_kpi_card("Avg CO\u2082 Saved", f"{kpis['avg_co2_saved']:.2f}", "tons/yr", "Mean annual tailpipe CO\u2082 reduction per vehicle")
    with c10:
        render_kpi_card("Top OEM & Segment", f"{kpis['top_manufacturer']}", f"| {kpis['top_category']}", f"Leaders by recorded unit sales ({kpis['top_manufacturer_sales']:,.0f} units)")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        yearly_sales = df.groupby("model_year", as_index=False)["annual_sales_units"].sum().sort_values("model_year")
        fig1 = px.line(
            yearly_sales, x="model_year", y="annual_sales_units", markers=True,
            title="Annual Recorded EV Sales Trend by Model Year",
            labels={"model_year": "Model Year", "annual_sales_units": "Recorded Sales (Units)"},
            color_discrete_sequence=["#059669"],
        )
        fig1.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(style_plotly_fig(fig1), use_container_width=True)

    with col_b:
        top10_mfg = (
            df.groupby("manufacturer", as_index=False)["annual_sales_units"]
            .sum().nlargest(10, "annual_sales_units").sort_values("annual_sales_units", ascending=True)
        )
        fig2 = px.bar(
            top10_mfg, x="annual_sales_units", y="manufacturer", orientation="h",
            title="Top 10 Manufacturers by Recorded Annual Sales",
            labels={"annual_sales_units": "Recorded Sales (Units)", "manufacturer": "Manufacturer"},
            color="annual_sales_units", color_continuous_scale="Tealgrn",
        )
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_plotly_fig(fig2), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        cat_dist = df.groupby("category", as_index=False)["annual_sales_units"].sum().sort_values("annual_sales_units", ascending=True)
        fig3 = px.bar(
            cat_dist, x="annual_sales_units", y="category", orientation="h",
            title="Recorded Sales Distribution by Vehicle Category",
            labels={"annual_sales_units": "Recorded Sales (Units)", "category": "Vehicle Category"},
            color_discrete_sequence=["#0D9488"],
        )
        st.plotly_chart(style_plotly_fig(fig3, height=420), use_container_width=True)

    with col_d:
        region_dist = df.groupby("market_region", as_index=False)["annual_sales_units"].sum().sort_values("annual_sales_units", ascending=False)
        fig4 = px.pie(
            region_dist, names="market_region", values="annual_sales_units", hole=0.45,
            title="Recorded Sales Share by Market Region", color_discrete_sequence=GREEN_PALETTE,
        )
        fig4.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_plotly_fig(fig4, height=420), use_container_width=True)

    col_e, col_f = st.columns(2)
    with col_e:
        bat_share = (
            df.groupby("battery_type", as_index=False)
            .agg(vehicle_count=("vehicle_id", "count"), total_sales=("annual_sales_units", "sum"))
            .sort_values("vehicle_count", ascending=False)
        )
        fig5 = px.pie(
            bat_share, names="battery_type", values="vehicle_count", hole=0.45,
            title="Battery Chemistry Market Share (by Model Count)",
            color="battery_type", color_discrete_map=BATTERY_COLORS,
        )
        fig5.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(style_plotly_fig(fig5), use_container_width=True)

    with col_f:
        sample_df = df.sample(n=min(2500, len(df)), random_state=42)
        fig6 = px.scatter(
            sample_df, x="range_km", y="price_usd", color="category", opacity=0.65,
            title="Price vs. Driving Range Relationship (Sampled n\u22642,500)",
            labels={"range_km": "Driving Range (km)", "price_usd": "Vehicle Price (USD)", "category": "Category"},
            hover_data=["manufacturer", "model_name", "battery_type"],
        )
        st.plotly_chart(style_plotly_fig(fig6), use_container_width=True)

    longest_range_cat = df.groupby("category")["range_km"].mean().idxmax() if not df.empty else "N/A"
    longest_range_val = df.groupby("category")["range_km"].mean().max() if not df.empty else 0.0
    top_bat = df["battery_type"].value_counts().idxmax() if not df.empty else "N/A"
    top_bat_pct = (df["battery_type"].value_counts().max() / len(df)) * 100 if not df.empty else 0.0
    top_region = df.groupby("market_region")["annual_sales_units"].sum().idxmax() if not df.empty else "N/A"

    st.markdown(
        f"""
        <div class="insight-box">
            <h4>\U0001f4ca Automated Executive Insights (Filtered Dataset)</h4>
            <ul>
                <li><b>Segment Leadership:</b> <b>{kpis['top_category']}</b> generates the highest recorded sales volume ({kpis['top_category_sales']:,.0f} units), while <b>{longest_range_cat}</b> achieves the longest average driving range at <b>{longest_range_val:,.1f} km</b>.</li>
                <li><b>Brand Benchmark:</b> <b>{kpis['top_manufacturer']}</b> leads all {kpis['total_manufacturers']} active manufacturers in recorded unit sales with <b>{kpis['top_manufacturer_sales']:,.0f} units</b>.</li>
                <li><b>Battery Chemistry Adoption:</b> <b>{top_bat}</b> is the most frequently deployed battery chemistry, powering <b>{top_bat_pct:.1f}%</b> of filtered models.</li>
                <li><b>Regional Demand & Pricing:</b> <b>{top_region}</b> leads regional unit volume. Across all filtered segments, the mean price is <b>${kpis['avg_price']:,.0f}</b> versus a median of <b>${kpis['median_price']:,.0f}</b>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sales_trends(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">2. Sales & Market Trends Analysis</div>', unsafe_allow_html=True)
    st.info("\u2139\ufe0f **Scope Note:** Sales figures represent recorded unit sales within the dataset rather than total global EV industry sales.")

    yearly = (
        df.groupby("model_year", as_index=False)
        .agg(
            total_sales=("annual_sales_units", "sum"),
            avg_sales_per_model=("annual_sales_units", "mean"),
            model_count=("vehicle_id", "count"),
            avg_price=("price_usd", "mean"),
        )
        .sort_values("model_year")
    )

    yearly["prev_year"] = yearly["model_year"].shift(1)
    yearly["prev_sales"] = yearly["total_sales"].shift(1)
    yearly["is_consecutive"] = (yearly["model_year"] - yearly["prev_year"]) == 1
    yearly["yoy_growth_pct"] = np.where(
        yearly["is_consecutive"] & (yearly["prev_sales"] > 0),
        ((yearly["total_sales"] - yearly["prev_sales"]) / yearly["prev_sales"]) * 100.0,
        np.nan,
    )

    best_year_row = yearly.loc[yearly["total_sales"].idxmax()] if not yearly.empty else None
    c1, c2, c3 = st.columns(3)
    if best_year_row is not None:
        with c1:
            render_kpi_card("Best-Performing Year", f"{int(best_year_row['model_year'])}", f"({best_year_row['total_sales']:,.0f} units)", "Model year with highest recorded sales volume")
        with c2:
            valid_yoy = yearly["yoy_growth_pct"].dropna()
            max_yoy = f"{valid_yoy.max():+.2f}%" if not valid_yoy.empty else "N/A"
            render_kpi_card("Peak YoY Sales Growth", max_yoy, "annual change", "Highest consecutive year-over-year sales growth rate")
        with c3:
            avg_models_yr = yearly["model_count"].mean()
            render_kpi_card("Avg Models per Year", f"{avg_models_yr:,.0f}", "models/yr", "Average vehicle configurations per model year")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_trend = go.Figure(go.Bar(x=yearly["model_year"], y=yearly["total_sales"], name="Total Sales (Units)", marker_color="#10B981", opacity=0.85))
        fig_trend.update_layout(title="Annual Recorded Sales Volume by Model Year", xaxis_title="Model Year", yaxis_title="Recorded Sales (Units)")
        st.plotly_chart(style_plotly_fig(fig_trend), use_container_width=True)

    with col2:
        yoy_df = yearly.dropna(subset=["yoy_growth_pct"]).copy()
        if not yoy_df.empty:
            yoy_df["color"] = np.where(yoy_df["yoy_growth_pct"] >= 0, "#059669", "#DC2626")
            fig_yoy = go.Figure(go.Bar(
                x=yoy_df["model_year"], y=yoy_df["yoy_growth_pct"],
                marker_color=yoy_df["color"],
                text=yoy_df["yoy_growth_pct"].apply(lambda v: f"{v:+.2f}%"),
                textposition="outside",
            ))
            fig_yoy.update_layout(title="Consecutive Year-over-Year (YoY) Sales Growth (%)", xaxis_title="Model Year", yaxis_title="YoY Growth (%)")
            st.plotly_chart(style_plotly_fig(fig_yoy), use_container_width=True)
        else:
            st.warning("Consecutive model years are required to compute YoY growth percentages.")

    col3, col4 = st.columns(2)
    with col3:
        top_cats = df.groupby("category")["annual_sales_units"].sum().nlargest(6).index.tolist()
        cat_yr = df[df["category"].isin(top_cats)].groupby(["model_year", "category"], as_index=False)["annual_sales_units"].sum()
        fig_cat_yr = px.line(
            cat_yr, x="model_year", y="annual_sales_units", color="category", markers=True,
            title="Sales Trends for Top 6 Vehicle Categories Over Time",
            labels={"model_year": "Model Year", "annual_sales_units": "Sales (Units)", "category": "Category"},
        )
        st.plotly_chart(style_plotly_fig(fig_cat_yr), use_container_width=True)

    with col4:
        reg_yr = df.groupby(["model_year", "market_region"], as_index=False)["annual_sales_units"].sum()
        fig_reg_yr = px.area(
            reg_yr, x="model_year", y="annual_sales_units", color="market_region",
            title="Regional Sales Trajectory Across Model Years",
            labels={"model_year": "Model Year", "annual_sales_units": "Sales (Units)", "market_region": "Region"},
            color_discrete_sequence=GREEN_PALETTE,
        )
        st.plotly_chart(style_plotly_fig(fig_reg_yr), use_container_width=True)

    st.markdown("#### \U0001f50d Interactive Period-to-Period Comparison")
    avail_years = sorted(df["model_year"].dropna().unique().astype(int).tolist())
    if len(avail_years) >= 2:
        yc1, yc2 = st.columns(2)
        with yc1:
            base_yr = st.selectbox("Baseline Year", avail_years, index=0)
        with yc2:
            comp_yr = st.selectbox("Comparison Year", avail_years, index=len(avail_years) - 1)

        base_sub = df[df["model_year"] == base_yr]
        comp_sub = df[df["model_year"] == comp_yr]
        comp_cat = (
            base_sub.groupby("category")["annual_sales_units"].sum().rename(f"Sales_{base_yr}").to_frame()
            .join(comp_sub.groupby("category")["annual_sales_units"].sum().rename(f"Sales_{comp_yr}"), how="outer")
            .fillna(0).reset_index()
        )
        comp_cat["Growth_%"] = np.where(
            comp_cat[f"Sales_{base_yr}"] > 0,
            ((comp_cat[f"Sales_{comp_yr}"] - comp_cat[f"Sales_{base_yr}"]) / comp_cat[f"Sales_{base_yr}"]) * 100.0,
            np.nan,
        )
        comp_cat = comp_cat.sort_values(f"Sales_{comp_yr}", ascending=False)
        st.dataframe(
            comp_cat.style.format({f"Sales_{base_yr}": "{:,.0f}", f"Sales_{comp_yr}": "{:,.0f}", "Growth_%": "{:+.2f}%"}),
            use_container_width=True, hide_index=True,
        )


def render_manufacturer_performance(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">3. Manufacturer Performance & Competitive Benchmarking</div>', unsafe_allow_html=True)
    total_dataset_sales = df["annual_sales_units"].sum()
    mfg_summary = (
        df.groupby("manufacturer", as_index=False)
        .agg(
            models_count=("vehicle_id", "count"),
            total_sales=("annual_sales_units", "sum"),
            avg_sales_per_model=("annual_sales_units", "mean"),
            avg_price=("price_usd", "mean"),
            avg_range_km=("range_km", "mean"),
            avg_battery_kwh=("battery_capacity_kwh", "mean"),
            avg_rating=("customer_rating", "mean"),
            avg_efficiency=("efficiency_wh_per_km", "mean"),
        )
        .sort_values("total_sales", ascending=False)
    )
    mfg_summary["dataset_sales_share_pct"] = np.where(total_dataset_sales > 0, (mfg_summary["total_sales"] / total_dataset_sales) * 100.0, 0.0)

    col1, col2 = st.columns(2)
    with col1:
        top15 = mfg_summary.head(15).sort_values("total_sales", ascending=True)
        fig_m1 = px.bar(
            top15, x="total_sales", y="manufacturer", orientation="h",
            title="Top 15 Manufacturers by Recorded Unit Sales",
            labels={"total_sales": "Total Sales (Units)", "manufacturer": "Manufacturer"},
            color="avg_rating", color_continuous_scale="Tealgrn",
            hover_data=["models_count", "avg_price", "avg_range_km"],
        )
        st.plotly_chart(style_plotly_fig(fig_m1, height=430), use_container_width=True)

    with col2:
        top40 = mfg_summary.head(40)
        fig_m2 = px.scatter(
            top40, x="avg_range_km", y="avg_price", size="total_sales", color="avg_rating",
            hover_name="manufacturer", title="Top 40 OEMs: Avg Range vs. Avg Price (Bubble = Sales Volume)",
            labels={"avg_range_km": "Avg Driving Range (km)", "avg_price": "Avg Price (USD)", "avg_rating": "Avg Rating"},
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(style_plotly_fig(fig_m2, height=430), use_container_width=True)

    st.markdown("#### \u2696\ufe0f Side-by-Side Manufacturer Benchmarking")
    default_mfgs = mfg_summary["manufacturer"].head(5).tolist()
    selected_mfgs = st.multiselect("Select 2 or more manufacturers to benchmark side by side:", options=mfg_summary["manufacturer"].tolist(), default=default_mfgs)

    if len(selected_mfgs) >= 2:
        sub_mfg = df[df["manufacturer"].isin(selected_mfgs)]
        bench_table = mfg_summary[mfg_summary["manufacturer"].isin(selected_mfgs)].copy()

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            fig_b1 = px.bar(
                bench_table, x="manufacturer", y=["avg_range_km", "avg_battery_kwh"], barmode="group",
                title="Avg Driving Range (km) & Battery Capacity (kWh) by Selected OEM",
                color_discrete_sequence=["#059669", "#0284C7"],
            )
            st.plotly_chart(style_plotly_fig(fig_b1), use_container_width=True)

        with b_col2:
            cat_mix = sub_mfg.groupby(["manufacturer", "category"], as_index=False)["annual_sales_units"].sum()
            fig_b2 = px.bar(
                cat_mix, x="manufacturer", y="annual_sales_units", color="category",
                title="Vehicle Category Sales Portfolio by Selected OEM",
                labels={"annual_sales_units": "Sales (Units)", "manufacturer": "Manufacturer"},
            )
            st.plotly_chart(style_plotly_fig(fig_b2), use_container_width=True)

        st.dataframe(
            bench_table.style.format({
                "models_count": "{:,}", "total_sales": "{:,.0f}", "avg_sales_per_model": "{:,.1f}",
                "avg_price": "${:,.2f}", "avg_range_km": "{:,.1f} km", "avg_battery_kwh": "{:,.1f} kWh",
                "avg_rating": "{:.2f} \u2605", "avg_efficiency": "{:,.1f} Wh/km", "dataset_sales_share_pct": "{:.2f}%",
            }),
            use_container_width=True, hide_index=True,
        )


def render_battery_analysis(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">4. Battery Technology & Chemistry Performance</div>', unsafe_allow_html=True)
    bat_stats = (
        df.groupby("battery_type", as_index=False)
        .agg(
            models_count=("vehicle_id", "count"),
            total_sales=("annual_sales_units", "sum"),
            avg_capacity_kwh=("battery_capacity_kwh", "mean"),
            avg_range_km=("range_km", "mean"),
            avg_cycle_life=("battery_cycle_life", "mean"),
            avg_degradation_5yr=("battery_degradation_pct_5yr", "mean"),
            avg_dc_fast_min=("charging_dc_fast_time_min", "mean"),
            avg_max_dc_kw=("max_dc_charging_kw", "mean"),
            avg_price_usd=("price_usd", "mean"),
            avg_efficiency=("efficiency_wh_per_km", "mean"),
        )
        .sort_values("models_count", ascending=False)
    )
    st.dataframe(
        bat_stats.style.format({
            "models_count": "{:,}", "total_sales": "{:,.0f}", "avg_capacity_kwh": "{:,.1f} kWh",
            "avg_range_km": "{:,.1f} km", "avg_cycle_life": "{:,.0f} cycles", "avg_degradation_5yr": "{:.2f}%",
            "avg_dc_fast_min": "{:.1f} min", "avg_max_dc_kw": "{:.1f} kW", "avg_price_usd": "${:,.0f}",
            "avg_efficiency": "{:.1f} Wh/km",
        }),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_bat1 = px.bar(
            bat_stats, x="battery_type", y="avg_cycle_life", color="battery_type",
            text=bat_stats["avg_cycle_life"].apply(lambda v: f"{v:,.0f}"),
            title="Average Battery Cycle Life by Chemistry", color_discrete_map=BATTERY_COLORS,
        )
        fig_bat1.update_traces(textposition="outside")
        st.plotly_chart(style_plotly_fig(fig_bat1), use_container_width=True)

    with c2:
        fig_bat2 = px.bar(
            bat_stats, x="battery_type", y="avg_degradation_5yr", color="battery_type",
            text=bat_stats["avg_degradation_5yr"].apply(lambda v: f"{v:.2f}%"),
            title="Average 5-Year Battery Degradation (%) by Chemistry", color_discrete_map=BATTERY_COLORS,
        )
        fig_bat2.update_traces(textposition="outside")
        st.plotly_chart(style_plotly_fig(fig_bat2), use_container_width=True)


def render_category_analysis(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">5. Vehicle Category & Segment Analysis (All 15 Segments)</div>', unsafe_allow_html=True)
    cat_summary = (
        df.groupby("category", as_index=False)
        .agg(
            models_count=("vehicle_id", "count"),
            manufacturers_count=("manufacturer", "nunique"),
            total_sales=("annual_sales_units", "sum"),
            avg_price_usd=("price_usd", "mean"),
            median_price_usd=("price_usd", "median"),
            avg_range_km=("range_km", "mean"),
            avg_battery_kwh=("battery_capacity_kwh", "mean"),
            avg_efficiency=("efficiency_wh_per_km", "mean"),
            avg_rating=("customer_rating", "mean"),
            avg_co2_saved=("co2_saved_tons_per_year", "mean"),
        )
        .sort_values("total_sales", ascending=False)
    )

    c1, c2 = st.columns(2)
    with c1:
        fig_c1 = px.bar(
            cat_summary.sort_values("avg_range_km", ascending=True),
            x="avg_range_km", y="category", orientation="h", color="avg_battery_kwh",
            title="Average Driving Range (km) & Battery Capacity (kWh) by Category",
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(style_plotly_fig(fig_c1, height=450), use_container_width=True)

    with c2:
        fig_c2 = px.scatter(
            cat_summary, x="avg_price_usd", y="total_sales", size="models_count",
            color="avg_rating", text="category",
            title="Segment Attractiveness: Avg Price vs. Total Recorded Sales",
            color_continuous_scale="Viridis",
        )
        fig_c2.update_traces(textposition="top center")
        st.plotly_chart(style_plotly_fig(fig_c2, height=450), use_container_width=True)

    st.dataframe(
        cat_summary.style.format({
            "models_count": "{:,}", "manufacturers_count": "{:,}", "total_sales": "{:,.0f}",
            "avg_price_usd": "${:,.2f}", "median_price_usd": "${:,.2f}", "avg_range_km": "{:,.1f} km",
            "avg_battery_kwh": "{:,.1f} kWh", "avg_efficiency": "{:,.1f} Wh/km",
            "avg_rating": "{:.2f} \u2605", "avg_co2_saved": "{:.2f} t/yr",
        }),
        use_container_width=True, hide_index=True,
    )


def render_customer_satisfaction(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">6. Customer Satisfaction & Rating Drivers</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig_r1 = px.histogram(df, x="customer_rating", nbins=35, title="Distribution of Customer Satisfaction Ratings", color_discrete_sequence=["#059669"])
        st.plotly_chart(style_plotly_fig(fig_r1), use_container_width=True)

    with c2:
        cat_rat = df.groupby("category", as_index=False)["customer_rating"].mean().sort_values("customer_rating", ascending=True)
        fig_r2 = px.bar(cat_rat, x="customer_rating", y="category", orientation="h", title="Average Customer Rating by Vehicle Category", color="customer_rating", color_continuous_scale="Tealgrn")
        fig_r2.update_layout(xaxis_range=[3.5, 4.5], coloraxis_showscale=False)
        st.plotly_chart(style_plotly_fig(fig_r2), use_container_width=True)


def render_geographic_analysis(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">7. Geographic Manufacturing & Market Region Intelligence</div>', unsafe_allow_html=True)
    ISO_MAP = {
        "China": "CHN", "USA": "USA", "Germany": "DEU", "Japan": "JPN",
        "South Korea": "KOR", "India": "IND", "Vietnam": "VNM",
        "France": "FRA", "Sweden": "SWE", "UK": "GBR",
    }
    ctry_stats = (
        df.groupby("manufacturing_country", as_index=False)
        .agg(
            models_count=("vehicle_id", "count"),
            total_sales=("annual_sales_units", "sum"),
            avg_price=("price_usd", "mean"),
            avg_range=("range_km", "mean"),
            avg_rating=("customer_rating", "mean"),
        )
        .sort_values("total_sales", ascending=False)
    )
    ctry_stats["iso_alpha"] = ctry_stats["manufacturing_country"].map(ISO_MAP)

    c1, c2 = st.columns(2)
    with c1:
        fig_map = px.choropleth(
            ctry_stats, locations="iso_alpha", color="total_sales",
            hover_name="manufacturing_country", hover_data=["models_count", "avg_price", "avg_range"],
            color_continuous_scale="Tealgrn", title="Global EV Manufacturing Footprint (by Recorded Unit Sales)",
        )
        fig_map.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth"))
        st.plotly_chart(style_plotly_fig(fig_map, height=420), use_container_width=True)

    with c2:
        fig_ctry_bar = px.bar(
            ctry_stats.sort_values("total_sales", ascending=True),
            x="total_sales", y="manufacturing_country", orientation="h", color="avg_price",
            title="Recorded Sales Volume & Avg Price by Manufacturing Country", color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(style_plotly_fig(fig_ctry_bar, height=420), use_container_width=True)


def render_sustainability_insights(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">8. Sustainability, Energy Efficiency & CO\u2082 Impact</div>', unsafe_allow_html=True)
    total_co2_all_models = df["co2_saved_tons_per_year"].sum()
    avg_eff = df["efficiency_wh_per_km"].mean()
    avg_cycle = df["battery_cycle_life"].mean()
    avg_resale = df["resale_value_pct_3yr"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Cumulative Model CO\u2082 Saved", f"{total_co2_all_models:,.0f}", "tons/yr", "Sum of annual CO\u2082 savings across active model records")
    with k2:
        render_kpi_card("Avg Energy Efficiency", f"{avg_eff:,.1f}", "Wh/km", "Mean energy consumption per kilometer")
    with k3:
        render_kpi_card("Avg Battery Cycle Life", f"{avg_cycle:,.0f}", "cycles", "Mean full charge-discharge cycles")
    with k4:
        render_kpi_card("Avg 3-Yr Resale Retention", f"{avg_resale:.1f}%", "of MSRP", "Mean projected value retention after 36 months")

    c1, c2 = st.columns(2)
    with c1:
        co2_cat = df.groupby("category", as_index=False)["co2_saved_tons_per_year"].mean().sort_values("co2_saved_tons_per_year", ascending=True)
        fig_s1 = px.bar(co2_cat, x="co2_saved_tons_per_year", y="category", orientation="h", color="co2_saved_tons_per_year", title="Average Annual CO\u2082 Saved (Tons/Year) by Vehicle Category", color_continuous_scale="Tealgrn")
        fig_s1.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_plotly_fig(fig_s1, height=430), use_container_width=True)

    with c2:
        sample_s = df.sample(n=min(2500, len(df)), random_state=42)
        fig_s2 = px.scatter(sample_s, x="weight_kg", y="efficiency_wh_per_km", color="category", opacity=0.65, title="Vehicle Curb Weight (kg) vs. Energy Consumption (Wh/km)")
        st.plotly_chart(style_plotly_fig(fig_s2, height=430), use_container_width=True)


def render_data_explorer(df: pd.DataFrame, quality_report: Dict[str, Any]) -> None:
    st.markdown('<div class="section-title">9. Data Explorer & Data Quality Audit</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Total Raw Records", f"{quality_report['initial_rows']:,}", "rows", f"Across {quality_report['total_columns']} schema columns")
    with k2:
        render_kpi_card("Filtered Records", f"{len(df):,}", "rows", f"{(len(df) / max(1, quality_report['cleaned_rows'])) * 100:.1f}% of dataset active")
    with k3:
        render_kpi_card("Exact / ID Duplicates", f"{quality_report['exact_duplicates']} / {quality_report['id_duplicates']}", "records", "Verified unique vehicle_id records")
    with k4:
        total_miss_before = sum(quality_report["missing_before"].values())
        total_miss_after = sum(quality_report["missing_after"].values())
        render_kpi_card("Missing Cells (Pre \u2192 Post)", f"{total_miss_before:,} \u2192 {total_miss_after:,}", "cells", "Group-median imputation status")

    if quality_report["imputation_details"]:
        imp_df = pd.DataFrame(quality_report["imputation_details"])
        imp_df.columns = ["Column Name", "Missing Count (Raw)", "Missing %", "Handling Strategy"]
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    search_term = st.text_input("Search by Vehicle ID, Model Name, or Manufacturer:", placeholder="e.g., SurgeWorks, EV00001...")
    display_df = df.copy()
    if search_term.strip():
        q = search_term.strip().lower()
        display_df = display_df[
            display_df["vehicle_id"].str.lower().str.contains(q, na=False)
            | display_df["model_name"].str.lower().str.contains(q, na=False)
            | display_df["manufacturer"].str.lower().str.contains(q, na=False)
        ]

    st.dataframe(display_df.head(1000), use_container_width=True, hide_index=True)
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"\u2b07\ufe0f Download Filtered Dataset as CSV ({len(display_df):,} rows)",
        data=csv_bytes,
        file_name="ev_market_intelligence_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>\u26a1 Electric Vehicle (EV) Market Intelligence Dashboard</h1>
            <p>Executive Business Intelligence, Competitive Benchmarking, Battery Chemistry Analytics, and Sustainability Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("\U0001f9ed Dashboard Navigation")
    sections = [
        "1. Executive Overview",
        "2. Sales and Market Trends",
        "3. Manufacturer Performance",
        "4. Battery Technology Analysis",
        "5. Vehicle Category Analysis",
        "6. Customer Satisfaction",
        "7. Geographic & Market Analysis",
        "8. Sustainability & Efficiency",
        "9. Data Explorer & Data Quality",
    ]
    active_section = st.sidebar.radio("Select Analysis Module:", sections)

    st.sidebar.markdown("---")
    st.sidebar.subheader("\U0001f4c2 Dataset Source & Cleaning")
    uploaded_file = st.sidebar.file_uploader("Upload Custom CSV/Excel (Optional):", type=["csv", "xlsx", "xls"])
    impute_toggle = st.sidebar.checkbox("Impute missing numeric values (Group Median)", value=True)

    raw_df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        raw_df = load_data(uploaded_file)
    else:
        for candidate_path in get_default_dataset_paths():
            if os.path.exists(candidate_path):
                raw_df = load_data(candidate_path)
                break

    if raw_df is None:
        st.warning("\u26a0\ufe0f `ev_market_test.csv` was not found. Please upload a CSV/Excel dataset in the sidebar.")
        return

    df, quality_report = clean_data(raw_df, impute_missing=impute_toggle)

    st.sidebar.markdown("---")
    st.sidebar.subheader("\U0001f39b\ufe0f Global Interactive Filters")
    all_years = sorted(df["model_year"].dropna().unique().astype(int).tolist())
    sel_years = st.sidebar.multiselect("Model Year:", options=all_years, default=all_years)
    all_regions = sorted(df["market_region"].dropna().unique().tolist())
    sel_regions = st.sidebar.multiselect("Market Region:", options=all_regions, default=all_regions)
    all_countries = sorted(df["manufacturing_country"].dropna().unique().tolist())
    sel_countries = st.sidebar.multiselect("Manufacturing Country:", options=all_countries, default=all_countries)
    all_categories = sorted(df["category"].dropna().unique().tolist())
    sel_categories = st.sidebar.multiselect("Vehicle Category (15 Segments):", options=all_categories, default=all_categories)
    all_batteries = sorted(df["battery_type"].dropna().unique().tolist())
    sel_batteries = st.sidebar.multiselect("Battery Technology:", options=all_batteries, default=all_batteries)

    filters = {
        "model_years": sel_years,
        "market_regions": sel_regions,
        "manufacturing_countries": sel_countries,
        "categories": sel_categories,
        "battery_types": sel_batteries,
        "impute_missing": impute_toggle,
    }
    filtered_df = filter_data(df, filters)
    if filtered_df.empty:
        st.error("\U0001f6ab No vehicles match your current filter criteria.")
        return

    kpis = calculate_kpis(filtered_df)
    if active_section.startswith("1."):
        render_executive_overview(filtered_df, kpis)
    elif active_section.startswith("2."):
        render_sales_trends(filtered_df)
    elif active_section.startswith("3."):
        render_manufacturer_performance(filtered_df)
    elif active_section.startswith("4."):
        render_battery_analysis(filtered_df)
    elif active_section.startswith("5."):
        render_category_analysis(filtered_df)
    elif active_section.startswith("6."):
        render_customer_satisfaction(filtered_df)
    elif active_section.startswith("7."):
        render_geographic_analysis(filtered_df)
    elif active_section.startswith("8."):
        render_sustainability_insights(filtered_df)
    elif active_section.startswith("9."):
        render_data_explorer(filtered_df, quality_report)


if __name__ == "__main__":
    main()