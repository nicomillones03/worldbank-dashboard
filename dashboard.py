"""
dashboard.py — Interactive LAC ODA Dashboard (Streamlit + Plotly)

Launch:
    cd priority_engine/
    streamlit run dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LAC ODA Partnership Dashboard",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"

# ─── Colour palettes ──────────────────────────────────────────────────────────
PRIORITY_COLOURS = {
    "High": "#2C73D2",
    "Medium": "#0CA4A5",
    "Low": "#F5A623",
    "Peripheral / not assessed": "#D5D5D5",
}
CONTEXT_COLOURS = {
    "Anchor partnership space": "#2C73D2",
    "Fragmented coordination space": "#FF6F61",
    "Thin / emerging space": "#0CA4A5",
    "Low-activity space": "#D5D5D5",
}
DTYPE_COLOURS = {
    "Bilateral (DAC)":     "#2C73D2",
    "Bilateral (Non-DAC)": "#0CA4A5",
    "MDB":                 "#F5A623",
    "UN Agency":           "#FF6F61",
    "Vertical Fund":       "#845EC2",
    "Other":               "#B0B0B0",
}
PRIORITY_ORDER = ["High", "Medium", "Low", "Peripheral / not assessed"]
CONTEXT_ORDER = [
    "Anchor partnership space",
    "Fragmented coordination space",
    "Thin / emerging space",
    "Low-activity space",
]
DTYPE_ORDER = ["Bilateral (DAC)", "MDB", "UN Agency", "Vertical Fund",
               "Bilateral (Non-DAC)", "Other"]


# ─── Data loading (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    cs  = pd.read_csv(DATA_DIR / "country_sector_context_table.csv")
    dcs = pd.read_csv(DATA_DIR / "donor_country_sector_priority_table.csv")
    by_year       = pd.read_csv(DATA_DIR / "agg_by_year.csv")
    by_dtype_year = pd.read_csv(DATA_DIR / "agg_by_donor_type_year.csv")
    by_sector_year = pd.read_csv(DATA_DIR / "agg_by_sector_year.csv")
    by_country_year = pd.read_csv(DATA_DIR / "agg_by_country_year.csv")

    dcs["priority"]   = pd.Categorical(dcs["priority"],   categories=PRIORITY_ORDER, ordered=True)
    dcs["cs_context"] = pd.Categorical(dcs["cs_context"], categories=CONTEXT_ORDER,  ordered=True)
    cs["cs_context"]  = pd.Categorical(cs["cs_context"],  categories=CONTEXT_ORDER,  ordered=True)

    return cs, dcs, by_year, by_dtype_year, by_sector_year, by_country_year


cs_raw, dcs_raw, by_year, by_dtype_year, by_sector_year, by_country_year = load_data()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌎 LAC ODA Dashboard")
    st.caption("World Bank Capstone · 2002–2024")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Financing Volumes & Trends",
            "Sectoral Analysis",
            "Country–Sector Mapping",
            "Priority Framework",
            "Donor Profiles",
            "Priority Table",
        ],
    )

    st.divider()
    st.subheader("Filters")

    all_countries = sorted(dcs_raw["country_name"].unique())
    sel_countries = st.multiselect("Country", all_countries, placeholder="All countries")

    all_sectors = sorted(dcs_raw["sector"].unique())
    sel_sectors = st.multiselect("Sector", all_sectors, placeholder="All sectors")

    all_dtypes = sorted(dcs_raw["donor_type"].unique())
    sel_dtypes = st.multiselect("Donor type", all_dtypes, placeholder="All types")

    sel_priorities = st.multiselect("Priority", PRIORITY_ORDER, placeholder="All priorities")
    sel_contexts   = st.multiselect("Context",  CONTEXT_ORDER,  placeholder="All contexts")
    hide_peripheral = st.checkbox("Hide Peripheral / not assessed", value=False)


# ─── Filter helpers ───────────────────────────────────────────────────────────
def filter_dcs(df):
    m = pd.Series(True, index=df.index)
    if sel_countries:  m &= df["country_name"].isin(sel_countries)
    if sel_sectors:    m &= df["sector"].isin(sel_sectors)
    if sel_dtypes:     m &= df["donor_type"].isin(sel_dtypes)
    if sel_priorities: m &= df["priority"].isin(sel_priorities)
    if sel_contexts:   m &= df["cs_context"].isin(sel_contexts)
    if hide_peripheral: m &= df["priority"] != "Peripheral / not assessed"
    return df[m].copy()

def filter_cs(df):
    m = pd.Series(True, index=df.index)
    if sel_countries: m &= df["country_name"].isin(sel_countries)
    if sel_sectors:   m &= df["sector"].isin(sel_sectors)
    if sel_contexts:  m &= df["cs_context"].isin(sel_contexts)
    return df[m].copy()

dcs = filter_dcs(dcs_raw)
cs  = filter_cs(cs_raw)


def fmt_m(v):
    if abs(v) >= 1000: return f"${v/1000:,.1f}B"
    return f"${v:,.1f}M"

def short_ctx(label):
    return str(label).replace(" space","").replace(" partnership","")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("LAC ODA Partnership Dashboard")
    st.caption(
        "Analysis of official development assistance to Latin America & the Caribbean, "
        "2002–2024. Source: OECD CRS / DAC2A."
    )

    assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_recent = by_year[by_year["year"] >= 2022]["total_oda"].sum() / 3
    c1.metric("Countries", dcs["country_name"].nunique())
    c2.metric("Sectors", dcs["sector"].nunique())
    c3.metric("Donors", dcs["donor_name"].nunique())
    c4.metric("Avg annual ODA (2022–24)", fmt_m(total_recent))
    c5.metric("Assessed partnerships", f"{len(assessed):,}")
    c6.metric("High priority", f"{(assessed['priority']=='High').sum():,}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Total ODA to LAC over time")
        fig = px.area(
            by_year, x="year", y="total_oda",
            labels={"year": "Year", "total_oda": "Total ODA ($M)"},
            color_discrete_sequence=["#2C73D2"],
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("ODA by donor type (2022–2024)")
        recent_dtype = (
            by_dtype_year[by_dtype_year["year"] >= 2022]
            .groupby("donor_type")["total_oda"].mean()
            .reset_index()
            .sort_values("total_oda", ascending=False)
        )
        fig2 = px.bar(
            recent_dtype, x="total_oda", y="donor_type",
            orientation="h",
            color="donor_type",
            color_discrete_map=DTYPE_COLOURS,
            labels={"total_oda": "Avg annual ODA ($M)", "donor_type": ""},
        )
        fig2.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Priority distribution")
        pri = (
            dcs["priority"].value_counts().reindex(PRIORITY_ORDER)
            .reset_index().rename(columns={"priority":"Priority","count":"Count"})
        )
        fig3 = px.bar(pri, x="Priority", y="Count", color="Priority",
                      color_discrete_map=PRIORITY_COLOURS, text_auto=True)
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.subheader("Context × Priority cross-tab")
        xtab = pd.crosstab(
            assessed["cs_context"], assessed["priority"], margins=True
        )
        cols = [c for c in PRIORITY_ORDER if c in xtab.columns] + ["All"]
        rows = [r for r in CONTEXT_ORDER if r in xtab.index] + ["All"]
        st.dataframe(xtab.reindex(index=rows, columns=cols, fill_value=0),
                     use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Financing Volumes & Trends
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Financing Volumes & Trends":
    st.title("Financing Volumes & Trends")
    st.caption(
        "Total ODA disbursements to LAC by year, actor type, and top donors. "
        "Covers the full analysis window (2002–2024). Country / sector filters apply where relevant."
    )

    # ── Total trend ──────────────────────────────────────────────────────────
    st.subheader("Total ODA to LAC, 2002–2024")
    fig_tot = px.area(
        by_year, x="year", y="total_oda",
        labels={"year": "Year", "total_oda": "Total ODA ($M)"},
        color_discrete_sequence=["#2C73D2"],
    )
    fig_tot.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_tot, use_container_width=True)

    st.divider()

    # ── By donor type ────────────────────────────────────────────────────────
    st.subheader("ODA by donor type over time")
    fig_dtype = px.area(
        by_dtype_year.sort_values(["year","donor_type"]),
        x="year", y="total_oda", color="donor_type",
        color_discrete_map=DTYPE_COLOURS,
        category_orders={"donor_type": DTYPE_ORDER},
        labels={"year": "Year", "total_oda": "ODA ($M)", "donor_type": "Donor type"},
    )
    fig_dtype.update_layout(height=400, legend_title="Donor type")
    st.plotly_chart(fig_dtype, use_container_width=True)

    st.divider()

    # ── Top donors (recent) ──────────────────────────────────────────────────
    st.subheader("Top donors by recent average disbursement (2022–2024)")
    top_n = st.slider("Show top N donors", 5, 30, 15, key="top_donors")

    top_donors = (
        dcs_raw[dcs_raw["priority"] != "Peripheral / not assessed"]
        .groupby(["donor_name", "donor_type"])["recent_avg"]
        .sum()
        .reset_index()
        .sort_values("recent_avg", ascending=False)
        .head(top_n)
    )
    fig_top = px.bar(
        top_donors, y="donor_name", x="recent_avg",
        color="donor_type", color_discrete_map=DTYPE_COLOURS,
        orientation="h",
        labels={"recent_avg": "Avg annual ODA ($M)", "donor_name": "",
                "donor_type": "Donor type"},
        category_orders={"donor_type": DTYPE_ORDER},
    )
    fig_top.update_layout(
        height=max(400, top_n * 28),
        yaxis=dict(categoryorder="total ascending"),
        legend_title="Donor type",
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.divider()

    # ── Top recipient countries ───────────────────────────────────────────────
    st.subheader("Top recipient countries by recent average ODA (2022–2024)")
    top_cty = st.slider("Show top N countries", 5, 35, 15, key="top_cty")
    recent_cty = (
        by_country_year[by_country_year["year"] >= 2022]
        .groupby("country_name")["total_oda"].mean()
        .reset_index()
        .sort_values("total_oda", ascending=False)
        .head(top_cty)
    )
    fig_cty = px.bar(
        recent_cty, y="country_name", x="total_oda",
        orientation="h",
        color_discrete_sequence=["#2C73D2"],
        labels={"total_oda": "Avg annual ODA ($M)", "country_name": ""},
    )
    fig_cty.update_layout(
        height=max(350, top_cty * 26),
        yaxis=dict(categoryorder="total ascending"),
        showlegend=False,
    )
    st.plotly_chart(fig_cty, use_container_width=True)

    st.divider()

    # ── Country trend ─────────────────────────────────────────────────────────
    st.subheader("ODA trend for selected countries")
    default_cty = ["Brazil", "Colombia", "Mexico", "Haiti", "Peru"]
    default_cty = [c for c in default_cty if c in by_country_year["country_name"].unique()]
    sel_cty_trend = st.multiselect(
        "Select countries", sorted(by_country_year["country_name"].unique()),
        default=default_cty[:5],
    )
    if sel_cty_trend:
        cty_trend = by_country_year[by_country_year["country_name"].isin(sel_cty_trend)]
        fig_cty_t = px.line(
            cty_trend, x="year", y="total_oda", color="country_name",
            labels={"year": "Year", "total_oda": "ODA ($M)", "country_name": "Country"},
            markers=True,
        )
        fig_cty_t.update_layout(height=400, legend_title="Country")
        st.plotly_chart(fig_cty_t, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Sectoral Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Sectoral Analysis":
    st.title("Sectoral Analysis")
    st.caption(
        "Which sectors receive the most financing, how that has changed over time, "
        "and which are growing fastest."
    )

    # ── Sector shares (recent) ───────────────────────────────────────────────
    st.subheader("Sector shares of total ODA (2022–2024)")
    recent_sec = (
        by_sector_year[by_sector_year["year"] >= 2022]
        .groupby("sector")["total_oda"].mean()
        .reset_index()
        .sort_values("total_oda", ascending=False)
    )
    recent_sec["share"] = recent_sec["total_oda"] / recent_sec["total_oda"].sum() * 100

    col_l, col_r = st.columns([3, 2])
    with col_l:
        fig_sec_bar = px.bar(
            recent_sec, y="sector", x="total_oda",
            orientation="h",
            color_discrete_sequence=["#2C73D2"],
            labels={"total_oda": "Avg annual ODA ($M)", "sector": ""},
            text=recent_sec["share"].round(1).astype(str) + "%",
        )
        fig_sec_bar.update_layout(
            height=520, yaxis=dict(categoryorder="total ascending"), showlegend=False
        )
        st.plotly_chart(fig_sec_bar, use_container_width=True)

    with col_r:
        # Group sectors 9+ into "Other" so pie % matches bar %
        pie_data = recent_sec.head(8).copy()
        if len(recent_sec) > 8:
            other = pd.DataFrame([{
                "sector": "Other",
                "total_oda": recent_sec.iloc[8:]["total_oda"].sum(),
                "share":     recent_sec.iloc[8:]["share"].sum(),
            }])
            pie_data = pd.concat([pie_data, other], ignore_index=True)
        fig_sec_pie = px.pie(
            pie_data, values="share", names="sector",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_sec_pie.update_traces(textinfo="percent+label")
        fig_sec_pie.update_layout(height=520, showlegend=False)
        st.plotly_chart(fig_sec_pie, use_container_width=True)

    st.divider()

    # ── Sector trends over time ──────────────────────────────────────────────
    st.subheader("Sector ODA trends over time")
    all_secs = sorted(by_sector_year["sector"].unique())
    default_secs = [
        "Health", "Education", "Governance & Civil Society",
        "Agriculture & Natural Resources", "Transport & Storage",
    ]
    default_secs = [s for s in default_secs if s in all_secs]
    sel_secs = st.multiselect("Select sectors", all_secs, default=default_secs[:5])

    if sel_secs:
        sec_trend = by_sector_year[by_sector_year["sector"].isin(sel_secs)]
        fig_sec_t = px.line(
            sec_trend, x="year", y="total_oda", color="sector",
            labels={"year": "Year", "total_oda": "ODA ($M)", "sector": "Sector"},
            markers=True,
        )
        fig_sec_t.update_layout(height=420, legend_title="Sector")
        st.plotly_chart(fig_sec_t, use_container_width=True)

    st.divider()

    # ── Growth rates ─────────────────────────────────────────────────────────
    st.subheader("Sector growth: 2002–2010 vs 2015–2024 average")
    early = (
        by_sector_year[by_sector_year["year"].between(2002, 2010)]
        .groupby("sector")["total_oda"].mean().rename("early_avg")
    )
    late = (
        by_sector_year[by_sector_year["year"].between(2015, 2024)]
        .groupby("sector")["total_oda"].mean().rename("late_avg")
    )
    growth = pd.concat([early, late], axis=1).dropna()
    growth["growth_pct"] = (growth["late_avg"] - growth["early_avg"]) / growth["early_avg"].abs() * 100
    growth = growth.reset_index().sort_values("growth_pct", ascending=False)

    colours = ["#2C73D2" if v >= 0 else "#FF6F61" for v in growth["growth_pct"]]
    fig_growth = go.Figure(go.Bar(
        y=growth["sector"], x=growth["growth_pct"],
        orientation="h",
        marker_color=colours,
        text=growth["growth_pct"].round(0).astype(int).astype(str) + "%",
        textposition="outside",
    ))
    fig_growth.update_layout(
        height=520,
        xaxis_title="Growth (%)",
        yaxis=dict(categoryorder="total ascending"),
        showlegend=False,
    )
    st.plotly_chart(fig_growth, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Country–Sector Mapping
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Country–Sector Mapping":
    st.title("Country–Sector Financing Mapping")
    st.caption(
        "Where development finance is concentrated across LAC, and where potential "
        "financing gaps exist. Each cell shows the Stage 1 context classification."
    )

    # ── Financing volume heatmap ─────────────────────────────────────────────
    st.subheader("Total recent avg disbursement by country × sector ($M)")
    vol_pivot = cs.pivot_table(
        index="country_name", columns="sector",
        values="cs_recent_avg", aggfunc="first",
    ).sort_index()

    fig_vol = go.Figure(data=go.Heatmap(
        z=np.log1p(vol_pivot.fillna(0).values),
        x=vol_pivot.columns.tolist(),
        y=vol_pivot.index.tolist(),
        customdata=vol_pivot.fillna(0).round(1).values,
        hovertemplate="<b>%{y}</b> × %{x}<br>$%{customdata}M<extra></extra>",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="log($M)"),
    ))
    fig_vol.update_layout(
        height=max(500, len(vol_pivot) * 22),
        xaxis=dict(tickangle=45, side="top"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200, t=130),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()

    # ── Context classification heatmap ────────────────────────────────────────
    st.subheader("Context classification heatmap")
    ctx_num = {c: i for i, c in enumerate(CONTEXT_ORDER)}
    cs_hm = cs.copy()
    cs_hm["ctx_num"] = cs_hm["cs_context"].map(ctx_num)
    pivot = cs_hm.pivot_table(
        index="country_name", columns="sector",
        values="ctx_num", aggfunc="first",
    ).sort_index()
    hover = cs_hm.pivot_table(
        index="country_name", columns="sector",
        values="cs_context", aggfunc="first",
    )

    colour_scale = [
        [0.0,  CONTEXT_COLOURS[CONTEXT_ORDER[0]]],
        [0.33, CONTEXT_COLOURS[CONTEXT_ORDER[1]]],
        [0.66, CONTEXT_COLOURS[CONTEXT_ORDER[2]]],
        [1.0,  CONTEXT_COLOURS[CONTEXT_ORDER[3]]],
    ]
    fig_ctx = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        customdata=hover.values,
        hovertemplate="<b>%{y}</b> × %{x}<br>%{customdata}<extra></extra>",
        colorscale=colour_scale,
        zmin=0, zmax=3,
        showscale=False,
    ))
    fig_ctx.update_layout(
        height=max(500, len(pivot) * 22),
        xaxis=dict(tickangle=45, side="top"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200, t=130),
    )
    st.plotly_chart(fig_ctx, use_container_width=True)

    legend_md = " | ".join(
        f'<span style="color:{CONTEXT_COLOURS[c]}">■</span> **{short_ctx(c)}**'
        for c in CONTEXT_ORDER
    )
    st.caption("Anchor = blue · Fragmented = red · Thin/emerging = teal · Low-activity = grey")

    st.divider()

    # ── Gap analysis ─────────────────────────────────────────────────────────
    st.subheader("Potential financing gaps")
    st.caption(
        "Country–sector cells with low activity (below P33 disbursement) and few donors — "
        "spaces where external financing presence is limited."
    )
    gaps = cs[cs["cs_context"] == "Low-activity space"].sort_values(
        "cs_recent_avg"
    )[["country_name", "sector", "cs_recent_avg", "cs_donor_count"]]
    gaps.columns = ["Country", "Sector", "Recent Avg ($M)", "Active Donors"]
    gaps["Recent Avg ($M)"] = gaps["Recent Avg ($M)"].round(2)
    st.dataframe(gaps.reset_index(drop=True), use_container_width=True, height=400)

    st.divider()

    # ── Disbursement vs donors scatter ────────────────────────────────────────
    st.subheader("Disbursement vs donor count — context classification")
    fig_sc = px.scatter(
        cs, x="cs_donor_count", y="cs_recent_avg",
        color="cs_context", color_discrete_map=CONTEXT_COLOURS,
        hover_data=["country_name", "sector"],
        log_y=True,
        render_mode="svg",
        category_orders={"cs_context": CONTEXT_ORDER},
        labels={
            "cs_donor_count": "Active donors (recent window)",
            "cs_recent_avg": "Recent avg disbursement ($M, log)",
            "cs_context": "Context",
        },
    )
    fig_sc.update_layout(height=500, legend_title="Context")
    st.plotly_chart(fig_sc, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Priority Framework
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Priority Framework":
    st.title("Priority Level of Engagement Framework")
    st.caption(
        "Stage 2: donor-country-sector priority assignment based on disbursement scale, "
        "persistence, and operational embeddedness."
    )

    assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assessed partnerships", f"{len(assessed):,}")
    c2.metric("High priority", f"{(assessed['priority']=='High').sum():,}")
    c3.metric("Medium priority", f"{(assessed['priority']=='Medium').sum():,}")
    c4.metric("Low priority", f"{(assessed['priority']=='Low').sum():,}")

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Priority by sector")
        sec_pri = (
            assessed.groupby(["sector", "priority"]).size()
            .reset_index(name="count")
        )
        fig_sp = px.bar(
            sec_pri, y="sector", x="count",
            color="priority", color_discrete_map=PRIORITY_COLOURS,
            orientation="h", barmode="stack",
            category_orders={"priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"]},
        )
        fig_sp.update_layout(
            height=500, yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Assessed partnerships", yaxis_title="", legend_title="Priority",
        )
        st.plotly_chart(fig_sp, use_container_width=True)

    with col_r:
        st.subheader("Priority by country")
        cty_pri = (
            assessed.groupby(["country_name", "priority"]).size()
            .reset_index(name="count")
        )
        fig_cp = px.bar(
            cty_pri, y="country_name", x="count",
            color="priority", color_discrete_map=PRIORITY_COLOURS,
            orientation="h", barmode="stack",
            category_orders={"priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"]},
        )
        fig_cp.update_layout(
            height=max(500, dcs["country_name"].nunique() * 20),
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="Assessed partnerships", yaxis_title="", legend_title="Priority",
        )
        st.plotly_chart(fig_cp, use_container_width=True)

    st.divider()

    st.subheader("Top High-priority partnerships by disbursement")
    top_n = st.slider("Show top N", 10, 50, 20)
    high = assessed[assessed["priority"] == "High"].sort_values("recent_avg", ascending=False)
    if len(high):
        show = high.head(top_n)[
            ["donor_name", "country_name", "sector", "recent_avg",
             "active_share", "sector_share", "n_strong", "cs_context"]
        ].copy()
        show["recent_avg"]   = show["recent_avg"].apply(lambda v: f"${v:,.2f}M")
        show["active_share"] = (show["active_share"]*100).round(0).astype(int).astype(str)+"%"
        show["sector_share"] = (show["sector_share"]*100).round(1).astype(str)+"%"
        show.columns = ["Donor","Country","Sector","Recent Avg","Active %","Sector %","Strong","Context"]
        st.dataframe(show.reset_index(drop=True), use_container_width=True, height=500)

    st.divider()

    st.subheader("Indicator scatter")
    if len(assessed):
        x_col = st.selectbox("X axis", ["recent_avg","active_share","sector_share"], index=0)
        y_col = st.selectbox("Y axis", ["active_share","recent_avg","sector_share"], index=0)
        fig_ind = px.scatter(
            assessed, x=x_col, y=y_col,
            color="priority", color_discrete_map=PRIORITY_COLOURS,
            hover_data=["donor_name","country_name","sector"],
            opacity=0.6,
            render_mode="svg",
            category_orders={"priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"]},
            labels={
                "recent_avg": "Recent avg disbursement ($M)",
                "active_share": "Active years share",
                "sector_share": "Sector share",
            },
        )
        fig_ind.update_layout(height=500)
        st.plotly_chart(fig_ind, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Donor Profiles
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Donor Profiles":
    st.title("Donor Profiles")
    st.caption("Deep-dive into a specific donor's partnerships across countries and sectors.")

    assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]
    donor_options = sorted(assessed["donor_name"].unique())

    if not donor_options:
        st.warning("No assessed donors with current filters.")
    else:
        sel_donor = st.selectbox("Select donor", donor_options)
        donor_df  = assessed[assessed["donor_name"] == sel_donor].copy()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Countries", donor_df["country_name"].nunique())
        c2.metric("Sectors",   donor_df["sector"].nunique())
        c3.metric("Total recent avg", fmt_m(donor_df["recent_avg"].sum()))
        c4.metric("High priority", (donor_df["priority"] == "High").sum())

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Priority breakdown")
            dp = (
                donor_df["priority"].value_counts()
                .reindex([p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"])
                .fillna(0).reset_index()
            )
            dp.columns = ["Priority", "Count"]
            fig_dp = px.pie(dp, values="Count", names="Priority",
                            color="Priority", color_discrete_map=PRIORITY_COLOURS, hole=0.4)
            fig_dp.update_traces(textinfo="value+percent")
            st.plotly_chart(fig_dp, use_container_width=True)

        with col_r:
            st.subheader("Context breakdown")
            dc = (
                donor_df["cs_context"].value_counts()
                .reindex(CONTEXT_ORDER).fillna(0).reset_index()
            )
            dc.columns = ["Context", "Count"]
            fig_dc = px.pie(dc, values="Count", names="Context",
                            color="Context", color_discrete_map=CONTEXT_COLOURS, hole=0.4)
            fig_dc.update_traces(textinfo="value+percent")
            st.plotly_chart(fig_dc, use_container_width=True)

        st.subheader(f"{sel_donor} — country × sector priority map")
        pri_num = {"High": 3, "Medium": 2, "Low": 1}
        donor_df["pri_num"] = donor_df["priority"].map(pri_num).fillna(0)
        piv_d  = donor_df.pivot_table(index="country_name", columns="sector",
                                      values="pri_num",    aggfunc="first").sort_index()
        hov_d  = donor_df.pivot_table(index="country_name", columns="sector",
                                      values="priority",   aggfunc="first")

        pri_cs = [[0.0, "#eeeeee"], [0.33, PRIORITY_COLOURS["Low"]],
                  [0.66, PRIORITY_COLOURS["Medium"]], [1.0, PRIORITY_COLOURS["High"]]]

        fig_dhm = go.Figure(data=go.Heatmap(
            z=piv_d.values, x=piv_d.columns.tolist(), y=piv_d.index.tolist(),
            customdata=hov_d.values,
            hovertemplate="<b>%{y}</b> × %{x}<br>%{customdata}<extra></extra>",
            colorscale=pri_cs, zmin=0, zmax=3, showscale=False,
        ))
        fig_dhm.update_layout(
            height=max(350, len(piv_d)*24),
            xaxis=dict(tickangle=45, side="top"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200, t=120),
        )
        st.plotly_chart(fig_dhm, use_container_width=True)

        st.subheader("All partnerships")
        detail = donor_df[[
            "country_name","sector","recent_avg","active_share",
            "sector_share","n_strong","priority","cs_context"
        ]].sort_values(["priority","recent_avg"], ascending=[True,False]).copy()
        detail["recent_avg"]   = detail["recent_avg"].apply(lambda v: f"${v:,.2f}M")
        detail["active_share"] = (detail["active_share"]*100).round(0).astype(int).astype(str)+"%"
        detail["sector_share"] = (detail["sector_share"]*100).round(1).astype(str)+"%"
        detail.columns = ["Country","Sector","Recent Avg","Active %","Sector %","Strong","Priority","Context"]
        st.dataframe(detail.reset_index(drop=True), use_container_width=True, height=500)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Priority Table
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Priority Table":
    st.title("Full Priority Table")
    st.caption(
        "Complete donor–country–sector priority table. Use sidebar filters to narrow "
        "results, then download."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        sort_col = st.selectbox("Sort by",
            ["recent_avg","n_strong","active_share","sector_share","donor_name"])
    with col_b:
        sort_asc = st.radio("Order", ["Descending","Ascending"], horizontal=True) == "Ascending"

    display = dcs.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    show = display[[
        "donor_name","donor_type","country_name","sector",
        "recent_avg","active_years","active_share","sector_share",
        "disbursement_strong","persistence_strong","embeddedness_strong",
        "n_strong","priority","cs_context"
    ]].copy()
    show["recent_avg"]   = show["recent_avg"].apply(lambda v: f"${v:,.2f}M")
    show["active_share"] = (show["active_share"]*100).round(0).astype(int).astype(str)+"%"
    show["sector_share"] = (show["sector_share"]*100).round(1).astype(str)+"%"
    show.columns = [
        "Donor","Type","Country","Sector",
        "Recent Avg","Active Yrs","Active %","Sector %",
        "Disburse","Persist","Embed","Strong","Priority","Context",
    ]
    st.dataframe(show, use_container_width=True, height=650)

    st.download_button(
        "⬇ Download filtered table (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="priority_table_filtered.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Summary statistics")
    assessed_d = display[display["priority"] != "Peripheral / not assessed"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{len(display):,}")
    c2.metric("Assessed", f"{len(assessed_d):,}")
    c3.metric("High priority", f"{(assessed_d['priority']=='High').sum():,}")
    c4.metric("Total recent avg", fmt_m(assessed_d["recent_avg"].sum()))
