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


# ─── Inline help strings (tooltips) ──────────────────────────────────────────
HELP = {
    "priority": (
        "Stage 2 engagement level: High = strong on ≥2 of 3 indicators "
        "(including disbursement or persistence); Medium = strong on 1; "
        "Low = assessed but strong on none; Peripheral = not assessed "
        "(no recent activity)."
    ),
    "context": (
        "Stage 1 country-sector classification based on total disbursement "
        "and number of active donors:\n"
        "• Anchor = large volume + few concentrated donors\n"
        "• Fragmented = many donors + moderate-to-high volume\n"
        "• Thin / emerging = few donors, growing activity\n"
        "• Low-activity = limited external financing presence"
    ),
    "recent_avg": "Average annual disbursement over the most recent 3 years (2022–2024), in USD millions.",
    "active_share": "Share of years in the analysis window where the donor was active (disbursed >0) in this country-sector.",
    "sector_share": "Donor's share of total disbursement in this country-sector (recent window).",
    "n_strong": "Number of the three Stage 2 indicators (disbursement, persistence, embeddedness) where the donor ranks strong.",
    "cs_context": "Stage 1 country-sector context label (Anchor / Fragmented / Thin / Low-activity).",
    "disburse_pctile": "Percentile rank of total disbursement for this country-sector across LAC.",
    "donor_count_pctile": "Percentile rank of donor count for this country-sector across LAC.",
    "cagr": "Compound Annual Growth Rate, measured between 3-year averages at each endpoint to smooth single-year noise.",
}


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
            "🏠 Guide",
            "📊 Descriptive Analysis",
            "🗺️ Country–Sector Mapping",
            "🎯 Priority Framework",
            "👥 Donor Profiles",
            "📋 Priority Table",
            "📚 Methodology & Glossary",
        ],
    )

    st.divider()
    st.subheader("Filters")
    st.caption("Filters apply to analytical pages (Mapping, Framework, Profiles, Table). The Guide, Descriptive Analysis, and Methodology pages always show the full dataset.")

    all_countries = sorted(dcs_raw["country_name"].unique())
    sel_countries = st.multiselect("Country", all_countries, placeholder="All countries")

    all_sectors = sorted(dcs_raw["sector"].unique())
    sel_sectors = st.multiselect("Sector", all_sectors, placeholder="All sectors")

    all_dtypes = sorted(dcs_raw["donor_type"].unique())
    sel_dtypes = st.multiselect("Donor type", all_dtypes, placeholder="All types")

    sel_priorities = st.multiselect(
        "Priority", PRIORITY_ORDER, placeholder="All priorities",
        help=HELP["priority"],
    )
    sel_contexts   = st.multiselect(
        "Context",  CONTEXT_ORDER,  placeholder="All contexts",
        help=HELP["context"],
    )
    hide_peripheral = st.checkbox(
        "Hide Peripheral / not assessed", value=False,
        help="Hide donor-country-sector cells that were too small or too short-lived to assess.",
    )


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
# PAGE 1 — Guide (new landing page)
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Guide":
    st.title("🏠 Welcome to the LAC ODA Partnership Dashboard")
    st.caption("A decision-support tool to identify partnership opportunities in Latin America & the Caribbean — designed for the World Bank, usable by any development organisation working in the region.")

    st.markdown(
        """
        ### What this dashboard does
        This tool was designed primarily for **World Bank country and sector
        teams**, but the framework is general: **any development organisation
        working in LAC** — other MDBs, bilateral agencies, UN bodies, or
        philanthropic foundations — can use it to identify **which other
        development agents are the best candidates for co-financing or
        coordination in a specific country and sector**.

        It answers the research question (framed here from the World Bank's
        perspective):

        > **Which development agents financing Latin America and the Caribbean
        > represent the highest priority partnership opportunities for the
        > World Bank, based on financing volume, sectoral focus, and country
        > presence — and how should priority levels (High / Medium / Low) be
        > assigned by country–sector combination?**

        Other organisations can read the question with their own institution
        in place of "the World Bank" — the underlying priority framework is
        donor-agnostic.

        **Typical use case.** You work on the World Bank's Argentina team and
        want to scope partners for a new Health programme. Filter the dashboard
        to *Argentina × Health* and instantly see: which donors are already
        active, how large and persistent their engagement is, and how they rank
        in the priority framework (High / Medium / Low). Use this to shortlist
        co-financing partners and benchmark the financing landscape. The same
        workflow applies to an IDB or UNDP team scoping their own country-sector.

        It combines 23 years of Official Development Assistance (ODA) data
        (OECD CRS / DAC2A, 2002–2024) into a two-stage prioritisation framework:

        1. **Stage 1** — classifies every **country × sector** space by how
           crowded and well-funded it is (the *context*). This tells you
           *what kind of space* you're operating in.
        2. **Stage 2** — classifies every **donor × country × sector**
           partnership by how central that donor is in that space (the
           *priority*). This tells you *which donors to prioritise* as
           potential partners.
        """
    )

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1️⃣ Explore")
        st.markdown(
            "Start with **📊 Descriptive Analysis** to understand overall LAC "
            "financing volumes, trends, top donors, and sector patterns."
        )
    with c2:
        st.markdown("#### 2️⃣ Scope your country × sector")
        st.markdown(
            "Use sidebar filters to narrow to *your* country and sector, then "
            "open **🗺️ Country–Sector Mapping** to read the financing landscape."
        )
    with c3:
        st.markdown("#### 3️⃣ Shortlist partners")
        st.markdown(
            "Move to **🎯 Priority Framework**, **👥 Donor Profiles**, and "
            "**📋 Priority Table** to identify High-priority donors as "
            "candidate partners for your project."
        )

    st.divider()

    st.markdown("### Page guide")
    st.markdown(
        """
| Page | What you'll find | Best for |
|---|---|---|
| **🏠 Guide** | This page — overview and navigation | First-time visitors |
| **📊 Descriptive Analysis** | Volumes, trends, top donors, sector landscape | Context & background |
| **🗺️ Country–Sector Mapping** | Heatmaps of financing and context labels; gap analysis | Spotting under-served areas |
| **🎯 Priority Framework** | High/Medium/Low priority distribution by sector, country, indicators | Strategic prioritisation |
| **👥 Donor Profiles** | Deep-dive into a single donor's partnership portfolio | Donor-specific planning |
| **📋 Priority Table** | Full downloadable table with filters | Data export for further analysis |
| **📚 Methodology & Glossary** | Plain-language explanations of every term and indicator | Understanding the methodology |
"""
    )

    st.divider()

    st.markdown("### How to use the sidebar")
    st.info(
        "**Filters** on the left (Country, Sector, Donor type, Priority, Context) "
        "apply to the analytical pages — Mapping, Framework, Profiles, and Table. "
        "Use them to zoom into a specific region, sector, or partner group. "
        "Leave them empty to see everything."
    )

    st.markdown("### Key terms at a glance")
    with st.expander("Show quick definitions (full glossary on the Methodology page)"):
        st.markdown(
            """
- **ODA** — Official Development Assistance (concessional finance to developing countries).
- **Country-sector context** (Stage 1): **Anchor**, **Fragmented**, **Thin / emerging**, or **Low-activity** — characterises the financing landscape in each country × sector.
- **Priority** (Stage 2): **High**, **Medium**, **Low**, or **Peripheral / not assessed** — characterises the strategic centrality of each donor in each country × sector.
- **Recent avg** — average annual disbursement over 2022–2024, in USD millions.
- **CAGR** — Compound Annual Growth Rate, computed between 3-year endpoint averages to smooth noise.
            """
        )

    st.caption(
        "ℹ️ Data source: OECD Creditor Reporting System (CRS) and DAC2A tables, "
        "via the Python ETL pipeline in this repository. Regional aggregates "
        "(e.g. \"America (Regional)\") are excluded from country-level analysis."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Descriptive Analysis (consolidated)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Descriptive Analysis":
    st.title("📊 Descriptive Analysis")
    st.caption(
        "Background context on LAC ODA flows, 2002–2024. "
        "Use the sub-tabs below to explore volumes & trends, the donor landscape, and the sector landscape."
    )

    tab_over, tab_trends, tab_donors, tab_sectors = st.tabs([
        "Overview", "Volumes & Trends", "Donors", "Sectors",
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-tab A — Overview (KPIs + snapshot charts)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_over:
        assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        total_recent = by_year[by_year["year"] >= 2022]["total_oda"].sum() / 3
        c1.metric("Countries", dcs["country_name"].nunique())
        c2.metric("Sectors", dcs["sector"].nunique())
        c3.metric("Donors", dcs["donor_name"].nunique())
        c4.metric(
            "Avg annual ODA (2022–24)", fmt_m(total_recent),
            help="Mean total ODA disbursed to LAC per year across 2022–2024.",
        )
        c5.metric(
            "Assessed partnerships", f"{len(assessed):,}",
            help="Donor-country-sector cells that passed the Stage 2 assessment thresholds.",
        )
        c6.metric(
            "High priority", f"{(assessed['priority']=='High').sum():,}",
            help=HELP["priority"],
        )

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
            st.subheader("Priority distribution", help=HELP["priority"])
            pri = (
                dcs["priority"].value_counts().reindex(PRIORITY_ORDER)
                .reset_index().rename(columns={"priority":"Priority","count":"Count"})
            )
            fig3 = px.bar(pri, x="Priority", y="Count", color="Priority",
                          color_discrete_map=PRIORITY_COLOURS, text_auto=True)
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            st.subheader("Context × Priority cross-tab", help=HELP["context"])
            xtab = pd.crosstab(
                assessed["cs_context"], assessed["priority"], margins=True
            )
            cols = [c for c in PRIORITY_ORDER if c in xtab.columns] + ["All"]
            rows = [r for r in CONTEXT_ORDER if r in xtab.index] + ["All"]
            st.dataframe(xtab.reindex(index=rows, columns=cols, fill_value=0),
                         use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-tab B — Volumes & Trends
    # ─────────────────────────────────────────────────────────────────────────
    with tab_trends:
        st.subheader("Total ODA to LAC, 2002–2024")
        fig_tot = px.area(
            by_year, x="year", y="total_oda",
            labels={"year": "Year", "total_oda": "Total ODA ($M)"},
            color_discrete_sequence=["#2C73D2"],
        )
        fig_tot.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_tot, use_container_width=True)

        st.divider()

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

        st.subheader("Top recipient countries by recent avg ODA (2022–2024)")
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

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-tab C — Donors
    # ─────────────────────────────────────────────────────────────────────────
    with tab_donors:
        st.subheader("Top donors by recent avg disbursement (2022–2024)")
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

        st.subheader("ODA share by donor type (2022–2024)")
        recent_dtype = (
            by_dtype_year[by_dtype_year["year"] >= 2022]
            .groupby("donor_type")["total_oda"].mean()
            .reset_index()
        )
        recent_dtype["share"] = recent_dtype["total_oda"] / recent_dtype["total_oda"].sum() * 100
        fig_share = px.pie(
            recent_dtype, values="share", names="donor_type",
            color="donor_type", color_discrete_map=DTYPE_COLOURS,
            hole=0.4,
        )
        fig_share.update_traces(textinfo="percent+label")
        fig_share.update_layout(height=420)
        st.plotly_chart(fig_share, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-tab D — Sectors
    # ─────────────────────────────────────────────────────────────────────────
    with tab_sectors:
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

        # ── CAGR helper ───────────────────────────────────────────────────────
        def sector_cagr(start_yrs, end_yrs):
            s = (by_sector_year[by_sector_year["year"].isin(start_yrs)]
                 .groupby("sector")["total_oda"].mean())
            e = (by_sector_year[by_sector_year["year"].isin(end_yrs)]
                 .groupby("sector")["total_oda"].mean())
            n = (sum(end_yrs) / len(end_yrs)) - (sum(start_yrs) / len(start_yrs))
            df = pd.concat([s.rename("s"), e.rename("e")], axis=1).dropna()
            df = df[(df["s"] > 0) & (df["e"] > 0)]
            df["cagr"] = ((df["e"] / df["s"]) ** (1 / n) - 1) * 100
            return df["cagr"].reset_index()

        st.subheader("Sector CAGR — full period (2002–2024)", help=HELP["cagr"])
        st.caption(
            "3-year averages at each endpoint: 2002–2004 → 2022–2024 · n = 20 years. "
            "Blue = positive growth, red = contraction."
        )
        full = sector_cagr([2002, 2003, 2004], [2022, 2023, 2024])
        full = full.sort_values("cagr", ascending=False)

        fig_full = go.Figure(go.Bar(
            y=full["sector"], x=full["cagr"],
            orientation="h",
            marker_color=["#2C73D2" if v >= 0 else "#FF6F61" for v in full["cagr"]],
            text=full["cagr"].round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig_full.update_layout(
            height=520,
            xaxis_title="CAGR (%)",
            yaxis=dict(categoryorder="total ascending"),
            showlegend=False,
        )
        st.plotly_chart(fig_full, use_container_width=True)

        st.divider()

        st.subheader("Sector CAGR — sub-periods (2002–2013 vs 2013–2024)")
        st.caption(
            "3-year averages at each endpoint. "
            "2002–2013: 2002–2004 → 2011–2013, n ≈ 9 years. "
            "2013–2024: 2011–2013 → 2022–2024, n ≈ 11 years."
        )
        c1 = sector_cagr([2002, 2003, 2004], [2011, 2012, 2013]).rename(columns={"cagr": "2002–2013"})
        c2 = sector_cagr([2011, 2012, 2013], [2022, 2023, 2024]).rename(columns={"cagr": "2013–2024"})
        sub = c1.merge(c2, on="sector", how="outer").sort_values("2013–2024", ascending=False)

        sub_long = sub.melt(id_vars="sector", var_name="Period", value_name="cagr")
        fig_sub = px.bar(
            sub_long, y="sector", x="cagr",
            color="Period", barmode="group", orientation="h",
            color_discrete_map={"2002–2013": "#0CA4A5", "2013–2024": "#2C73D2"},
            text=sub_long["cagr"].round(1).astype(str) + "%",
            labels={"cagr": "CAGR (%)", "sector": ""},
        )
        fig_sub.update_layout(
            height=560,
            yaxis=dict(categoryorder="total ascending"),
            xaxis_title="CAGR (%)",
            legend_title="Period",
            uniformtext_minsize=8,
            uniformtext_mode="hide",
        )
        st.plotly_chart(fig_sub, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Country–Sector Mapping (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Country–Sector Mapping":
    st.title("🗺️ Country–Sector Financing Mapping")
    st.caption(
        "Where development finance is concentrated across LAC, and where potential "
        "financing gaps exist. Each cell shows the Stage 1 context classification."
    )

    with st.expander("ℹ️ What do the context labels mean?"):
        st.markdown(
            """
- **Anchor partnership space** — large volume with **few, concentrated donors** (established, well-funded partnerships).
- **Fragmented coordination space** — many donors with moderate-to-high total volume; coordination is the key challenge.
- **Thin / emerging space** — few donors, growing activity; strategic entry opportunity.
- **Low-activity space** — low disbursements regardless of donor count; potential financing gap.

See the **📚 Methodology** page for the full definitions and thresholds.
            """
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
    st.subheader("Context classification heatmap", help=HELP["context"])
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
# PAGE 4 — Priority Framework (unchanged, with tooltips added)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Priority Framework":
    st.title("🎯 Priority Level of Engagement Framework")
    st.caption(
        "Stage 2: donor-country-sector priority assignment based on disbursement scale, "
        "persistence, and operational embeddedness."
    )

    with st.expander("ℹ️ What do the priority levels mean?"):
        st.markdown(
            """
Each donor × country × sector cell is assessed on three indicators:
1. **💰 Disbursement strength** — donor ranks in the top third within its country-sector AND above the global P50 across all assessed donors.
2. **🔄 Persistence** — donor was active in at least 4 of the last 5 years (2020–2024) with stable disbursements (CV < 1.0).
3. **🎯 Embeddedness** — this country-sector represents ≥ 15% of the donor's total country portfolio.

`n_strong` = count of True flags (0–3). Priority rule:
- **High** → `n_strong` ≥ 2, including disbursement or persistence
- **Medium** → `n_strong` = 1
- **Low** → assessed but `n_strong` = 0
- **Peripheral / not assessed** → no active year in the recent window

See the **📚 Methodology** page for precise thresholds.
            """
        )

    assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assessed partnerships", f"{len(assessed):,}",
              help="Donor-country-sector cells that passed the Stage 2 minimum activity rule.")
    c2.metric("High priority", f"{(assessed['priority']=='High').sum():,}", help=HELP["priority"])
    c3.metric("Medium priority", f"{(assessed['priority']=='Medium').sum():,}", help=HELP["priority"])
    c4.metric("Low priority", f"{(assessed['priority']=='Low').sum():,}", help=HELP["priority"])

    # ── Indicator flag rates ───────────────────────────────────────────────────
    n = len(assessed)
    d_pct = assessed["disbursement_strong"].sum() / n * 100
    p_pct = assessed["persistence_strong"].sum()  / n * 100
    e_pct = assessed["embeddedness_strong"].sum()  / n * 100
    c5, c6, c7 = st.columns(3)
    c5.metric("💰 Disbursement strong", f"{d_pct:.0f}% of assessed",
              help="Donors ranking in the top third within their country-sector AND above global P50.")
    c6.metric("🔄 Persistence strong", f"{p_pct:.0f}% of assessed",
              help="Donors active in ≥ 4 of the last 5 years with CV < 1.0.")
    c7.metric("🎯 Embeddedness strong", f"{e_pct:.0f}% of assessed",
              help="Donors allocating ≥ 15% of their country portfolio to this sector.")

    st.divider()

    # ── Priority by sector / country ──────────────────────────────────────────
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

    # ── What drives priority? ─────────────────────────────────────────────────
    st.subheader("What drives each priority level?")
    st.caption(
        "Share of assessed partnerships that are strong on each indicator, "
        "broken down by priority level."
    )
    flag_rates = (
        assessed.groupby("priority")[
            ["disbursement_strong", "persistence_strong", "embeddedness_strong"]
        ]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    flag_long = flag_rates.melt(
        id_vars="priority", var_name="Indicator", value_name="% strong"
    )
    flag_long["Indicator"] = flag_long["Indicator"].map({
        "disbursement_strong": "💰 Disbursement",
        "persistence_strong":  "🔄 Persistence",
        "embeddedness_strong": "🎯 Embeddedness",
    })
    INDICATOR_COLOURS = {
        "💰 Disbursement": "#2C73D2",
        "🔄 Persistence":  "#0CA4A5",
        "🎯 Embeddedness": "#F5A623",
    }
    fig_flags = px.bar(
        flag_long,
        x="priority", y="% strong", color="Indicator",
        barmode="group",
        color_discrete_map=INDICATOR_COLOURS,
        category_orders={
            "priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"],
            "Indicator": ["💰 Disbursement", "🔄 Persistence", "🎯 Embeddedness"],
        },
        labels={"priority": "Priority level", "% strong": "% of partnerships"},
        text_auto=".0f",
    )
    fig_flags.update_layout(
        height=380, legend_title="Indicator",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
    )
    st.plotly_chart(fig_flags, use_container_width=True)

    st.divider()

    # ── Top High-priority partnerships table ──────────────────────────────────
    st.subheader("Top High-priority partnerships by disbursement")
    top_n = st.slider("Show top N", 10, 50, 20)
    high = assessed[assessed["priority"] == "High"].sort_values("recent_avg", ascending=False)
    if len(high):
        cols_needed = [
            "donor_name", "country_name", "sector",
            "recent_avg", "disburse_pctile_within_cs",
            "persistence_active_years", "cv_recent", "sector_share",
            "disbursement_strong", "persistence_strong", "embeddedness_strong",
            "cs_context",
        ]
        show = high.head(top_n)[[c for c in cols_needed if c in high.columns]].copy()

        show["recent_avg"] = show["recent_avg"].apply(lambda v: f"${v:,.1f}M")
        show["disburse_pctile_within_cs"] = show["disburse_pctile_within_cs"].round(0).astype("Int64").astype(str) + "%"
        if "persistence_active_years" in show.columns:
            show["persistence_active_years"] = show["persistence_active_years"].astype(int).astype(str) + "/5 yrs"
        if "cv_recent" in show.columns:
            show["cv_recent"] = show["cv_recent"].round(2)
        show["sector_share"] = (show["sector_share"] * 100).round(1).astype(str) + "%"
        for flag in ["disbursement_strong", "persistence_strong", "embeddedness_strong"]:
            if flag in show.columns:
                show[flag] = show[flag].map({True: "✓", False: "✗"})

        show.columns = [
            "Donor", "Country", "Sector",
            "Recent Avg", "CS Rank",
            "5yr Active", "CV", "Sector %",
            "💰 Disburse", "🔄 Persist", "🎯 Embed",
            "Context",
        ]
        st.dataframe(show.reset_index(drop=True), use_container_width=True, height=500)
        st.caption(
            "CS Rank = percentile within country-sector (assessed donors only). "
            "5yr Active = active years out of last 5. CV = coefficient of variation of recent disbursements."
        )

    st.divider()

    # ── Indicator scatter ─────────────────────────────────────────────────────
    st.subheader("Indicator scatter")
    st.caption("Explore how any two indicators relate across assessed partnerships. Colour = priority level.")
    if len(assessed):
        axis_options = {
            "Recent avg disbursement ($M)":       "recent_avg",
            "CS rank — disbursement percentile":  "disburse_pctile_within_cs",
            "5yr active years (out of 5)":        "persistence_active_years",
            "CV of recent disbursements":         "cv_recent",
            "Sector share (portfolio %)":         "sector_share",
        }
        # filter to options whose column exists
        axis_options = {k: v for k, v in axis_options.items() if v in assessed.columns}
        ax_labels = list(axis_options.keys())

        col_x, col_y = st.columns(2)
        x_label = col_x.selectbox("X axis", ax_labels, index=0)
        y_label = col_y.selectbox("Y axis", ax_labels, index=2)
        x_col = axis_options[x_label]
        y_col = axis_options[y_label]

        fig_ind = px.scatter(
            assessed, x=x_col, y=y_col,
            color="priority", color_discrete_map=PRIORITY_COLOURS,
            hover_data=["donor_name", "country_name", "sector"],
            opacity=0.6,
            render_mode="svg",
            category_orders={"priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"]},
            labels={x_col: x_label, y_col: y_label},
        )
        fig_ind.update_layout(height=500, legend_title="Priority")
        st.plotly_chart(fig_ind, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Donor Profiles (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Donor Profiles":
    st.title("👥 Donor Profiles")
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
        c3.metric("Total recent avg", fmt_m(donor_df["recent_avg"].sum()),
                  help=HELP["recent_avg"])
        c4.metric("High priority", (donor_df["priority"] == "High").sum(),
                  help=HELP["priority"])

        st.divider()
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Priority breakdown", help=HELP["priority"])
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
            st.subheader("Context breakdown", help=HELP["context"])
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
# PAGE 6 — Priority Table (unchanged)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋 Priority Table":
    st.title("📋 Full Priority Table")
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

    with st.expander("ℹ️ Column definitions"):
        st.markdown(
            f"""
- **Recent Avg** — {HELP['recent_avg']}
- **Active Yrs** — number of years the donor disbursed anything in this country-sector.
- **Active %** — {HELP['active_share']}
- **Sector %** — {HELP['sector_share']}
- **Disburse / Persist / Embed** — boolean flags (True/False) for each of the three Stage 2 indicators.
- **Strong** — {HELP['n_strong']}
- **Priority** — {HELP['priority']}
- **Context** — {HELP['cs_context']}
            """
        )

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


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Methodology & Glossary (new)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📚 Methodology & Glossary":
    st.title("📚 Methodology & Glossary")
    st.caption("Plain-language guide to how each indicator and label in this dashboard is produced.")

    tab_m, tab_g, tab_tech = st.tabs(["Methodology", "Glossary", "Technical details"])

    # ─────────────────────────────────────────────────────────────────────────
    # Methodology — plain language
    # ─────────────────────────────────────────────────────────────────────────
    with tab_m:
        st.markdown(
            """
## The research question

> **Which development agents financing Latin America and the Caribbean
> represent the highest priority partnership opportunities for the World Bank,
> based on financing volume, sectoral focus, and country presence — and how
> should priority levels (High / Medium / Low) be assigned by country–sector
> combination?**

This dashboard answers that question with a **two-stage framework** built on
23 years of ODA data (2002–2024). It was designed primarily for **World Bank
country and sector teams**, but the framework is donor-agnostic and can be
used by **any development organisation working in LAC** (other MDBs, bilateral
agencies, UN bodies, philanthropies) scoping partnership opportunities for a
specific project in a specific country × sector.

---

## Overview

The framework assigns partnership priority levels to development finance actors
at the granularity of **donor × country × sector**. It uses a two-stage
approach: first classifying the structural context of each country-sector
space, then evaluating each donor's position within that space.

The framework is transparent, reproducible, and avoids arbitrary weighting
schemes, PCA, Pareto dominance, or black-box composite scores. Classification
is purely rule-based with interpretable thresholds.

---

## Data sources

- **OECD Creditor Reporting System (CRS)** — donor × recipient × sector × year
  activity-level records. Source of sector composition.
- **OECD DAC2A (Gross ODA)** — donor × recipient × year authoritative volume
  totals. Used to anchor absolute disbursement amounts.

We combine the two OECD sources using an **allocation method**: DAC2A provides
the total volume envelope, and CRS sector shares determine how that envelope
is split across sectors. This preserves headline volumes while keeping
sectoral detail.

**Scope:** Latin America & the Caribbean, 35 countries, 17 sectors,
2002–2024. Regional aggregates (e.g. "America (Regional)") are **excluded**
from country-level analysis.

---

## Indicators

The framework uses **four indicators**, each tied to a dimension of aid quality:

| Indicator | Definition | What it captures |
|---|---|---|
| **Recent avg disbursement** (`recent_avg`) | Mean annual ODA over 2022–2024. Negative values retained. | Financing scale |
| **Persistence** (`persistence_active_years`) | Active years in the last 5-year window (2020–2024), combined with a stability check (CV < 1.0). | Recent committed presence |
| **Sector share** (`sector_share`) | Donor's total in (country, sector) ÷ donor's total in that country (all sectors). | Operational focus |
| **Donor count** (`cs_donor_count`) | Distinct donors with any positive disbursement in 2022–2024. | Crowding / coordination complexity |

**Why volume alone is not enough.** A large one-time grant differs fundamentally
from sustained programmatic engagement. The framework requires donors to
demonstrate both scale *and* commitment (through persistence or focus) before
receiving a High priority label.

---

## Stage 1 — Country-sector context

**Unit:** country × sector. **Inputs:** `cs_recent_avg` and `cs_donor_count`.

Percentile ranks are computed across all 573 country-sector cells.
Classification uses **sequential rules (first match wins)**:

| Order | Label | Condition | Interpretation |
|---|---|---|---|
| 1 | **Low-activity space** | `disburse_pctile` < P33 | Minimal financing — potential gap or structural barrier |
| 2 | **Fragmented coordination space** | `disburse_pctile` ≥ P33 AND `donor_count_pctile` ≥ P67 | Moderate-to-high financing, many donors — coordination challenge |
| 3 | **Anchor partnership space** | `disburse_pctile` ≥ P67 (and donor count < P67, by exclusion) | Well-funded, concentrated partnerships |
| 4 | **Thin / emerging space** | Everything else (P33 ≤ disburse < P67, donor count < P67) | Some financing, few donors — growth potential |

Low-activity spaces are identified first (bottom third by disbursement,
regardless of donor count). Among the rest, high donor-count spaces are
Fragmented; high-disbursement / low-donor-count spaces are Anchor; the
residual is Thin/emerging.

---

## Stage 2 — Donor priority

**Unit:** donor × country × sector.

### Step 1 — Minimum activity rule

A cell is **assessed** only if both hold:
- At least **1 active year** (disbursement > 0) in the recent 3-year window
- `recent_avg > 0`

Cells failing this rule are labeled **Peripheral / not assessed** and kept
in the output (not dropped). They are excluded from within-country-sector
percentile rankings.

### Step 2 — Indicator flags (assessed cells only)

| Flag | Condition | What it means |
|---|---|---|
| `disbursement_strong` | `recent_avg` ≥ P67 within country-sector (assessed donors only) **AND** ≥ P50 globally across all assessed rows | Top contributor in this space, above global median scale |
| `persistence_strong` | `persistence_active_years` ≥ 4 out of last 5 years **AND** `cv_recent` < 1.0 | Committed recent presence with stable disbursements |
| `embeddedness_strong` | `sector_share` ≥ 0.15 | This sector is ≥ 15% of the donor's country portfolio |

`n_strong` = count of True flags (0, 1, 2, or 3).

**Persistence focuses on the last 5 years (2020–2024)**, not the full
23-year history. A donor that entered the space recently but has been
consistently active and stable is treated equivalently to one with a long
track record. The CV check (`cv_recent < 1.0`) ensures that erratic or
one-off disbursements don't masquerade as sustained engagement.

The **global P50 floor** on disbursement prevents a top-ranked donor in a
very small niche from qualifying on relative position alone.

### Step 3 — Priority rules

| Priority | Rule |
|---|---|
| **High** | `n_strong` ≥ 2 **AND** (`disbursement_strong` OR `persistence_strong`) |
| **Medium** | `n_strong` ≥ 1 (but not qualifying for High) |
| **Low** | `n_strong` = 0 |
| **Peripheral / not assessed** | Did not pass the minimum activity rule |

The High rule requires that at least one of the two qualifying strong signals
is scale or continuity — volume alone (embeddedness only) is not sufficient.

---

## Combined interpretation

The two stages compose into a practical matrix:

| Context | High priority | Medium priority | Low priority |
|---|---|---|---|
| **Anchor** | Flagship partner | Supporting actor | Peripheral presence |
| **Fragmented** | Coordination leader | Part of the crowd | Noise |
| **Thin / emerging** | Dominant player | Occasional contributor | Passing presence |
| **Low-activity** | Solo anchor | Sporadic engagement | Inactive |

"High in an Anchor space" is the strongest partnership signal.
"High in a Low-activity space" identifies a leading donor in a thin market —
important for gap analysis.

---

## CAGR (Descriptive Analysis)

Where we report sector growth, we use **Compound Annual Growth Rate** between
**3-year endpoint averages** to smooth single-year volatility:

$$\\text{CAGR} = \\left(\\frac{\\overline{Y_{end}}}{\\overline{Y_{start}}}\\right)^{1/n} - 1$$

where $n$ = years between the midpoints of the two windows.

---

## Interpreting the dashboard

Each page answers a specific question from the perspective of a development
team scoping partners in their country × sector:

- **🗺️ Country–Sector Mapping** — *"What is the financing landscape in my
  country × sector, and how crowded is it?"*
- **🎯 Priority Framework** — *"Across all donors in my country × sector,
  who ranks High / Medium / Low as a partnership candidate?"*
- **👥 Donor Profiles** — *"Donor X looks promising — where else are they a
  central partner, and how big is their engagement in my space?"*
- **📋 Priority Table** — the full underlying dataset for export, filtering,
  and sharing with colleagues.
            """
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Glossary — alphabetical
    # ─────────────────────────────────────────────────────────────────────────
    with tab_g:
        st.markdown(
            """
## Glossary

**Active share** (`active_share`) — Count of years with disbursement > 0 in the full 23-year window ÷ 23. Ranges 0–1. Shown in the Priority Table for reference, but no longer used as a threshold in `persistence_strong`.

**Active year** — A year in which the donor's annual disbursement in a given country-sector is strictly greater than zero.

**Anchor partnership space** — A country-sector classified as having high total disbursement (≥ P67) and a lower number of active donors (< P67). Established, concentrated, well-funded space.

**CAGR (Compound Annual Growth Rate)** — Annualised growth rate between two points in time. Computed between 3-year endpoint averages to reduce single-year noise.

**Context (Stage 1 label)** — The classification of a country-sector into one of four categories: Anchor, Fragmented coordination, Thin/emerging, or Low-activity. Reflects the financing landscape the World Bank team is operating in.

**CRS (Creditor Reporting System)** — OECD's activity-level database of aid flows. Source of sector breakdowns used in this framework.

**`cv_recent`** — Coefficient of variation of annual disbursements over the recent 3-year window (`std_recent / recent_avg`). Computed for assessed rows only. If ≥ 1.0, `persistence_strong` is forced to False, preventing highly erratic donors from being labelled as persistent partners.

**DAC (Development Assistance Committee)** — The OECD committee whose members are the traditional bilateral donors (e.g. USA, Germany, Japan).

**DAC2A** — OECD's aggregated table of Gross ODA by donor and recipient. Source of authoritative volume totals used in this framework.

**Disbursement strength** (`disbursement_strong`) — True if the donor's `recent_avg` ranks at or above P67 *within* its country-sector (among assessed donors only) **and** at or above P50 globally across all assessed rows. The global P50 floor prevents a top-ranked donor in a very small niche from qualifying on relative position alone.

**Donor count** (`cs_donor_count`) — Number of distinct donors with at least one active year in the recent 3-year window (2022–2024). Used in Stage 1 to characterise coordination complexity.

**Donor type** — Bilateral (DAC), Bilateral (Non-DAC), MDB, UN Agency, Vertical Fund, Other.

**Embeddedness** (`embeddedness_strong`) — True if `sector_share ≥ 0.15` (this sector is at least 15% of the donor's country portfolio). Captures operational focus rather than token presence.

**Fragmented coordination space** — A country-sector with moderate-to-high total disbursement (≥ P33) and many active donors (≥ P67). Many players, coordination is the key challenge.

**Gap** — Informal term for Low-activity spaces where external financing presence is limited.

**High priority** — `n_strong ≥ 2` AND (`disbursement_strong` OR `persistence_strong`). At least two strong signals, including scale or continuity.

**LAC** — Latin America & the Caribbean.

**Low-activity space** — A country-sector with total disbursement below P33, regardless of donor count. Potential financing gap or structural barrier to engagement.

**MDB** — Multilateral Development Bank (e.g. IDB, CAF, World Bank).

**`n_strong`** — Count of True flags among the three Stage 2 indicators (0–3). Determines the priority label.

**ODA (Official Development Assistance)** — Concessional government finance to developing countries, as defined by the OECD DAC. Negative values are valid OECD corrections and are retained throughout.

**Peripheral / not assessed** — A donor-country-sector cell that did not meet the minimum activity rule (< 1 active year in recent window or `recent_avg` ≤ 0). Kept in the output with flags set to False; excluded from within-country-sector rankings.

**Persistence** (`persistence_strong`) — True if the donor was active in at least **4 of the last 5 years** (2020–2024) AND `cv_recent < 1.0` (stable recent disbursements). Focuses entirely on recent committed presence — long-run historical track record is not penalised or rewarded.

**Priority (Stage 2 label)** — High / Medium / Low / Peripheral. Classifies the strategic centrality of a donor in a specific country-sector.

**Recent avg** (`recent_avg`) — Mean annual disbursement over 2022–2024 (recent 3-year window), in USD millions. Negative values retained as valid OECD corrections.

**Recent active years** (`recent_active_years`) — Count of years with disbursement > 0 in the recent 3-year window only. Used in the minimum activity rule and in `persistence_strong`.

**Sector share** (`sector_share`) — Donor's total disbursement in (country, sector) ÷ donor's total disbursement in that country across all sectors (full analysis window). Ranges 0–1.

**Stage 1** — Country-sector context classification. Characterises the financing landscape of each country × sector space.

**Stage 2** — Donor-country-sector priority classification. Evaluates each donor's position within each space.

**`std_recent`** — Standard deviation of annual disbursements over the recent 3-year window. Used to compute `cv_recent`.

**Thin / emerging space** — A country-sector with mid-range disbursement (P33–P67) and few active donors (< P67). Some financing, concentrated among few players — potential strategic entry point.

**Vertical Fund** — A multilateral fund with a specific thematic focus (e.g. Green Climate Fund, Global Fund, GAVI).
            """
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Technical — for curious readers
    # ─────────────────────────────────────────────────────────────────────────
    with tab_tech:
        st.markdown(
            """
## Technical details

Precise thresholds and rules for technically curious readers.

### Fixed assumptions

1. Analysis window: 2002–2024 (23 years).
2. Recent window: 2022–2024 (last 3 years) — used for `recent_avg` and `cv_recent`.
3. Persistence window: 2020–2024 (last 5 years) — used for `persistence_active_years`.
4. Negative disbursements: retained everywhere as valid OECD corrections.
5. No rows are dropped: donors failing the minimum activity rule are labeled
   Peripheral / not assessed, not removed.

### Stage 1 — Percentile thresholds

- `disburse_pctile` = percentile rank of `cs_recent_avg` across all 573
  country-sector cells.
- `donor_count_pctile` = percentile rank of `cs_donor_count` across all cells.
- Cutoffs: **P33** (low) and **P67** (high), configurable in `settings.yaml`.

### Stage 1 — Classification rules (sequential, first match wins)

```
if disburse_pctile < P33:
    → "Low-activity space"

elif disburse_pctile >= P33 and donor_count_pctile >= P67:
    → "Fragmented coordination space"

elif disburse_pctile >= P67:
    → "Anchor partnership space"   # high disburse, low donor count by exclusion

else:
    → "Thin / emerging space"      # P33 <= disburse < P67, donor_count < P67
```

### Stage 2 — Minimum activity rule

A cell is **assessed** if and only if:
- `recent_active_years >= 1` (at least 1 active year in 2022–2024), **and**
- `recent_avg > 0`

All other cells receive **Peripheral / not assessed** (flags = False,
`n_strong` = 0) and are excluded from within-CS percentile rankings.

### Stage 2 — Indicator flags

```python
# Disbursement strength
#   within-CS P67 among assessed donors only
#   AND above global P50 across all assessed rows (disburse_floor_pctile)
disbursement_strong = (
    disburse_pctile_within_cs >= P67_within_cs
    AND recent_avg >= global_P50_assessed
)

# Persistence strength
#   Active in >= 4 of the last 5 years AND stable recent disbursements
#   cv_recent = std_recent / recent_avg  (computed over 3-year recent window)
persistence_strong = (
    persistence_active_years >= 4   # out of last 5 years (2020–2024)
    AND cv_recent < 1.0
)

# Embeddedness strength
embeddedness_strong = sector_share >= 0.15

n_strong = disbursement_strong + persistence_strong + embeddedness_strong
```

### Stage 2 — Priority rules

```python
if n_strong >= 2 and (disbursement_strong or persistence_strong):
    priority = "High"
elif n_strong >= 1:
    priority = "Medium"
elif n_strong == 0:
    priority = "Low"
# (Peripheral already assigned above)
```

The High rule requires that at least one of the two qualifying signals is
scale or continuity — embeddedness alone cannot produce a High.

### Output tables

Two canonical CSV outputs (also saved as `.parquet`):

| Table | Rows | Key columns |
|---|---|---|
| `country_sector_context_table.csv` | 573 (35 countries × 17 sectors) | `cs_context`, `cs_recent_avg`, `cs_donor_count`, `disburse_pctile`, `donor_count_pctile` |
| `donor_country_sector_priority_table.csv` | 8,762 (83 donors × 35 countries × 17 sectors) | `priority`, `n_strong`, `disbursement_strong`, `persistence_strong`, `embeddedness_strong`, `cv_recent`, `cs_context` |

### Configuration

All thresholds (P33, P67, P50 global floor, 0.50, 0.15, 2 years, CV 1.0)
are configurable in `config/settings.yaml`. The pipeline (`main.py`) regenerates
both output tables from raw OECD data with a single command. This dashboard
reads the pre-generated CSVs from `data/` — no calculations run here.
            """
        )
