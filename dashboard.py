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
        "Stage 2 engagement level: High = donor is a core partner in that "
        "country-sector (strong on ≥2 of 3 indicators); Medium = strong on 1; "
        "Low = assessed but not strong on any; Peripheral = not assessed "
        "(too small or too short-lived to evaluate)."
    ),
    "context": (
        "Stage 1 country-sector classification based on total disbursement "
        "and number of active donors:\n"
        "• Anchor = large volume + many donors\n"
        "• Fragmented = many donors but small volume each\n"
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
- **Anchor partnership space** — large volume **and** many donors (established, well-funded).
- **Fragmented coordination space** — many donors but no single one dominates volume-wise.
- **Thin / emerging space** — few donors, growing activity; strategic entry opportunity.
- **Low-activity space** — limited disbursements and few donors; potential financing gap.

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
1. **Disbursement strength** — is the donor in the top tier of disbursement for this country-sector?
2. **Persistence** — has the donor been consistently active across years?
3. **Embeddedness** — does this country-sector represent a meaningful share of the donor's portfolio?

The number of "strong" indicators (`n_strong`) determines priority:
- **High** → strong on ≥2 of 3 indicators
- **Medium** → strong on exactly 1
- **Low** → assessed but strong on 0
- **Peripheral / not assessed** → too small / too short-lived to assess

See the **📚 Methodology** page for the precise thresholds.
            """
        )

    assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assessed partnerships", f"{len(assessed):,}",
              help="Donor-country-sector cells that passed the Stage 2 assessment thresholds.")
    c2.metric("High priority", f"{(assessed['priority']=='High').sum():,}", help=HELP["priority"])
    c3.metric("Medium priority", f"{(assessed['priority']=='Medium').sum():,}", help=HELP["priority"])
    c4.metric("Low priority", f"{(assessed['priority']=='Low').sum():,}", help=HELP["priority"])

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
    st.caption("Visualise how the three Stage 2 indicators relate across partnerships. Colour = priority.")
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

## Data sources

- **OECD Creditor Reporting System (CRS)** — donor × recipient × sector × year
  activity-level records. Source of sector composition.
- **OECD DAC2A (Gross ODA)** — donor × recipient × year authoritative volume
  totals. Used to anchor absolute disbursement amounts.
- **World Bank WDI (GDP)** — for contextual normalisation (exploratory only).

We combine the two OECD sources using an **allocation method**: DAC2A provides
the total envelope, and CRS shares determine how that envelope is split across
sectors. This preserves headline volumes while keeping sectoral detail.

**Scope:** Latin America & the Caribbean, 35 countries. Regional aggregates
(e.g. "America (Regional)") are **excluded** from country-level analysis.

---

## Stage 1 — Classifying country-sector context

For every **country × sector** space, we compute two things:

1. **Recent avg disbursement** — mean annual ODA over the last 3 years (2022–2024), in USD millions.
2. **Active donor count** — number of distinct donors that disbursed in the recent window.

We then rank each country-sector on these two dimensions (percentile ranks
across LAC) and assign one of four **context labels**:

| Context | Disbursement | Donor count | Interpretation |
|---|---|---|---|
| **Anchor partnership space** | High (≥ P66) | High (≥ P66) | Established, well-funded space |
| **Fragmented coordination space** | Low | High | Many donors but small volume each — coordination challenge |
| **Thin / emerging space** | High | Low | Concentrated financing from few donors — strategic entry point |
| **Low-activity space** | Low (< P33) | Low | Limited external financing — potential gap |

---

## Stage 2 — Classifying donor priority

For every **donor × country × sector** cell that meets the assessment
thresholds, we evaluate three indicators:

1. **Disbursement strength** — is this donor in the top of the disbursement
   distribution *within this country-sector*?
2. **Persistence** — has the donor been active (disbursed > 0) in a high share
   of the analysis years?
3. **Embeddedness** — does this country-sector represent a meaningful share of
   the donor's total LAC portfolio?

Each indicator is a boolean (strong / not strong). The number of "strong"
indicators — `n_strong` — determines the priority label:

| `n_strong` | Priority |
|---|---|
| 2 or 3 | **High** |
| 1 | **Medium** |
| 0 | **Low** |
| (not assessed) | **Peripheral / not assessed** |

A cell is **Peripheral** when it's too small or too short-lived to assess
(e.g. a donor that disbursed once, years ago, in trivial amounts).

---

## CAGR calculation

Where we report sector growth, we use **Compound Annual Growth Rate** between
**3-year endpoint averages** — e.g. the average of 2002–2004 vs. the average of
2022–2024 — to smooth single-year volatility.

Formula:
$$\\text{CAGR} = \\left(\\frac{\\overline{Y_{end}}}{\\overline{Y_{start}}}\\right)^{1/n} - 1$$

where $n$ is the number of years between the midpoints of the two windows.

---

## Interpreting the dashboard

Each page answers a specific question from the perspective of a development
team scoping partners in their country × sector (phrased from the World Bank's
point of view, but the same logic applies to any other organisation):

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

**Active share** — Share of years in the analysis window where a donor disbursed > 0 in a given country-sector. A persistence measure.

**Anchor partnership space** — A country-sector with both high total disbursement and a high number of active donors. Established, well-funded space.

**CAGR (Compound Annual Growth Rate)** — Annualised growth rate between two points in time. We compute it between 3-year endpoint averages to reduce single-year noise.

**Context (Stage 1 label)** — The classification of a country-sector into one of four categories: Anchor, Fragmented coordination, Thin/emerging, or Low-activity.

**CRS (Creditor Reporting System)** — OECD's activity-level database of aid flows. Source of sector breakdowns.

**DAC (Development Assistance Committee)** — The OECD committee whose members are the traditional bilateral donors.

**DAC2A** — OECD's aggregated table of Gross ODA by donor and recipient. Source of authoritative volume totals.

**Disbursement strength** — One of the three Stage 2 indicators. True if the donor is in the top tier of disbursement for this country-sector.

**Donor type** — Classification of donors into categories: Bilateral (DAC), Bilateral (Non-DAC), MDB (multilateral development bank), UN Agency, Vertical Fund (climate/health-specific funds), Other.

**Embeddedness** — One of the three Stage 2 indicators. True if this country-sector represents a meaningful share of the donor's total portfolio.

**Fragmented coordination space** — A country-sector with many active donors but low total disbursement. Suggests dispersed, small-scale engagement.

**Gap** — Used informally for Low-activity spaces and for country-sectors where external financing presence is thin.

**High priority** — A donor-country-sector partnership where the donor is strong on ≥2 of the 3 Stage 2 indicators.

**LAC** — Latin America & the Caribbean.

**Low-activity space** — A country-sector with both low disbursement and few donors. Potential financing gap.

**MDB** — Multilateral Development Bank (e.g. IDB, CAF, World Bank).

**`n_strong`** — The number of Stage 2 indicators (0–3) on which a donor is "strong". Determines the priority label.

**ODA (Official Development Assistance)** — Concessional government finance to developing countries, as defined by the OECD DAC.

**Peripheral / not assessed** — A donor-country-sector cell that failed the Stage 2 assessment thresholds (too small or too short-lived). Not ranked.

**Persistence** — One of the three Stage 2 indicators. True if the donor has been active in a high share of the analysis years.

**Priority (Stage 2 label)** — The classification of a donor-country-sector partnership into High / Medium / Low / Peripheral.

**Recent avg** — Average annual disbursement over the last 3 years of the analysis window (2022–2024), in USD millions.

**Sector share** — A donor's share of total disbursement in a given country-sector. A volume-relative measure.

**Stage 1** — The country-sector context classification.

**Stage 2** — The donor-country-sector priority classification.

**Thin / emerging space** — A country-sector with high disbursement but few donors. Concentrated financing, often a strategic entry point.

**Vertical Fund** — A multilateral fund with a specific thematic focus (e.g. Global Fund, GAVI, Green Climate Fund).
            """
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Technical — for curious readers
    # ─────────────────────────────────────────────────────────────────────────
    with tab_tech:
        st.markdown(
            """
## Technical details

For readers who want to understand the precise thresholds and formulas.

### Percentile cutoffs

- **Disbursement percentile** (`disburse_pctile`) — rank of `cs_recent_avg`
  across all LAC country-sectors.
- **Donor count percentile** (`donor_count_pctile`) — rank of `cs_donor_count`
  across all LAC country-sectors.
- **Cutoffs:** P33 (low threshold) and P66 (high threshold).

### Stage 1 decision rule

```
if disburse_pctile ≥ P66 and donor_count_pctile ≥ P66:
    → "Anchor partnership space"
elif donor_count_pctile ≥ P66:
    → "Fragmented coordination space"
elif disburse_pctile ≥ P66:
    → "Thin / emerging space"
elif disburse_pctile < P33 and donor_count_pctile < P33:
    → "Low-activity space"
else:
    → (intermediate, absorbed into nearest label)
```

### Stage 2 indicator definitions

- **`disbursement_strong`** — True if the donor ranks in the top percentile
  tier of `recent_avg` *within* its country-sector.
- **`persistence_strong`** — True if `active_share ≥ high threshold`
  (e.g. donor active in most years of the window).
- **`embeddedness_strong`** — True if `sector_share ≥ high threshold`
  (donor's own portfolio concentrated here).

### Stage 2 priority assignment

```
n_strong = disbursement_strong + persistence_strong + embeddedness_strong

if n_strong ≥ 2:    priority = "High"
elif n_strong == 1: priority = "Medium"
elif n_strong == 0: priority = "Low"
else:               priority = "Peripheral / not assessed"
```

### Assessment eligibility

A donor-country-sector cell is **assessed** (receives High/Medium/Low) only if:
- `recent_avg > 0`, **and**
- `active_years ≥ 2`, **and**
- a minimum total disbursement threshold is met.

Cells below these thresholds are tagged **Peripheral / not assessed**.

### Configuration

All thresholds are defined in `priority_engine/config/settings.yaml` and can
be adjusted without touching dashboard code. The analytical pipeline lives in
`priority_engine/src/` and produces the CSV outputs that this dashboard reads.

### Source code

- `src/classify.py` — Stage 1 and Stage 2 classification logic
- `src/panels.py` — panel construction (donor × country × sector × year)
- `src/visuals.py` — static figure generation (for reports)
- `dashboard.py` — this Streamlit app (display only; no calculations here)

See `methodology.md` in the repository root for the fully documented spec.
            """
        )
