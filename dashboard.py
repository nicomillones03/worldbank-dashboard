"""
dashboard.py — Interactive Priority-Engine Dashboard (Streamlit + Plotly)

Launch:
    cd priority_engine/
    streamlit run dashboard.py

Reads the two canonical CSV tables from outputs/tables/.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LAC ODA Partnership Priority Dashboard",
    page_icon=":earth_americas:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ──────────────────────────────────────────────────────────────────
TABLE_DIR = Path(__file__).parent / "data"
CS_FILE = TABLE_DIR / "country_sector_context_table.csv"
DCS_FILE = TABLE_DIR / "donor_country_sector_priority_table.csv"

# ─── Colour palettes (consistent with static figures) ──────────────────────
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
PRIORITY_ORDER = ["High", "Medium", "Low", "Peripheral / not assessed"]
CONTEXT_ORDER = [
    "Anchor partnership space",
    "Fragmented coordination space",
    "Thin / emerging space",
    "Low-activity space",
]


# ─── Data loading (cached) ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    cs = pd.read_csv(CS_FILE)
    dcs = pd.read_csv(DCS_FILE)
    # Ordered categoricals for consistent sort
    dcs["priority"] = pd.Categorical(
        dcs["priority"], categories=PRIORITY_ORDER, ordered=True
    )
    dcs["cs_context"] = pd.Categorical(
        dcs["cs_context"], categories=CONTEXT_ORDER, ordered=True
    )
    cs["cs_context"] = pd.Categorical(
        cs["cs_context"], categories=CONTEXT_ORDER, ordered=True
    )
    return cs, dcs


cs_raw, dcs_raw = load_data()

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Filters")

# Country
all_countries = sorted(dcs_raw["country_name"].unique())
sel_countries = st.sidebar.multiselect(
    "Country", all_countries, default=[], placeholder="All countries"
)

# Sector
all_sectors = sorted(dcs_raw["sector"].unique())
sel_sectors = st.sidebar.multiselect(
    "Sector", all_sectors, default=[], placeholder="All sectors"
)

# Donor type
all_dtypes = sorted(dcs_raw["donor_type"].unique())
sel_dtypes = st.sidebar.multiselect(
    "Donor type", all_dtypes, default=[], placeholder="All types"
)

# Priority
sel_priorities = st.sidebar.multiselect(
    "Priority", PRIORITY_ORDER, default=[], placeholder="All priorities"
)

# Context
sel_contexts = st.sidebar.multiselect(
    "Context", CONTEXT_ORDER, default=[], placeholder="All contexts"
)

# Peripheral toggle
hide_peripheral = st.sidebar.checkbox("Hide Peripheral / not assessed", value=False)


# ─── Apply filters ──────────────────────────────────────────────────────────
def apply_filters(df, is_cs=False):
    """Filter a dataframe based on sidebar selections."""
    mask = pd.Series(True, index=df.index)
    if sel_countries:
        mask &= df["country_name"].isin(sel_countries)
    if sel_sectors:
        mask &= df["sector"].isin(sel_sectors)
    if not is_cs:
        if sel_dtypes:
            mask &= df["donor_type"].isin(sel_dtypes)
        if sel_priorities:
            mask &= df["priority"].isin(sel_priorities)
        if hide_peripheral:
            mask &= df["priority"] != "Peripheral / not assessed"
    if sel_contexts:
        mask &= df["cs_context"].isin(sel_contexts)
    return df[mask].copy()


dcs = apply_filters(dcs_raw)
cs = apply_filters(cs_raw, is_cs=True)

# ─── Navigation ─────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Country-Sector Context",
        "Priority Explorer",
        "Donor Profiles",
        "Priority Table",
    ],
)

# ─── Helpers ────────────────────────────────────────────────────────────────
def fmt_millions(v):
    if abs(v) >= 1000:
        return f"${v / 1000:,.1f}B"
    return f"${v:,.1f}M"


def short_ctx(label):
    return str(label).replace(" space", "").replace(" partnership", "")


# =====================================================================
# PAGE: Overview
# =====================================================================
if page == "Overview":
    st.title("LAC ODA Partnership Priority Dashboard")
    st.caption(
        "Two-stage framework: country-sector context classification + "
        "donor priority assignment.  Source: OECD CRS / DAC2A, 2002-2024."
    )

    # ── KPI row ──────────────────────────────────────────────────────────
    assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Countries", f"{dcs['country_name'].nunique()}")
    c2.metric("Sectors", f"{dcs['sector'].nunique()}")
    c3.metric("Donors", f"{dcs['donor_name'].nunique()}")
    c4.metric("Assessed partnerships", f"{len(assessed):,}")
    c5.metric("High-priority", f"{(assessed['priority'] == 'High').sum():,}")

    st.divider()
    col_left, col_right = st.columns(2)

    # ── Priority distribution ────────────────────────────────────────────
    with col_left:
        st.subheader("Priority distribution")
        pri_counts = (
            dcs["priority"]
            .value_counts()
            .reindex(PRIORITY_ORDER)
            .reset_index()
        )
        pri_counts.columns = ["Priority", "Count"]
        fig_pri = px.bar(
            pri_counts,
            x="Priority",
            y="Count",
            color="Priority",
            color_discrete_map=PRIORITY_COLOURS,
            text_auto=True,
        )
        fig_pri.update_layout(showlegend=False, yaxis_title="Donor-country-sector rows")
        st.plotly_chart(fig_pri, use_container_width=True)

    # ── Context distribution ─────────────────────────────────────────────
    with col_right:
        st.subheader("Context distribution")
        ctx_counts = (
            cs["cs_context"]
            .value_counts()
            .reindex(CONTEXT_ORDER)
            .reset_index()
        )
        ctx_counts.columns = ["Context", "Count"]
        fig_ctx = px.bar(
            ctx_counts,
            x="Context",
            y="Count",
            color="Context",
            color_discrete_map=CONTEXT_COLOURS,
            text_auto=True,
        )
        fig_ctx.update_layout(showlegend=False, yaxis_title="Country-sector cells")
        fig_ctx.update_xaxes(
            ticktext=[short_ctx(c) for c in CONTEXT_ORDER],
            tickvals=CONTEXT_ORDER,
        )
        st.plotly_chart(fig_ctx, use_container_width=True)

    # ── Context x Priority cross-tab ─────────────────────────────────────
    st.subheader("Context x Priority cross-tabulation")
    assessed_only = dcs[dcs["priority"] != "Peripheral / not assessed"]
    if len(assessed_only) > 0:
        xtab = pd.crosstab(
            assessed_only["cs_context"],
            assessed_only["priority"],
            margins=True,
        )
        # Reorder
        cols = [c for c in PRIORITY_ORDER if c in xtab.columns] + ["All"]
        rows = [r for r in CONTEXT_ORDER if r in xtab.index] + ["All"]
        xtab = xtab.reindex(index=rows, columns=cols, fill_value=0)
        st.dataframe(xtab, use_container_width=True)
    else:
        st.info("No assessed rows with current filters.")

    # ── Priority by donor type ───────────────────────────────────────────
    st.subheader("Priority breakdown by donor type")
    if len(assessed_only) > 0:
        type_pri = (
            assessed_only.groupby(["donor_type", "priority"])
            .size()
            .reset_index(name="count")
        )
        fig_type = px.bar(
            type_pri,
            x="donor_type",
            y="count",
            color="priority",
            color_discrete_map=PRIORITY_COLOURS,
            barmode="stack",
            category_orders={"priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"]},
        )
        fig_type.update_layout(
            xaxis_title="Donor type",
            yaxis_title="Assessed partnerships",
            legend_title="Priority",
        )
        st.plotly_chart(fig_type, use_container_width=True)


# =====================================================================
# PAGE: Country-Sector Context
# =====================================================================
elif page == "Country-Sector Context":
    st.title("Stage 1: Country-Sector Context")
    st.caption(
        "Each cell is a country-sector space classified by total recent "
        "disbursement and donor count."
    )

    # ── Heatmap ──────────────────────────────────────────────────────────
    st.subheader("Context heatmap")
    ctx_num = {c: i for i, c in enumerate(CONTEXT_ORDER)}
    cs_hm = cs.copy()
    cs_hm["ctx_num"] = cs_hm["cs_context"].map(ctx_num)

    pivot = cs_hm.pivot_table(
        index="country_name",
        columns="sector",
        values="ctx_num",
        aggfunc="first",
    )
    # Sort countries alphabetically
    pivot = pivot.sort_index(ascending=True)

    # Build a custom heatmap with plotly graph_objects
    ctx_labels = {i: short_ctx(c) for c, i in ctx_num.items()}
    colour_scale = [
        [0.0, CONTEXT_COLOURS[CONTEXT_ORDER[0]]],
        [0.33, CONTEXT_COLOURS[CONTEXT_ORDER[1]]],
        [0.66, CONTEXT_COLOURS[CONTEXT_ORDER[2]]],
        [1.0, CONTEXT_COLOURS[CONTEXT_ORDER[3]]],
    ]

    hover_text = pivot.copy()
    for col in hover_text.columns:
        hover_text[col] = hover_text[col].map(
            lambda v: ctx_labels.get(v, "") if pd.notna(v) else ""
        )

    fig_hm = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            customdata=hover_text.values,
            hovertemplate="<b>%{y}</b> x %{x}<br>%{customdata}<extra></extra>",
            colorscale=colour_scale,
            zmin=0,
            zmax=3,
            showscale=False,
        )
    )
    fig_hm.update_layout(
        height=max(500, len(pivot) * 22),
        xaxis=dict(tickangle=45, side="top"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200, t=120),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Legend
    legend_md = " | ".join(
        f"**{short_ctx(c)}** = {c}" for c in CONTEXT_ORDER
    )
    st.caption(legend_md)

    # ── Scatter: Disbursement vs Donor Count ─────────────────────────────
    st.subheader("Disbursement vs Donor Count (country-sector)")
    fig_scatter = px.scatter(
        cs,
        x="cs_donor_count",
        y="cs_recent_avg",
        color="cs_context",
        color_discrete_map=CONTEXT_COLOURS,
        hover_data=["country_name", "sector"],
        log_y=True,
        category_orders={"cs_context": CONTEXT_ORDER},
        labels={
            "cs_donor_count": "Active donors (recent window)",
            "cs_recent_avg": "Recent avg disbursement ($M, log)",
            "cs_context": "Context",
        },
    )
    fig_scatter.update_layout(height=550)
    st.plotly_chart(fig_scatter, use_container_width=True)


# =====================================================================
# PAGE: Priority Explorer
# =====================================================================
elif page == "Priority Explorer":
    st.title("Stage 2: Priority Explorer")
    st.caption(
        "Explore donor-country-sector priority assignments.  "
        "Use sidebar filters to drill down."
    )

    assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]

    # ── Top partners ─────────────────────────────────────────────────────
    st.subheader("Top High-priority partnerships by disbursement")
    high = assessed[assessed["priority"] == "High"].sort_values(
        "recent_avg", ascending=False
    )
    if len(high) > 0:
        top_n = st.slider("Show top N", 10, 50, 20, key="top_n")
        top = high.head(top_n)
        top_display = top[
            ["donor_name", "country_name", "sector", "recent_avg",
             "active_share", "sector_share", "n_strong", "cs_context"]
        ].copy()
        top_display["recent_avg"] = top_display["recent_avg"].apply(
            lambda v: f"${v:,.2f}M"
        )
        top_display["active_share"] = (top_display["active_share"] * 100).round(0).astype(int).astype(str) + "%"
        top_display["sector_share"] = (top_display["sector_share"] * 100).round(1).astype(str) + "%"
        top_display.columns = [
            "Donor", "Country", "Sector", "Recent Avg ($M)",
            "Active Share", "Sector Share", "Strong Flags", "Context",
        ]
        st.dataframe(top_display.reset_index(drop=True), use_container_width=True, height=min(700, 35 * top_n + 38))
    else:
        st.info("No High-priority rows with current filters.")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Priority by sector ───────────────────────────────────────────────
    with col_left:
        st.subheader("Priority by sector")
        if len(assessed) > 0:
            sec_pri = (
                assessed.groupby(["sector", "priority"])
                .size()
                .reset_index(name="count")
            )
            fig_sec = px.bar(
                sec_pri,
                y="sector",
                x="count",
                color="priority",
                color_discrete_map=PRIORITY_COLOURS,
                orientation="h",
                barmode="stack",
                category_orders={
                    "priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"],
                },
            )
            fig_sec.update_layout(
                height=500,
                yaxis=dict(categoryorder="total ascending"),
                xaxis_title="Assessed partnerships",
                yaxis_title="",
                legend_title="Priority",
            )
            st.plotly_chart(fig_sec, use_container_width=True)

    # ── Priority by country ──────────────────────────────────────────────
    with col_right:
        st.subheader("Priority by country")
        if len(assessed) > 0:
            cty_pri = (
                assessed.groupby(["country_name", "priority"])
                .size()
                .reset_index(name="count")
            )
            fig_cty = px.bar(
                cty_pri,
                y="country_name",
                x="count",
                color="priority",
                color_discrete_map=PRIORITY_COLOURS,
                orientation="h",
                barmode="stack",
                category_orders={
                    "priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"],
                },
            )
            fig_cty.update_layout(
                height=max(500, dcs["country_name"].nunique() * 20),
                yaxis=dict(categoryorder="total ascending"),
                xaxis_title="Assessed partnerships",
                yaxis_title="",
                legend_title="Priority",
            )
            st.plotly_chart(fig_cty, use_container_width=True)

    # ── Indicator scatter ────────────────────────────────────────────────
    st.divider()
    st.subheader("Indicator scatter (assessed donors)")
    if len(assessed) > 0:
        x_col = st.selectbox("X axis", ["recent_avg", "active_share", "sector_share"], index=0)
        y_col = st.selectbox("Y axis", ["active_share", "recent_avg", "sector_share"], index=0)
        fig_ind = px.scatter(
            assessed,
            x=x_col,
            y=y_col,
            color="priority",
            color_discrete_map=PRIORITY_COLOURS,
            hover_data=["donor_name", "country_name", "sector"],
            opacity=0.6,
            category_orders={"priority": [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"]},
            labels={
                "recent_avg": "Recent avg disbursement ($M)",
                "active_share": "Active years share",
                "sector_share": "Sector share",
            },
        )
        fig_ind.update_layout(height=550)
        st.plotly_chart(fig_ind, use_container_width=True)


# =====================================================================
# PAGE: Donor Profiles
# =====================================================================
elif page == "Donor Profiles":
    st.title("Donor Profiles")
    st.caption("Deep-dive into a specific donor's partnerships across countries and sectors.")

    # Donor selector
    assessed = dcs[dcs["priority"] != "Peripheral / not assessed"]
    donor_options = sorted(assessed["donor_name"].unique())

    if len(donor_options) == 0:
        st.warning("No assessed donors with current filters.")
    else:
        sel_donor = st.selectbox("Select donor", donor_options)
        donor_df = assessed[assessed["donor_name"] == sel_donor].copy()

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Countries", donor_df["country_name"].nunique())
        c2.metric("Sectors", donor_df["sector"].nunique())
        c3.metric(
            "Total recent avg",
            fmt_millions(donor_df["recent_avg"].sum()),
        )
        c4.metric("High-priority", (donor_df["priority"] == "High").sum())

        st.divider()

        # Priority breakdown for this donor
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Priority breakdown")
            dp = donor_df["priority"].value_counts().reindex(
                [p for p in PRIORITY_ORDER if p != "Peripheral / not assessed"]
            ).fillna(0).reset_index()
            dp.columns = ["Priority", "Count"]
            fig_dp = px.pie(
                dp, values="Count", names="Priority",
                color="Priority", color_discrete_map=PRIORITY_COLOURS,
                hole=0.4,
            )
            fig_dp.update_traces(textinfo="value+percent")
            st.plotly_chart(fig_dp, use_container_width=True)

        with col_r:
            st.subheader("Context breakdown")
            dc = donor_df["cs_context"].value_counts().reindex(CONTEXT_ORDER).fillna(0).reset_index()
            dc.columns = ["Context", "Count"]
            fig_dc = px.pie(
                dc, values="Count", names="Context",
                color="Context", color_discrete_map=CONTEXT_COLOURS,
                hole=0.4,
            )
            fig_dc.update_traces(textinfo="value+percent")
            st.plotly_chart(fig_dc, use_container_width=True)

        # Heatmap: country x sector for this donor, coloured by priority
        st.subheader(f"{sel_donor} — Country x Sector priority map")
        pri_num = {"High": 3, "Medium": 2, "Low": 1}
        donor_hm = donor_df.copy()
        donor_hm["pri_num"] = donor_hm["priority"].map(pri_num).fillna(0)

        pivot_d = donor_hm.pivot_table(
            index="country_name", columns="sector",
            values="pri_num", aggfunc="first",
        )
        pivot_d = pivot_d.sort_index()

        hover_d = donor_hm.pivot_table(
            index="country_name", columns="sector",
            values="priority", aggfunc="first",
        )

        pri_colourscale = [
            [0.0, "#eeeeee"],
            [0.33, PRIORITY_COLOURS["Low"]],
            [0.66, PRIORITY_COLOURS["Medium"]],
            [1.0, PRIORITY_COLOURS["High"]],
        ]

        fig_dhm = go.Figure(
            data=go.Heatmap(
                z=pivot_d.values,
                x=pivot_d.columns.tolist(),
                y=pivot_d.index.tolist(),
                customdata=hover_d.values,
                hovertemplate="<b>%{y}</b> x %{x}<br>%{customdata}<extra></extra>",
                colorscale=pri_colourscale,
                zmin=0, zmax=3,
                showscale=False,
            )
        )
        fig_dhm.update_layout(
            height=max(350, len(pivot_d) * 24),
            xaxis=dict(tickangle=45, side="top"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=200, t=120),
        )
        st.plotly_chart(fig_dhm, use_container_width=True)

        # Detail table
        st.subheader("All partnerships")
        detail = donor_df[
            ["country_name", "sector", "recent_avg", "active_share",
             "sector_share", "n_strong", "priority", "cs_context"]
        ].sort_values(["priority", "recent_avg"], ascending=[True, False])
        detail_show = detail.copy()
        detail_show["recent_avg"] = detail_show["recent_avg"].apply(lambda v: f"${v:,.2f}M")
        detail_show["active_share"] = (detail_show["active_share"] * 100).round(0).astype(int).astype(str) + "%"
        detail_show["sector_share"] = (detail_show["sector_share"] * 100).round(1).astype(str) + "%"
        detail_show.columns = [
            "Country", "Sector", "Recent Avg", "Active Share",
            "Sector Share", "Strong Flags", "Priority", "Context",
        ]
        st.dataframe(detail_show.reset_index(drop=True), use_container_width=True, height=600)


# =====================================================================
# PAGE: Priority Table
# =====================================================================
elif page == "Priority Table":
    st.title("Full Priority Table")
    st.caption(
        "Browse and download the complete donor-country-sector priority "
        "table. Use sidebar filters to narrow results."
    )

    # Display options
    col_a, col_b = st.columns(2)
    with col_a:
        sort_col = st.selectbox(
            "Sort by",
            ["recent_avg", "n_strong", "active_share", "sector_share", "donor_name"],
            index=0,
        )
    with col_b:
        sort_asc = st.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Ascending"

    display = dcs.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

    # Format for display
    show = display[
        ["donor_name", "donor_type", "country_name", "sector",
         "recent_avg", "active_years", "active_share", "sector_share",
         "disbursement_strong", "persistence_strong", "embeddedness_strong",
         "n_strong", "priority", "cs_context"]
    ].copy()
    show["recent_avg"] = show["recent_avg"].apply(lambda v: f"${v:,.2f}M")
    show["active_share"] = (show["active_share"] * 100).round(0).astype(int).astype(str) + "%"
    show["sector_share"] = (show["sector_share"] * 100).round(1).astype(str) + "%"
    show.columns = [
        "Donor", "Type", "Country", "Sector",
        "Recent Avg", "Active Yrs", "Active %",
        "Sector %", "Disburse Flag", "Persist Flag", "Embed Flag",
        "Strong", "Priority", "Context",
    ]

    st.dataframe(show, use_container_width=True, height=700)

    st.download_button(
        "Download filtered table (CSV)",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="priority_table_filtered.csv",
        mime="text/csv",
    )

    # Summary stats
    st.divider()
    st.subheader("Summary statistics (filtered)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", f"{len(display):,}")
    assessed_d = display[display["priority"] != "Peripheral / not assessed"]
    c2.metric("Assessed", f"{len(assessed_d):,}")
    c3.metric("High-priority", f"{(assessed_d['priority'] == 'High').sum():,}")
    c4.metric(
        "Total recent avg ($M)",
        fmt_millions(assessed_d["recent_avg"].sum()),
    )
