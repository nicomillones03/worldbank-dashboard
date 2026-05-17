"""
export.py — Save the two canonical output tables.

Final outputs (outputs/tables/):
    country_sector_context_table.csv        Stage 1 result
    donor_country_sector_priority_table.csv Stage 2 result

Both are also saved as .parquet for Power BI.

Intermediate panels are saved to data_processed/ as .parquet for
fast reloading during development.
"""

from pathlib import Path

import pandas as pd


def export_intermediate(
    dcsy: pd.DataFrame,
    dcs: pd.DataFrame,
    cs: pd.DataFrame,
    processed_dir: Path,
) -> None:
    """Save intermediate parquet files for fast reloading."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    dcsy.to_parquet(processed_dir / "donor_country_sector_year.parquet",
                    index=False)
    dcs.to_parquet(processed_dir / "donor_country_sector.parquet",
                   index=False)
    cs.to_parquet(processed_dir / "country_sector.parquet",
                  index=False)

    print(f"  Intermediate parquet saved to {processed_dir}/")


def export_final(
    dcs: pd.DataFrame,
    cs: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Save the two canonical output tables (CSV + Parquet).

    1. country_sector_context_table
       One row per country x sector.

    2. donor_country_sector_priority_table
       One row per donor x country x sector.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Country-sector context table ───────────────────────────────────
    cs_cols = [
        "country_code",
        "country_name",
        "sector",
        "cs_recent_avg",
        "cs_donor_count",
        "disburse_pctile",
        "donor_count_pctile",
        "cs_context",
    ]
    cs_out = cs[[c for c in cs_cols if c in cs.columns]].copy()
    cs_out = cs_out.sort_values(["country_name", "sector"])

    cs_out.to_csv(output_dir / "country_sector_context_table.csv",
                  index=False)
    cs_out.to_parquet(output_dir / "country_sector_context_table.parquet",
                      index=False)
    print(f"  country_sector_context_table: {len(cs_out):,} rows")

    # ── 2. Donor-country-sector priority table ────────────────────────────
    dcs_cols = [
        "donor_code",
        "donor_name",
        "donor_type",
        "country_code",
        "country_name",
        "sector",
        "recent_avg",
        "active_years",
        "active_share",
        "sector_share",
        "disburse_pctile_within_cs",
        "persistence_active_years",
        "cv_recent",
        "disbursement_strong",
        "persistence_strong",
        "embeddedness_strong",
        "n_strong",
        "priority",
        "cs_context",
    ]
    dcs_out = dcs[[c for c in dcs_cols if c in dcs.columns]].copy()
    dcs_out = dcs_out.sort_values(
        ["country_name", "sector", "priority", "recent_avg"],
        ascending=[True, True, True, False],
    )

    dcs_out.to_csv(
        output_dir / "donor_country_sector_priority_table.csv",
        index=False,
    )
    dcs_out.to_parquet(
        output_dir / "donor_country_sector_priority_table.parquet",
        index=False,
    )
    print(f"  donor_country_sector_priority_table: {len(dcs_out):,} rows")

    print(f"\n  All outputs saved to {output_dir}/")


_ACTOR_GROUP_TO_DTYPE = {
    "DAC Bilateral":              "Bilateral (DAC)",
    "Non-DAC Bilateral":          "Bilateral (Non-DAC)",
    "Development Banks":          "MDB",
    "UN System":                  "UN Agency",
    "Agriculture Multilaterals":  "Vertical Fund",
    "Climate Multilaterals":      "Vertical Fund",
    "Environment Multilaterals":  "Vertical Fund",
    "Health Multilaterals":       "Vertical Fund",
    "Other Multilaterals":        "Other",
    "Private Philanthropy":       "Other",
}


def export_descriptive(
    dcsy: pd.DataFrame,
    output_dir: Path,
    volumes_path: Path | None = None,
) -> None:
    """
    Export pre-aggregated time-series tables for the descriptive analysis
    pages of the dashboard.

    Sector-panel tables (from dcsy — CRS sector coverage):
        agg_by_sector_year.csv       total ODA by sector x year

    Volumes-based tables (from oda_volumes.csv — complete DAC2A coverage):
        agg_by_year_vol.csv              total ODA by year
        agg_by_donor_type_year_vol.csv   total ODA by donor_type x year
        agg_by_country_year_vol.csv      total ODA by country x year

    The volumes-based tables use DAC2A as the authoritative source so that
    bilateral donors are present from 2002, avoiding the sector-panel
    coverage gap in 2002–2005.

    volumes_path: path to oda_volumes.csv from the ETL output.  If None,
    the function falls back to deriving all four tables from dcsy (legacy
    behaviour — produces the coverage-gap artefact for 2002–2005).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Sector-panel table (kept as-is — CRS needed for sector breakdown) ─
    by_sector_year = (
        dcsy.groupby(["year", "sector"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total_oda"})
        .sort_values(["year", "sector"])
    )
    by_sector_year.to_csv(output_dir / "agg_by_sector_year.csv", index=False)

    # ── Volumes-based tables ──────────────────────────────────────────────
    if volumes_path is not None and Path(volumes_path).exists():
        vol = pd.read_csv(volumes_path)
        vol["donor_type"]   = vol["actor_group"].map(_ACTOR_GROUP_TO_DTYPE)
        vol["country_name"] = vol["country_name_clean"]
        vol["amount"]       = vol["amount_usd_millions"]

        by_year = (vol.groupby("year")["amount"]
                   .sum().reset_index()
                   .rename(columns={"amount": "total_oda"})
                   .sort_values("year"))

        by_dtype_year = (vol.groupby(["year", "donor_type"])["amount"]
                         .sum().reset_index()
                         .rename(columns={"amount": "total_oda"})
                         .sort_values(["year", "donor_type"]))

        by_country_year = (vol.groupby(["year", "country_name"])["amount"]
                           .sum().reset_index()
                           .rename(columns={"amount": "total_oda"})
                           .sort_values(["year", "country_name"]))

        suffix = "_vol"
        print(f"  Volumes-based tables built from {volumes_path.name}")
    else:
        # Fallback: derive from sector panel (legacy — coverage gap 2002-2005)
        by_year = (dcsy.groupby("year")["amount"]
                   .sum().reset_index()
                   .rename(columns={"amount": "total_oda"})
                   .sort_values("year"))
        by_dtype_year = (dcsy.groupby(["year", "donor_type"])["amount"]
                         .sum().reset_index()
                         .rename(columns={"amount": "total_oda"})
                         .sort_values(["year", "donor_type"]))
        by_country_year = (dcsy.groupby(["year", "country_name"])["amount"]
                           .sum().reset_index()
                           .rename(columns={"amount": "total_oda"})
                           .sort_values(["year", "country_name"]))
        suffix = ""
        print("  WARNING: volumes_path not supplied — falling back to sector panel "
              "(coverage gap in 2002-2005 will affect total/donor/country charts)")

    by_year.to_csv(output_dir / f"agg_by_year{suffix}.csv", index=False)
    by_dtype_year.to_csv(output_dir / f"agg_by_donor_type_year{suffix}.csv", index=False)
    by_country_year.to_csv(output_dir / f"agg_by_country_year{suffix}.csv", index=False)

    print(f"  Descriptive tables saved to {output_dir}/")
    print(f"    agg_by_year{suffix}:              {len(by_year):,} rows")
    print(f"    agg_by_donor_type_year{suffix}:   {len(by_dtype_year):,} rows")
    print(f"    agg_by_sector_year:              {len(by_sector_year):,} rows")
    print(f"    agg_by_country_year{suffix}:      {len(by_country_year):,} rows")
