#!/usr/bin/env python3
"""
main.py — Priority Engine for World Bank LAC ODA Capstone

End-to-end pipeline:
    1. Load and clean data (from Python ETL outputs)
    2. Build analysis panels (3 levels of aggregation)
    3. Stage 1: Classify country-sector context
    4. Stage 2: Assign donor priority levels
    5. Validate results
    6. Export tables and figures

Usage:
    cd priority_engine/
    python main.py

Configuration:
    Edit config/settings.yaml to change thresholds, paths, and parameters.
    Edit config/sector_mapping.csv to modify the 17-sector classification.
"""

import sys
import time
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.ingest import load_settings, load_sectors, load_volumes, load_gdp
from src.panels import (
    build_donor_country_sector_year,
    build_donor_country_sector,
    build_country_sector,
)
from src.classify import (
    classify_country_sector,
    classify_donor_priority,
    merge_context_to_donors,
)
from src.validate import run_all_checks, summarise_panels
from src.export import export_intermediate, export_final, export_descriptive
from src.visuals import generate_all_figures


def main() -> None:
    t_start = time.time()
    config_dir = Path(__file__).parent / "config"
    settings = load_settings(config_dir / "settings.yaml")

    print("=" * 65)
    print(" PRIORITY ENGINE — World Bank LAC ODA Capstone")
    print("=" * 65)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n--- Step 1: Loading data ---")
    sectors = load_sectors(settings, config_dir)
    volumes = load_volumes(settings)  # optional, for validation
    gdp = load_gdp(settings)          # optional, exploratory

    # ── Step 2: Build panels ──────────────────────────────────────────────
    print("\n--- Step 2: Building panels ---")
    dcsy = build_donor_country_sector_year(sectors)
    dcs = build_donor_country_sector(dcsy, settings)
    cs = build_country_sector(dcsy, settings)

    summarise_panels(dcsy, dcs, cs)

    # ── Step 3: Stage 1 — Country-sector context ─────────────────────────
    print("\n--- Step 3: Stage 1 classification ---")
    cs = classify_country_sector(cs, settings)

    # ── Step 4: Stage 2 — Donor priority ─────────────────────────────────
    print("\n--- Step 4: Stage 2 classification ---")
    dcs = classify_donor_priority(dcs, settings)

    # ── Merge context into donor panel ────────────────────────────────────
    dcs = merge_context_to_donors(dcs, cs)

    # ── Step 5: Validate ──────────────────────────────────────────────────
    print("\n--- Step 5: Validation ---")
    report = run_all_checks(sectors, dcsy, dcs, cs, settings)

    # ── Step 6: Export ────────────────────────────────────────────────────
    print("\n--- Step 6: Exporting outputs ---")
    processed_dir = Path(settings["paths"]["processed_dir"])
    output_tables = Path(settings["paths"]["output_tables_dir"])
    output_figures = Path(settings["paths"]["output_figures_dir"])

    export_intermediate(dcsy, dcs, cs, processed_dir)
    export_final(dcs, cs, output_tables)
    volumes_path = Path(settings["paths"]["etl_output_dir"]) / "oda_volumes.csv"
    export_descriptive(dcsy, output_tables, volumes_path=volumes_path)

    # ── Step 7: Figures ───────────────────────────────────────────────────
    print("\n--- Step 7: Generating figures ---")
    generate_all_figures(cs, dcs, output_figures)

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("=" * 65)
    print(f" Pipeline complete in {elapsed:.1f} seconds")
    print(f" Tables  -> {output_tables}/")
    print(f" Figures -> {output_figures}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
