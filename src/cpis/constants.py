"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ARID_SHP = PROJECT_ROOT / "Africa_Arid_Regions_All-shp" / "Africa_Arid_Regions_All-shp.shp"
DEFAULT_REGION_GEOJSON = PROJECT_ROOT / "assets" / "regions" / "arid_ssa.geojson"
DEFAULT_REGION_SUMMARY = PROJECT_ROOT / "assets" / "regions" / "arid_ssa_summary.json"
DEFAULT_SSA_COUNTRIES_CONFIG = PROJECT_ROOT / "configs" / "ssa_countries.yaml"

DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_NEW_METHOD_DIR = DEFAULT_RUNS_DIR / "new_method"
DEFAULT_RSE2023_2015_RUN_DIR = DEFAULT_NEW_METHOD_DIR / "rse2023_2015_v1"

DEFAULT_ANCHOR_2000 = PROJECT_ROOT / "Africa_CPIS-shp" / "Africa_CPIS_2000.shp"
DEFAULT_ANCHOR_2021 = PROJECT_ROOT / "Africa_CPIS-shp" / "Africa_CPIS_2021.shp"
DEFAULT_EQUAL_AREA_CRS = "EPSG:6933"

DEFAULT_FINAL_SHAPEFILE_TEMPLATE = "cpis_arid_ssa_{year}.shp"
DEFAULT_FINAL_CENTERS_TEMPLATE = "cpis_arid_ssa_{year}_centers.shp"

EXPORT_STATUS_QUEUED = "queued"
EXPORT_STATUS_RUNNING = "running"
EXPORT_STATUS_SUCCEEDED = "succeeded"
EXPORT_STATUS_FAILED = "failed"
EXPORT_STATUS_RETRIED = "retried"

VALID_EXPORT_STATUSES = {
    EXPORT_STATUS_QUEUED,
    EXPORT_STATUS_RUNNING,
    EXPORT_STATUS_SUCCEEDED,
    EXPORT_STATUS_FAILED,
    EXPORT_STATUS_RETRIED,
}
