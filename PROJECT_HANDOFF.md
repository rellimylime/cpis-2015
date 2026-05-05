# CPIS-2015 Project Handoff

## What this project is

Building a 2015 center-pivot irrigation system (CPIS) inventory for arid sub-Saharan Africa from Landsat satellite imagery. This is Emily's WaVeS Lab work at UCSB Bren School.

The published Chen et al. inventories exist for 2000 and 2021. We need a comparable 2015 inventory to study irrigation expansion over time.

## The old repo and why we left it

The original repo (`global_cpis_codes`, sibling directory) had accumulated serious structural and methodological problems:

**Sensor/domain mismatch (the critical scientific bug):** The 2021 reproduction model was trained on Sentinel-2 10m imagery (`imgs_cache_raw/`, files named `africa_s2_2021_tile_*`) but the 2015 branch uses Landsat 30m composites. Pixel value scales differ by orders of magnitude (2021 S2 stats ~1149/851/577/2151 vs 2015 Landsat stats ~0.054/0.086/0.115/0.216). Transfer learning across this domain gap is why 2015 AP was stuck around 0.24.

**Resolution vs task mismatch:** Cascade Mask R-CNN + PointRend was designed for precise instance masks. At 30m Landsat, a 200m-radius pivot is ~13px across, a 100m pivot is ~7px. The mask head is predicting circles on objects barely larger than its receptive field. Studies that successfully detect CPIs from Landsat use semantic segmentation + post-processing, not instance segmentation.

**Structural mess:** 720MB of .rar weights in git history, dead method branches (random forest classifier, centerpoint bootstrap), 6 overlapping markdown docs, code split across `tools/`, `src/cpis/`, `mm_scripts/` with unclear boundaries, misleading naming (`sentinel_scripts` used for all imagery).

**Decision:** New repo (`cpis-2015`). Pulled useful code (GEE export, anchor prep, gold eval, dataset builder, MMDet overrides). Left behind dead branches and binary blobs.

## Current repo structure

```
cpis-2015/
├── configs/
│   ├── defaults.yaml
│   └── regions.yaml
├── data/
│   ├── anchors/                  # Raw Chen et al. shapefiles (2000, 2021)
│   ├── anchors_prepared/         # Processed: stable_pivots, change_zones, etc.
│   │   └── anchor_truth/
│   │       ├── anchors/          # anchor_2000_normalized.gpkg, anchor_2021_normalized.gpkg
│   │       ├── overlays/         # stable_pivots.gpkg, change_zones.gpkg, etc.
│   │       └── summary.json
│   ├── gold/                     # Gold holdout for evaluation
│   │   ├── 2015_gold_val_v2_holdout.gpkg
│   │   ├── 2015_gold_val_v1.gpkg
│   │   └── 2015_gold_val_tiles_v1.txt
│   ├── imagery/
│   │   ├── 2015_ssa/             # Landsat 2015 tiles, 4-band (107 tiles, paper_rgbnir_v1)
│   │   ├── 2021_ssa/             # Landsat 2021 tiles, 4-band (202 tiles, paper_rgbnir_v1)
│   │   ├── 2015_ssa_stats_v1/    # Landsat 2015 tiles, 11-band stats_v1 (download in progress)
│   │   └── 2021_ssa_stats_v1/    # Landsat 2021 tiles, 11-band stats_v1 (download in progress)
│   ├── regions/                  # SSA_Arid_by_Country.shp + sidecars (tracked, ~3 MB)
│   └── splits/
│       ├── positive_tiles_2021.txt    # 102 tiles intersecting 2021 anchors
│       ├── negative_tiles_2021.txt    # 100 random background tiles
│       ├── positive_tiles_2015.txt    # Tiles intersecting stable_pivots
│       ├── 2021_val_tiles_landsat.txt # 10 held-out positive tiles for 2021 val
│       ├── 2015_val_tiles.txt
│       ├── 2015_ssa_val_v1/
│       ├── 2015_ssa_val_v6/
│       └── 2021_repro_val_v1/
├── src/cpis/
│   ├── gee/export_year.py        # GEE Landsat export
│   ├── data/
│   │   ├── prepare_anchors.py    # Normalize + intersect anchor shapefiles
│   │   ├── build_dataset.py      # COCO-format dataset from tiles + labels
│   │   ├── merge_labels.py       # Merge manual review labels
│   │   └── rasterize_labels.py   # Binary masks for semseg (NEW)
│   ├── eval/
│   │   ├── prepare_gold.py
│   │   ├── run_gold.py
│   │   ├── cocoeval.py
│   │   └── eval_file.py
│   ├── post/merge_tiles.py
│   ├── instseg/                  # Instance segmentation branch
│   │   ├── config.py             # Cascade Mask R-CNN config
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── mm_overrides/         # MMDet customizations (CBAM, PointRend)
│   └── semseg/                   # Semantic segmentation branch
│       ├── model.py              # U-Net, band-count agnostic, binary output
│       ├── train.py              # BCE + Dice loss, balanced sampling, cosine LR, --in-channels
│       ├── infer.py              # Sliding window; band count auto-derived from band_stats.json
│       ├── postprocess.py        # Connected components → circle fitting → polygons
│       ├── circle_detect.py      # Hough/shape postprocessing experiment (retired as primary)
│       ├── threshold_sweep.py    # F1 vs threshold over saved prob maps
│       └── mine_hard_negatives.py  # High-confidence FP mining, anchor-buffer-excluded
├── scripts/
│   └── setup_data.sh
├── runs/                         # Runtime artifacts (gitignored)
│   ├── instseg/
│   │   └── 2021_dataset/         # Currently being built (~3hrs)
│   └── semseg/
├── outputs/
└── tests/
```

## Model branches and what survived

### Instance segmentation (retired)

Cascade Mask R-CNN + PointRend + CBAM was tested for comparability with Chen et al., but it does not work well enough at 30 m Landsat resolution.

**Confirmed result:** 2021 Landsat instseg peaked at `segm_mAP = 0.197` (epoch 19). The old 10 m Sentinel-2 branch reached `0.608`. This is not a tuning issue. A mask head is the wrong tool when many pivots are only ~7-13 pixels across.

**Decision:** Instseg is retired. Do not spend more time here unless someone explicitly wants a methodological appendix about failed approaches.

### Semantic segmentation (active branch)

U-Net binary mask -> probability map -> threshold / postprocess -> polygons. This is the only branch that has learned usable signal at 30 m.

**Pipeline:**
1. Rasterize anchor labels to binary masks: `python -m cpis.data.rasterize_labels`
2. Train U-Net: `python -m cpis.semseg.train`
3. Infer probability maps: `python -m cpis.semseg.infer`
4. Convert to polygons with thresholding / shape filtering: `python -m cpis.semseg.postprocess`
5. Evaluate against gold holdout: `python -m cpis.eval.run_gold`

**Key design choices currently in use:**
- Input channels: band-count agnostic (auto-derived from `band_stats.json` length, or `--in-channels` flag)
  - Legacy `paper_rgbnir_v1`: 4 bands (B/G/R/NIR medians) — used by all completed runs to date
  - New `stats_v1`: 11 bands (B/G/R/NIR/SWIR1/SWIR2 medians + NDVI p10/p50/p90/amp + NDWI p50) — used for v3
- 32 base filters, 4 encoder levels
- BCE + Dice loss (1:1)
- Balanced crop sampling so training actually sees pivots; supports `--negative-centers` GPKG for hard-neg sampling
- Per-band normalization saved in `band_stats.json`
- Sliding-window inference saving float32 probability maps for offline threshold sweeps

### Shape filtering / Hough-style postprocessing (secondary only)

Circle fitting, connected-component filtering, and Hough-like circle scoring can help precision a bit, but they do not fix the real problem when the model misses pivots or fires on the wrong agricultural patterns.

**Confirmed result:** tuned circle detection improved 2021-val modestly but failed on 2015 gold. Best gold F1 from that path was about `0.065`, worse than plain thresholded semseg.

**Decision:** keep shape filtering as a bounded postprocessing experiment, not the main detection method.

## Current state (as of 2026-05-05)

### What is done
- New repo is the active workspace; region shapefile now lives in-repo at `data/regions/` (no longer depends on old repo)
- 4-band imagery (paper_rgbnir_v1): 2021 (202 tiles) and 2015 (107 tiles) — original baseline data
- 11-band imagery (stats_v1) re-export: **all GEE tasks succeeded** for both years
  - 2021: 205/205 succeeded on GEE (20 smoke + 185 full)
  - 2015: 107/107 succeeded on GEE
- **11-band rclone downloads completed** (jobs 226386/226387 finished 2026-05-05):
  - 2021: 205 tiles → `data/imagery/2021_ssa_stats_v1/`
  - 2015: 107 tiles → `data/imagery/2015_ssa_stats_v1/`
- **Anchor mask rasterization completed** (job 226388 finished 2026-05-05):
  - 205 binary masks → `runs/semseg/stats_v1_masks/` (aligned to 11-band imagery)
- **v3 training completed** (job 226389 finished 2026-05-05):
  - 11 channels (stats_v1) + 13,801 hard negatives
  - 50 epochs, best val loss **0.2704** (improvement over 2021_v1 baseline 0.3759)
  - Checkpoint: `runs/semseg/2021_v3_hardneg_stats_v1/best.pth`
  - Band stats: 11 channels validated ✓
- Gold holdout labels in: `data/gold/2015_gold_val_v2_holdout.gpkg`
- Anchor truth prepared: `anchor_2000_normalized.gpkg`, `anchor_2021_normalized.gpkg`, `stable_pivots.gpkg` (5,612), `change_zones.gpkg`
- 2015 semseg v2 (trained on `stable_pivots`) — failed to generalize to South Africa
- **2021 semseg v1** (trained on 2021 anchors) — current baseline; gold F1 = 0.1448 @ thresh 0.8
- 2021 semseg v2 (augmentation/sampling) — worse than v1
- Circle-detector / shape-filter sweeps — worse than plain semseg threshold baseline
- **Hard-negative mining completed** (job 216726 finished 2026-04-29):
  - `outputs/semseg/hard_negatives_2021_v1.gpkg`: 13,801 negatives
  - Median area 46 px, median distance to nearest anchor 369 km (clean — far from pivots)
- **Code refactored to be band-count agnostic:** `model.py`, `train.py`, `infer.py` all auto-detect `n_channels` from `band_stats.json` or first tile; `--in-channels` flag added for explicit override
- **`gee/export_year.py` extended:** `--tile-id-list` filter avoids wasting GEE quota on unused tiles (synced both old and new repo); SLURM env vars `TILE_ID_LIST`, `REBUILD_TILES`, `SKIP_LOCAL_DIR`
- **Smoke train (job 215758) on 20-tile stats_v1 subset passed all 3 gates:** band_stats.json has 11 entries, loss decreased monotonically, infer produced non-degenerate prob map
- Repository hygiene: 10 retired SLURM scripts moved to `scripts/archive/`; `.gitignore` cleaned up (`outputs/`, `data/anchors_prepared/` excluded; `data/regions/` intentionally tracked)

### In progress / queued (2026-05-05)
- `226585` infer v3 on 2015 stats_v1 tiles → `outputs/semseg/probs_2021_v3_2015/`
- `226586` threshold sweep v3 against 2015 gold → `outputs/semseg/sweep_2021_v3_2015/` (depends on 226585)

### Known blocker
None. Rclone conda-path issue was resolved by using conda activate within the SLURM script. Both download jobs (226386/226387) completed successfully.

### Best results so far

#### 2021 semseg v3 trained with 11-band stats_v1 + hard negatives

This is the active model under evaluation. Trained on 11 bands with 13,801 mined hard negatives.

Training results:
- Best val loss: `0.2704` (lower than 2021_v1 baseline 0.3759 ✓)
- Training trajectory: smooth convergence, loss decreased monotonically from epoch 1 to epoch ~21, then stable
- Epochs: 50
- Checkpoint: `runs/semseg/2021_v3_hardneg_stats_v1/best.pth`
- Band stats: 11 entries validated ✓

Gold eval: **pending** (jobs 226585/226586, ETA ~2 hours)

#### 2015 semseg v2 trained on stable pivots

This model learned some pivot signal on training tiles but had a catastrophic geographic coverage gap on the gold holdout.

Gold sweep:

| Threshold | Predictions | TP | FP | FN | Precision | Recall | F1 |
|-----------|-------------|----|----|----|-----------|--------|----|
| 0.5 | 49,306 | 24 | 6,182 | 1,258 | 0.004 | 0.019 | 0.006 |
| 0.7 | 33,481 | 67 | 5,267 | 1,215 | 0.013 | 0.052 | 0.020 |
| 0.9 | 13,349 | 158 | 2,547 | 1,124 | 0.058 | 0.123 | 0.079 |

**Conclusion:** threshold tuning is exhausted. The ceiling is the model.

#### 2021 semseg v1 trained on 2021 anchors

This is the current best branch because the 2021 anchors add much broader geographic diversity.

Best gold result:
- best F1 `0.1448` at threshold `0.8`
- max TP `172` at threshold `0.5`

Checkpoint:
- `runs/semseg/2021_v1/best.pth`

#### 2021 semseg v2 trained with extra augmentation / sampling

This was the first rescue attempt after `2021_v1`.

Results:
- training job `212427`
- best val loss at epoch 38: `0.6505`
- final epoch val loss: `1.2236`
- worse than `2021_v1` best val loss `0.3759`

2015 gold sweep best result:
- threshold `0.30`
- `TP=73`, `FP=382`, `FN=1209`
- precision `0.1604`, recall `0.0569`, F1 `0.0841`

**Conclusion:** `2021_v2` did not help. Keep `2021_v1` as the baseline to beat.

### What we think is actually wrong

The problem is no longer "can the network detect circular irrigation at all?" It can. The diagnosis has two layers:

1. **Feature limitation (newly identified):** The on-disk 2021/2015 tiles were exported with `paper_rgbnir_v1` (4 bands: annual median B/G/R/NIR). The literature converges on the point that what discriminates irrigated CPIs from rainfed agriculture at moderate resolution is *temporal/spectral* — NDVI amplitude (pivots stay green when rainfed crops brown out), NDVI p10 (dry-season floor), NDWI/SWIR. An annual median collapses all of that to a single per-pixel value, leaving the model to lean entirely on shape. The `stats_v1` contract in `gee/export_year.py` already produces all 11 bands; we just hadn't been using them.
2. **Discrimination / hard negatives:** The model confuses non-pivot agricultural patterns with pivots; threshold/postprocess tweaks help precision a bit but do not move recall enough. Better negatives — especially agricultural false positives the model currently loves — are needed.

These reinforce each other (better features make hard-negative mining cleaner, and explicit hard negatives sharpen the boundary in feature space). v3 attacks both at once.

## Next plan of action

### v3: 11-band stats_v1 + hard negatives (final evaluation)

The v3 model trained successfully with improved validation loss (0.2704 vs 2021_v1 baseline 0.3759):

**Training complete:**
- Input: 11 bands (B/G/R/NIR/SWIR1/SWIR2 medians + NDVI p10/p50/p90/amp + NDWI p50) ✓
- Sampling: ~50% pivot patches / 20% mined hard-neg / 30% random background ✓
- Loss/optimizer: BCE+Dice 1:1, AdamW 1e-3, cosine, 50 epochs, pos_weight=10 ✓
- Checkpoint: `runs/semseg/2021_v3_hardneg_stats_v1/best.pth` ✓

**Evaluation in progress (2026-05-05):**
1. `226585` — infer v3 on 2015 stats_v1 imagery → save prob maps to `outputs/semseg/probs_2021_v3_2015/`
2. `226586` — threshold sweep v3 probs against `data/gold/2015_gold_val_v2_holdout.gpkg` → results in `outputs/semseg/sweep_2021_v3_2015/`

### Decision gates (expected when sweep completes)

Compare v3 F1 @ best threshold vs 2021_v1 baseline (0.1448 @ thresh 0.8):

- **F1 > 0.20:** clear win — proceed to inference on full 2015 SSA, consider minor TTA refinements
- **0.15 < F1 ≤ 0.20:** incremental win — model improvement validated; full SSA inference worthwhile
- **F1 ≤ 0.15:** features + hard-negs insufficient — escalate to seasonal multi-date stacking (export wet+dry composites separately) as next experiment

### If v3 wins

1. Infer v3 on all 2015 SSA (full 107 tiles)
2. Convert prob maps → polygons with optimized threshold
3. Produce 2015 CPIS inventory for ground-truth validation and time-series comparison with 2000/2021

### Optional follow-ups (if needed)

- **Ablation:** 11-band-features-only run (no hard-neg sampling) to attribute gain between features vs. negatives
- **Test-time augmentation (TTA):** 4-rotation mean to squeeze extra F1 points
- **Multi-date stacking:** if single-date 11-band insufficient, export wet-season and dry-season composites separately for temporal signal

## What not to do

- Do not go back to raw Hough Circle Transform as the main detector
- Do not spend more time on instseg experiments
- Do not keep sweeping thresholds on the old 2015 stable-pivot model
- Do not contaminate the gold holdout by training on it

## Research takeaways

The outside literature and adjacent project experience point in the same direction:

- semantic segmentation is a normal choice for center-pivot detection at moderate resolution
- cross-region generalization is a known pain point
- geometric postprocessing can improve precision
- geometric postprocessing does not rescue a detector that is firing on the wrong stuff or missing most true targets

So the current path is basically right. It just needs a better negative set.

## Data flow

```
GEE → Drive → rclone → data/imagery/{year}_ssa/
                              ↓
              build_dataset.py (instseg) or rasterize_labels.py (semseg)
                              ↓
                     runs/{branch}/dataset/
                              ↓
                        train.py
                              ↓
                     runs/{branch}/best.pth
                              ↓
                        infer.py
                              ↓
                     outputs/{branch}/
                              ↓
                     eval/run_gold.py
                              ↓
                        comparison
```

## Key numbers from old repo (baseline to beat)

Old 2021 S2 model (the one with the sensor mismatch):
- segm_mAP = 0.6082, segm_mAP_50 = 0.7398 (best epoch 19)

Old 2015 transfer results (from mismatched S2→Landsat chain):
- V6 gold holdout: AP@0.50:0.95 = 0.2505, AP@0.50 = 0.4870
- These are the numbers we need to beat with the clean Landsat chain

Anchor truth stats:
- stable_pivots = 5,612
- change_zones = 2,954
- stable_background_cells = 5,465

## Important caveats

**"More pivots in 2021" may be partly methodological.** The 2021 anchor has 29,154 pivots vs 8,584 in 2000, but only 5,612 stable matches. Many 2021-only detections are small pivots (equiv_radius < 120m) that may reflect higher S2 resolution rather than real expansion. The stable_pivots layer accounts for this.

**The old val splits don't apply.** They used S2 tile names (`africa_s2_2021_tile_0025`). New Landsat exports use a different grid (`cpis_landsat_2021_tile_000025`). We created fresh val splits.

**Tile count mismatch is NOT cloud cover.** 2015 exported 398/509 tiles because the export job stopped partway through (sequential tile IDs 398-508 missing). Re-export was run with `--resume` and remaining 111 tiles completed.

## Server environment

- HPC cluster, login node: `pod-login1`
- Conda env: `cpi_fix` (Python 3.10), python at `/home/ermiller/.conda/envs/cpi_fix/bin/python`
- `rclone` binary at `/home/ermiller/.conda/envs/cpi_fix/bin/rclone` (activating `cpi_fix` within SLURM scripts works fine)
- Repos: `~/global_cpis_codes/` (old), `~/cpis-2015/` (new)
- Google Drive access via `rclone` (configured as `gdrive`)
- GEE project: `africa-irrigation-mine`
- Drive folders: `CPIS_2021_LANDSAT`, `CPIS_2015_LANDSAT` (4-band); `CPIS_2021_LANDSAT_STATS_V1`, `CPIS_2015_LANDSAT_STATS_V1` (11-band stats_v1)
- GEE batch concurrency cap: 2 tasks per project (no quota increase available via UI). Use `--tile-id-list` to avoid wasting submissions on tiles you don't need.

## Running the old repo's tools from the new repo

Most active semseg work is now in `cpis-2015`. The old repo is still only relevant for a few legacy utilities and reference code. Current known live dependency: `~/global_cpis_codes/tools/new_method/build_paper_dataset.py`. There was also a small bug fix in `tools/Image_preprocessing/show_result.py` (duplicate lines removed at 182-183).

If you need one of those old tools, run it from `~/global_cpis_codes/` and point paths at `~/cpis-2015/data/` and `~/cpis-2015/runs/`. Example:

```bash
cd ~/global_cpis_codes
python tools/new_method/build_paper_dataset.py \
  --imagery-dir ~/cpis-2015/data/imagery/2021_ssa/ \
  --labels ~/cpis-2015/data/anchors_prepared/anchor_truth/anchors/anchor_2021_normalized.gpkg \
  --out-root ~/cpis-2015/runs/instseg/2021_dataset \
  --val-sources-file ~/cpis-2015/data/splits/2021_val_tiles_landsat.txt \
  --keep-empty
```

## Migration plan: making cpis-2015 self-sufficient

The goal is to stop relying on `global_cpis_codes` for anything, but only migrate scripts that have proven useful. Don't migrate prematurely — clutter is how the old repo got bad.

### What's currently borrowed from global_cpis_codes

Four scripts are in active use. Each depends on `cpis.common` (3 tiny utility files) and `mm_scripts` (the MMDet customizations for CBAM + PointRend). The new repo already has `src/cpis/instseg/mm_overrides/` mirroring `mm_scripts`, but import paths still point at the old location.

| Script | Status | Migrate when |
|--------|--------|--------------|
| `tools/new_method/build_paper_dataset.py` | Done its job (2021 dataset built) | Before next dataset build |
| `tools/new_method/train_paper_model.py` | **Running now** | After 2021 training result confirms instseg is viable |
| `tools/new_method/run_paper_inference.py` | Not yet needed | After inference runs successfully |
| `tools/new_method/run_gold_eval.py` | Not yet needed | After eval runs successfully |

### How to migrate (when the time comes)

The new repo already has stub files in the right places. Migration = fold the working logic in, don't create new files.

**Wave 1 — after 2021 training result:**
- `build_paper_dataset.py` → `src/cpis/data/build_dataset.py`
- `train_paper_model.py` → `src/cpis/instseg/train.py`
- Fix `mm_overrides/` import paths: replace `mm_scripts.*` → `cpis.instseg.mm_overrides.*` throughout
- Add `logging_utils.py` and `time_utils.py` to `src/cpis/` (alongside existing `file_utils.py`)
- Update SLURM script to `cd ~/cpis-2015` and run `python -m cpis.instseg.train ...`

**Wave 2 — after inference + gold eval run successfully:**
- `run_paper_inference.py` → `src/cpis/instseg/infer.py`
- `run_gold_eval.py` → `src/cpis/eval/run_gold.py`
- Note: inference and eval have deeper deps on `tools/detect_scripts/` and `tools/evaluation/` in the old repo — audit carefully before pulling those in

Once Wave 2 is done, `global_cpis_codes` is fully retired from the active pipeline.

## Emily's preferences

Direct, concise communication. Dry humor appreciated. Challenge bad approaches. Don't over-engineer. Only make changes that are directly requested or clearly necessary. Plain language over formal/AI-sounding prose. Prioritize keeping track of the current status of the project and work flow and checking logic/intuition as we go as not to get off track (check/compare against existing successful research when possible/applicable). Justify/explain complex choices, outline lines of logic, or important decisions.
