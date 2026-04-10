# cpis-2015

2015 center-pivot irrigation inventory for arid sub-Saharan Africa.

Two parallel detection branches evaluated on the same gold holdout:

- **Instance segmentation** (`src/cpis/instseg/`): Cascade Mask R-CNN + PointRend + CBAM, following Chen et al. Transfer from a 2021 Landsat model to 2015 Landsat.
- **Semantic segmentation** (`src/cpis/semseg/`): U-Net binary mask → connected components → circle fitting. Trained directly on 2015 Landsat.

Both branches produce polygon inventories with centers and equivalent radii.

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate cpis2015

# Download anchor shapefiles and GEE imagery
bash scripts/setup_data.sh
```

## Quick start

```bash
# Prepare anchor truth (shared)
python -m cpis.data.prepare_anchors --config configs/defaults.yaml

# Instance seg branch
python -m cpis.instseg.train --config configs/defaults.yaml --year 2021
python -m cpis.instseg.train --config configs/defaults.yaml --year 2015 --checkpoint runs/instseg/2021/best.pth
python -m cpis.instseg.infer --config configs/defaults.yaml --checkpoint runs/instseg/2015/best.pth

# Semantic seg branch
python -m cpis.semseg.train --config configs/defaults.yaml
python -m cpis.semseg.infer --config configs/defaults.yaml --checkpoint runs/semseg/best.pth

# Evaluate both on gold holdout
python -m cpis.eval.run_gold --predictions outputs/instseg/ --gold data/gold/
python -m cpis.eval.run_gold --predictions outputs/semseg/ --gold data/gold/
```

## Data

All imagery is Landsat Collection 2, 4-band (blue, green, red, NIR), 30m resolution. Anchor inventories are the published Chen et al. 2000 and 2021 CPIS shapefiles. Gold holdout labels are manually reviewed 2015 tiles withheld from training.

Imagery and anchors are not committed to this repo. Run `scripts/setup_data.sh` to download them.

## Directory layout

```
configs/          Pipeline configuration
src/cpis/
  gee/            GEE export scripts
  data/           Anchor prep, label building, dataset creation
  eval/           Gold holdout evaluation (shared across branches)
  post/           Polygonize, circle fitting, tile merging, QA
  instseg/        Instance segmentation branch (MMDetection)
  semseg/         Semantic segmentation branch (PyTorch)
scripts/          Shell scripts for setup and batch jobs
data/             Downloaded imagery and labels (gitignored)
runs/             Training artifacts (gitignored)
outputs/          Final polygon inventories
```
