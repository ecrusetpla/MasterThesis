# MasterThesis

This repository contains the full codebase for my Master's thesis. The project investigates the relationship between socioeconomic conditions, proximity factors, criminality, and public transport disruptions at the neighbourhood (buurt/wijk) level in the Netherlands, using a panel dataset and a multi-tiered modelling approach.

---

## Repository Structure

```
MasterThesis/
├── data/
│   ├── socioeconomic/        # Raw SES input files
│   ├── criminality/          # Raw criminality input files
│   ├── proximity/            # Raw proximity input files
│   └── processed/            # Intermediate and final processed datasets
├── src/
│   ├── Data processing/      # Data ingestion and preprocessing notebooks
│   ├── EDA/                  # Exploratory data analysis notebook
│   └── results/              # Feature engineering and modelling scripts
├── research/                 # Supporting research materials
├── results/                  # Output files from modelling scripts (by tier)
├── requirements.txt
└── README.md
```

---

## Data

All raw data used in this project is included in the `data/` folder, **with one exception**:

- **`wijkenbuurten_2024.gpkg`** — used in `src/Data processing/1.4. Merge all data.ipynb` — must be downloaded manually from the CBS website:  
  [https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2024](https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2024)  
  Place the downloaded file in the appropriate data directory before running the merge script.

> **Note on large intermediate files:** Some intermediate datasets exceed GitHub's 100 MB file size limit and are therefore not included in this repository (e.g., `data/processed/final_dataset.parquet`). These files can be fully reproduced by running the data processing pipeline from the raw input files. Alternatively, the author can be contacted directly to obtain them.

---

## Pipeline Overview

The pipeline runs in the following order:

### Step 1 — Data Processing (`src/Data processing/`)

| Script | Input | Output |
|--------|-------|--------|
| `1.1. Process SES variables.ipynb` | All files in `data/socioeconomic/` | `data/processed/socioeconomic_all.csv` |
| `1.2. Process criminality.ipynb` | `data/criminality/Criminality by neighbourhood and district.csv` | `data/processed/criminality.csv` |
| `1.3. Compile and process proximity datasets.ipynb` | All files in `data/proximity/` | `data/processed/proximity.csv` |
| `1.4. Merge all data.ipynb` | Outputs from steps 1.1–1.3 + `wijkenbuurten_2024.gpkg` + [`daily_disruptions_weather.csv`](https://github.com/Brebber/Thesis/blob/main/MSc-Thesis-Brent/NS-Disruptions-2025/Data/daily_disruptions_weather.csv) | `data/processed/final_dataset.parquet` |

### Step 2 — Exploratory Data Analysis (`src/EDA/`)

| Script | Input | Output |
|--------|-------|--------|
| `EDA.ipynb` | `data/processed/final_dataset.parquet` | Visualisations and summary statistics |

### Step 3 — Feature Engineering (`src/results/`)

| Script | Input | Output |
|--------|-------|--------|
| `00_feature_engineering.ipynb` | `data/processed/final_dataset.parquet` | `modelling_panel_THRESHOLD.parquet` |

### Step 4 — Modelling (`src/results/`)

All modelling scripts take `modelling_panel_THRESHOLD.parquet` as input and write their outputs to the `results/` folder.

| Script | Output folder | Files produced |
|--------|---------------|----------------|
| `02_tier1.py` | `results/tier 1 THRESHOLD/` | 3 files |
| `03_tier2.py` | `results/tier 2 THRESHOLD/` | 5 files |
| `04_tier3.py` | `results/tier 3 THRESHOLD/` | 7 files |

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ecrusetpla/MasterThesis.git
   cd MasterThesis
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download `wijkenbuurten_2024.gpkg` from the [CBS website](https://www.cbs.nl/nl-nl/dossier/nederland-regionaal/geografische-data/wijk-en-buurtkaart-2024) and place it in the correct data directory.

---

## Dependencies

Key libraries used in this project:

- `pandas`, `numpy` — data manipulation
- `geopandas`, `shapely`, `folium` — geospatial processing and visualisation
- `scikit-learn`, `xgboost`, `imbalanced-learn` — machine learning
- `torch`, `torch_geometric` — deep learning and graph neural networks
- `statsmodels` — statistical modelling
- `shap` — model interpretability
- `matplotlib`, `seaborn`, `plotly` — visualisation
- `pyarrow` — Parquet file handling
- `networkx` — graph analysis

See `requirements.txt` for the full list with pinned versions.

---

## Contact

For questions about the project, or to request access to large intermediate datasets that could not be included in this repository, please contact the author via GitHub: [@ecrusetpla](https://github.com/ecrusetpla).