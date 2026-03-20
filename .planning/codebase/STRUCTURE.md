# Codebase Structure

**Analysis Date:** 2026-03-20

## Directory Layout

```
tesis-cancer-seno/
├── .planning/                    # GSD planning artifacts
│   ├── codebase/                 # Codebase analysis docs (this dir)
│   ├── phases/                   # Phase-specific plans
│   ├── research/                 # Research findings
│   ├── config.json               # GSD configuration
│   ├── PROJECT.md                # Project overview
│   ├── REQUIREMENTS.md           # Feature requirements
│   ├── ROADMAP.md                # Implementation roadmap
│   └── STATE.md                  # Current project state
├── docs/                         # Project documentation
│   ├── 00_design_doc.md          # Architecture design document
│   └── guia_tesis_meritoria.md   # Thesis guide (meritoria requirements)
├── src/                          # Source code (to be created)
│   ├── __init__.py
│   ├── data/                     # Data handling modules
│   ├── clinical/                 # Stage 1: Clinical risk model
│   ├── vision/                   # Stage 2a: Vision feature extraction
│   ├── fusion/                   # Stage 2b: Multimodal fusion
│   ├── xai/                      # Explainability modules
│   ├── evaluation/               # Metrics and evaluation
│   └── ui/                       # Web interface
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Unit and integration tests
├── configs/                      # Model and training configurations
├── checkpoints/                # Saved model weights (gitignored)
├── data/                         # Dataset storage (gitignored)
├── outputs/                      # Results, figures, logs (gitignored)
├── train.py                      # Main training entry point
├── predict.py                    # Inference script
├── app.py                        # Web application entry point
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── README.md                     # Project readme
└── .gitignore                    # Git ignore rules
```

## Directory Purposes

**`.planning/`:**
- Purpose: GSD (Get Shit Done) methodology planning artifacts
- Contains: Roadmaps, requirements, phase plans, research, and codebase documentation
- Key files: `ROADMAP.md`, `REQUIREMENTS.md`, `PROJECT.md`, `phases/01-preparacion-y-baseline-clinico/`

**`docs/`:**
- Purpose: External-facing and detailed design documentation
- Contains: Architecture diagrams, thesis guide, methodology documents
- Key files: `00_design_doc.md`, `guia_tesis_meritoria.md`

**`src/`:**
- Purpose: Main source code directory (not yet created, planned structure)
- Contains: Modular implementation of all ML pipeline components
- Subdirectories:
  - `data/`: Dataset loaders, preprocessors, splitters
  - `clinical/`: Tabular models (XGBoost, MLP), SHAP explainers
  - `vision/`: CNN/ViT backbones, image preprocessing
  - `fusion/`: Late fusion networks, attention mechanisms
  - `xai/`: Explainability (SHAP, Grad-CAM), fairness analysis
  - `evaluation/`: Metrics calculation, statistical tests
  - `ui/`: Streamlit/Gradio web interface

**`scripts/`:**
- Purpose: Standalone utility and setup scripts
- Contains: Data downloaders, preprocessing pipelines, batch inference scripts

**`notebooks/`:**
- Purpose: Jupyter notebooks for exploratory analysis and visualization
- Contains: EDA notebooks, evaluation reports, thesis figure generation

**`tests/`:**
- Purpose: Test suite
- Contains: Unit tests for modules, integration tests for pipeline

**`configs/`:**
- Purpose: YAML/JSON configuration files
- Contains: Model hyperparameters, training configs, dataset paths

**`checkpoints/`:**
- Purpose: Saved model weights and training states
- Generated: Yes (during training)
- Committed: No (gitignored, stored via git-lfs or external storage)

**`data/`:**
- Purpose: Downloaded and processed datasets
- Generated: Yes (via scripts)
- Committed: No (gitignored, large files)
- Subdirectories: `raw/`, `processed/`, `external/`

**`outputs/`:**
- Purpose: Generated results, figures, logs
- Generated: Yes (during experiments)
- Committed: No (gitignored)
- Subdirectories: `figures/`, `logs/`, `results/`, `shap/`, `gradcam/`

## Key File Locations

**Entry Points:**
- `train.py`: Orchestrates full training pipeline (clinical → vision → fusion)
- `predict.py`: Single inference execution from command line
- `app.py`: Launches web interface (`streamlit run app.py`)
- `scripts/prepare_data.py`: One-time dataset setup

**Configuration:**
- `configs/clinical.yaml`: XGBoost/MLP hyperparameters
- `configs/vision.yaml`: CNN/ViT architecture and training settings
- `configs/fusion.yaml`: Late fusion network configuration
- `.env`: Environment variables (dataset paths, API keys - not committed)

**Core Logic:**
- `src/clinical/risk_scorer.py`: Clinical risk prediction interface
- `src/vision/backbone.py`: Image feature extraction base classes
- `src/fusion/late_fusion.py`: Multimodal integration implementation
- `src/xai/explainer.py`: SHAP and Grad-CAM generation

**Testing:**
- `tests/test_clinical.py`: Unit tests for clinical module
- `tests/test_vision.py`: Unit tests for vision module
- `tests/test_fusion.py`: Unit tests for fusion module
- `tests/test_pipeline.py`: Integration test for full inference pipeline

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `risk_scorer.py`, `data_loader.py`)
- Classes: `PascalCase` (e.g., `XGBoostRiskScorer`, `ResNetBackbone`)
- Functions/variables: `snake_case` (e.g., `extract_embeddings`, `risk_score`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `RANDOM_SEED`, `IMAGE_SIZE`)
- Notebooks: `NN_descriptive_name.ipynb` (e.g., `01_eda_cbis_ddsm.ipynb`)
- Configs: `descriptive_name.yaml`

**Directories:**
- Source modules: `snake_case/` (e.g., `clinical/`, `vision/`)
- Notebooks prefixed with order number: `01_`, `02_`, etc.

## Where to Add New Code

**New Feature (e.g., new clinical model type):**
- Implementation: `src/clinical/{model_name}.py`
- Base class: `src/clinical/base.py` (if creating new abstraction)
- Tests: `tests/test_clinical.py`
- Config: `configs/clinical.yaml` (add section)

**New Component/Module (e.g., add MRI support):**
- Implementation: `src/mri/` (new directory)
- Add to pipeline: `src/pipeline.py` or `train.py`
- Tests: `tests/test_mri.py`

**Utilities:**
- Shared helpers: `src/utils.py` or `src/common/`
- Data utilities: `src/data/utils.py`
- Visualization: `src/visualization.py`

**Experiments:**
- New notebook: `notebooks/XX_experiment_name.ipynb`
- Results: `outputs/results/experiment_name/`

## Special Directories

**`.planning/`:**
- Purpose: GSD methodology artifacts and project management
- Generated: Yes (manually created)
- Committed: Yes (except potentially sensitive research notes)
- Structure follows GSD conventions with PROJECT.md, ROADMAP.md, etc.

**`.worktrees/`:**
- Purpose: Git worktrees for parallel development branches
- Generated: Yes (git worktree feature)
- Committed: No (git metadata)
- Contains: `task/demo`, `task/feature-dataset` with their own planning dirs

**`data/` (gitignored):**
- Purpose: Dataset storage
- Expected structure:
  ```
  data/
  ├── raw/
  │   ├── cbis-ddsm/
  │   └── cmmd/
  ├── processed/
  │   ├── clinical_features.csv
  │   └── images_preprocessed/
  └── external/           # Optional third-party data
  ```

**`checkpoints/` (gitignored):**
- Purpose: Model weight storage
- Structure mirrors training runs:
  ```
  checkpoints/
  ├── clinical/
  │   └── xgboost_best.pkl
  ├── vision/
  │   └── resnet50_best.pth
  └── fusion/
      └── late_fusion_best.pth
  ```

---
*Structure analysis: 2026-03-20*
