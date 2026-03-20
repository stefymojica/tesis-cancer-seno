# Coding Conventions

**Analysis Date:** 2026-03-20

## Project Context

This is a machine learning thesis project for breast cancer prediction using a hierarchical multimodal approach (clinical data + mammography images). The codebase follows Python data science conventions with PyTorch for deep learning components.

## Naming Patterns

**Files:**
- **Scripts:** `snake_case.py` - e.g., `train_clinical.py`, `predict_clinical.py`, `preprocess.py`
- **Modules:** Descriptive names indicating purpose - e.g., `src/models/`, `src/data/`, `src/vision/`, `src/fusion/`
- **Notebooks:** `descriptive_name_v1.ipynb` for experiments, never commit without version suffix
- **Configuration:** `config.yaml` or `settings.py` for project-wide parameters

**Functions:**
- **Public API:** `snake_case` - e.g., `get_clinical_risk_vector()`, `extract_visual_embeddings()`
- **Private helpers:** `_leading_underscore` - e.g., `_normalize_image()`, `_validate_patient_id()`
- **ML-specific:** Include action in name - e.g., `train_xgboost_model()`, `predict_proba()`, `compute_shap_values()`

**Variables:**
- **Constants:** `UPPER_SNAKE_CASE` - e.g., `RANDOM_SEED = 42`, `IMAGE_SIZE = 512`, `BATCH_SIZE = 32`
- **Model outputs:** Descriptive - e.g., `risk_vector`, `clinical_features`, `visual_embeddings`, `fused_prediction`
- **Data splits:** Explicit naming - `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`
- **PyTorch tensors:** Include type hint - e.g., `image_tensor: torch.Tensor`, `risk_tensor: torch.Tensor`

**Classes:**
- **Models:** `PascalCase` with descriptive names - e.g., `ClinicalRiskEncoder`, `VisionFeatureExtractor`, `HierarchicalFusionModel`
- **Datasets:** `PascalCaseDataset` - e.g., `MammographyDataset`, `ClinicalDataset`, `CBISDDSDataset`
- **Transforms:** `PascalCase` - e.g., `DicomToTensor`, `MammogramNormalizer`

**Types:**
- **Type hints required** for all function parameters and returns
- **Custom types:** Define in `src/types.py` - e.g., `RiskVector = np.ndarray`, `PatientId = str`
- **Generic types:** Use `typing` module - e.g., `List[PatientId]`, `Dict[str, torch.Tensor]`

## Code Style

**Formatting:**
- **Tool:** Black (line length 100)
- **Import sorting:** isort with `black` profile
- **Configuration:** `.pre-commit-config.yaml` with black, isort, flake8

**Linting:**
- **Tool:** flake8 with extensions:
  - `flake8-docstrings` (Google style)
  - `flake8-bugbear` (common pitfalls)
  - `pep8-naming` (naming conventions)
- **Max line length:** 100
- **Ignore:** E203, W503 (Black compatibility)

**Docstrings:**
- **Style:** Google docstring format
- **Required for:** All public functions, classes, modules
- **Content:** Args, Returns, Raises, Examples for complex functions
- **ML-specific:** Include expected shapes for arrays/tensors in docstrings

## Import Organization

**Order (with blank lines between groups):**
1. **Standard library:** `import os`, `from pathlib import Path`
2. **Third-party packages:** 
   - Scientific: `import numpy as np`, `import pandas as pd`
   - ML: `import torch`, `import xgboost as xgb`
   - XAI: `import shap`, `from captum.attr import LayerGradCam`
   - Viz: `import matplotlib.pyplot as plt`
3. **First-party (project):**
   - Absolute imports: `from src.data.preprocess import ClinicalPreprocessor`
   - Never use relative imports: ~~`from ..models import ...`~~

**Path Aliases:**
- Standard scientific: `np`, `pd`, `plt`, `torch`, `xgb`, `sklearn`
- XAI: `shap`

## Error Handling

**Patterns:**
- **Validation errors:** Raise `ValueError` with descriptive messages
- **Data errors:** Raise `DataValidationError` (custom exception) for preprocessing failures
- **Model errors:** Raise `ModelInferenceError` for prediction failures
- **File operations:** Use `pathlib` with explicit existence checks

**ML-Specific Error Patterns:**
```python
def predict_clinical_risk(patient_data: pd.DataFrame) -> np.ndarray:
    """Generate clinical risk vector from patient data.
    
    Raises:
        ValueError: If patient_data is missing required columns.
        ModelInferenceError: If model prediction fails.
    """
    required_cols = {'age', 'density', 'family_history'}
    missing = required_cols - set(patient_data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    try:
        return model.predict_proba(patient_data)[:, 1]
    except Exception as e:
        raise ModelInferenceError(f"Prediction failed: {e}") from e
```

## Reproducibility Conventions

**Random Seeds:**
- **Global seed:** `RANDOM_SEED = 42` in `src/config.py`
- **Set everywhere:**
  ```python
  np.random.seed(RANDOM_SEED)
  torch.manual_seed(RANDOM_SEED)
  torch.cuda.manual_seed_all(RANDOM_SEED)
  xgb.set_config(verbosity=0)
  ```
- **Data splits:** Use `sklearn.model_selection.StratifiedKFold` with fixed `random_state`

**Deterministic Operations:**
- PyTorch: Set `torch.backends.cudnn.deterministic = True`
- PyTorch: Set `torch.backends.cudnn.benchmark = False`
- Document any non-deterministic operations in comments

## Data Handling Conventions

**Patient ID Protection:**
- **Never drop:** Always keep `patient_id` column through pipeline
- **Split by patient:** Use `GroupShuffleSplit` to prevent data leakage
- **Validation:** Assert no patient appears in multiple splits

**Data Leakage Prevention:**
- **Preprocessing:** Fit on train only, transform on all
- **Cross-validation:** Use sklearn pipelines with ColumnTransformer
- **Images:** Same patient CC and MLO views must stay in same split

**Missing Data:**
- **Strategy:** Document in `src/data/missing_values.py`
- **Clinical:** Impute with median for numeric, mode for categorical
- **Images:** Flag and exclude or generate synthetic (document decision)

## ML Code Structure

**Model Training Scripts (`src/models/train_*.py`):**
```python
# 1. Imports
import numpy as np
import xgboost as xgb
from src.config import RANDOM_SEED
from src.data.preprocess import ClinicalPreprocessor

# 2. Configuration
RANDOM_SEED = 42
N_ESTIMATORS = 100

# 3. Main training function
def train_clinical_model(train_path: str, val_path: str) -> xgb.XGBClassifier:
    """Train and return XGBoost model on clinical data.
    
    Args:
        train_path: Path to training CSV.
        val_path: Path to validation CSV.
        
    Returns:
        Trained XGBClassifier with best hyperparameters.
    """
    pass

# 4. CLI entry point
if __name__ == "__main__":
    # argparse or typer for CLI
    pass
```

**PyTorch Modules:**
- Inherit from `nn.Module`
- Define layers in `__init__`, logic in `forward`
- Include input/output shape comments

## Logging

**Framework:** `logging` module (standard library)

**Levels:**
- **DEBUG:** Detailed training metrics per epoch, batch losses
- **INFO:** High-level milestones - "Training complete", "Model saved"
- **WARNING:** Data quality issues - "Missing values in column X"
- **ERROR:** Failures that don't stop execution
- **CRITICAL:** Fatal errors requiring intervention

**Configuration:**
- Log to both console and file (`logs/training_YYYYMMDD.log`)
- Use structured format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Never use `print()` for logging (use logger instead)

## Comments and Documentation

**When to Comment:**
- **Complex algorithms:** Explain mathematical operations
- **Data decisions:** Why certain preprocessing choices were made
- **Workarounds:** Explain non-obvious code with reference to issues
- **ML-specific:** Document expected tensor shapes, normalization ranges

**README Files:**
- Each module directory (`src/models/`, `src/data/`) has README.md
- Document module purpose, key classes/functions
- Include usage examples

## Function Design

**Size:**
- **Max 50 lines** for main logic (excluding docstring)
- **Single responsibility:** One function = one ML operation
- **Refactor:** Extract helper functions for repeated operations

**Parameters:**
- **Data first:** DataFrame/array parameters first, then config
- **Config objects:** Use dataclasses for model hyperparameters
- **Defaults:** Sensible defaults for all optional parameters

**Return Values:**
- **Always typed:** Use `-> np.ndarray`, `-> torch.Tensor`, `-> Dict[str, float]`
- **Multiple values:** Return dataclass or namedtuple, not bare tuple
- **Side effects:** Document any file I/O or model state changes

## Module Design

**Exports:**
- **Public API:** Define `__all__` in each module
- **Barrel files:** `src/models/__init__.py` exports main classes
- **Internal only:** Prefix with underscore, don't export

**Directory Structure:**
```
src/
├── __init__.py           # Package version, main exports
├── config.py             # Global configuration, constants
├── types.py              # Type aliases, custom types
├── data/                 # Data loading and preprocessing
│   ├── __init__.py
│   ├── preprocess.py     # ClinicalPreprocessor class
│   ├── datasets.py       # PyTorch Dataset classes
│   └── transforms.py     # Image transformations
├── models/               # Model training and inference
│   ├── __init__.py
│   ├── train_clinical.py # XGBoost training
│   ├── predict_clinical.py
│   ├── train_vision.py   # CNN/ViT training
│   └── predict_vision.py
├── vision/               # Computer vision components
│   ├── __init__.py
│   ├── backbones.py      # ResNet, EfficientNet, ViT
│   └── features.py       # Feature extraction
├── fusion/               # Multimodal fusion
│   ├── __init__.py
│   └── hierarchical.py   # Late fusion implementation
├── xai/                  # Explainability
│   ├── __init__.py
│   ├── shap_clinical.py  # SHAP for clinical model
│   └── gradcam.py        # Grad-CAM for vision model
├── evaluation/           # Metrics and fairness
│   ├── __init__.py
│   ├── metrics.py        # AUC, sensitivity, specificity
│   └── fairness.py       # Demographic parity analysis
└── ui/                   # Interface
    ├── __init__.py
    └── app.py            # Streamlit/Gradio app
```

## Experiment Tracking

**MLflow Integration:**
- Log parameters: model architecture, hyperparameters, data version
- Log metrics: AUC, sensitivity, specificity per epoch
- Log artifacts: model files, SHAP plots, Grad-CAM images
- Set experiment name: `tesis-cancer-mama-{phase}`

**Version Control for Data:**
- Use DVC (Data Version Control) for datasets
- Track with `.dvc` files, not raw data
- Document dataset versions in `data/README.md`

---

*Convention analysis: 2026-03-20*
