# Technology Stack

**Analysis Date:** 2026-03-20

## Languages

**Primary:**
- **Python 3.10+** - All machine learning, data processing, and model training code

## Runtime

**Environment:**
- Python 3.10+ with virtual environment (venv/conda recommended)

**Package Manager:**
- pip - Standard Python package manager
- Lockfile: Not present (requirements.txt recommended)

## Frameworks

**Core ML/Deep Learning:**
- **PyTorch** - Deep learning framework for CNN/ViT models
- **torchvision** - Pre-trained models (ResNet, EfficientNet) and image transforms
- **timm** (PyTorch Image Models) - Vision Transformer (ViT) and other SOTA architectures

**Tabular ML:**
- **XGBoost** 2.x - Gradient boosting for clinical risk prediction (baseline model)
- **scikit-learn** 1.x - Feature engineering, preprocessing, metrics, MLP baseline

**Medical Imaging:**
- **pydicom** 2.x - DICOM file reading and medical image metadata extraction
- **OpenCV (opencv-python)** 4.x - Image resizing, normalization, preprocessing
- **albumentations** - Data augmentation for medical images
- **scikit-image** - Advanced image processing utilities
- **Pillow (PIL)** - Image format handling

**Explainable AI (XAI):**
- **pytorch-grad-cam** - Grad-CAM heatmaps for mammography visualization
- **shap** - SHAP values for clinical feature importance

**Data & Processing:**
- **pandas** 2.x - Tabular data manipulation and clinical metadata handling
- **numpy** - Numerical computations
- **scipy** - Statistical tests (DeLong test for AUC comparison)

**Experiment Tracking:**
- **MLflow** or **Weights & Biases (wandb)** - Experiment tracking and model versioning

**UI/Dashboard:**
- **Gradio** or **Streamlit** - Medical interface for interactive predictions

**Data Acquisition:**
- **kagglehub** 1.x - Automated CBIS-DDSM dataset download
- **tcia_utils** - TCIA (The Cancer Imaging Archive) data access

## Key Dependencies

**Critical for Core Functionality:**
- `torch`, `torchvision` - Deep learning backbone
- `xgboost` - Clinical tabular model
- `pydicom` - Medical image format support
- `scikit-learn` - ML pipeline infrastructure

**Infrastructure:**
- `pandas` - Data manipulation
- `opencv-python` - Image preprocessing
- `shap` - Clinical model explainability
- `pytorch-grad-cam` - Visual explainability

## Configuration

**Environment:**
- Python virtual environment required
- No `.env` file detected (all public datasets, no API keys needed)

**Build:**
- No build configuration files present (pure Python project)
- Recommended: `requirements.txt` for dependency management

**Project Structure:**
```
.
├── docs/                   # Documentation (Design Docs)
├── data/
│   ├── raw/                # Original downloaded datasets
│   └── processed/          # Cleaned metadata and preprocessed images
└── src/
    ├── data/               # Data download and preprocessing scripts
    └── models/             # Training and prediction scripts
```

## Platform Requirements

**Development:**
- Python 3.10+
- 8GB+ RAM recommended (medical imaging datasets)
- GPU recommended for deep learning training (CUDA-compatible)
- 50GB+ disk space for CBIS-DDSM dataset

**Production:**
- Local deployment on medical workstation (laptop/desktop)
- Not designed for cloud/SaaS deployment (academic proof-of-concept)
- Inference should complete in < 10 seconds per patient

---

*Stack analysis: 2026-03-20*
