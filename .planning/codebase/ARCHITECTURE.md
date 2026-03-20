# Architecture

**Analysis Date:** 2026-03-20

## Pattern Overview

**Overall:** Hierarchical Multimodal Pipeline (2-Stage Architecture)

**Key Characteristics:**
- Sequential processing mimicking real clinical workflow (clinical assessment before image analysis)
- Late fusion mechanism combining tabular and visual modalities
- Explanability built into both stages (SHAP for clinical, Grad-CAM for vision)
- Fairness-aware validation across demographic subgroups

## Layers

**Data Preparation Layer:**
- Purpose: Data ingestion, cleaning, preprocessing, and alignment of clinical and imaging data
- Location: `src/data/`
- Contains: Dataset downloaders, DICOM/PNG processors, metadata parsers, train/val/test splitters
- Depends on: External datasets (CBIS-DDSM, CMMD), storage system
- Used by: Training pipelines for both clinical and vision modules

**Clinical Module (Stage 1 - Triaje):**
- Purpose: Risk stratification using tabular clinical features only
- Location: `src/clinical/`
- Contains: Feature encoders, XGBoost/MLP models, SHAP explainers, risk score generators
- Depends on: pandas, scikit-learn, XGBoost, SHAP library
- Used by: Fusion module, UI layer for triage display

**Vision Module (Stage 2a - Feature Extraction):**
- Purpose: Deep learning-based feature extraction from mammography images
- Location: `src/vision/`
- Contains: CNN/ViT backbones (ResNet, EfficientNet, ViT), image preprocessing pipelines, embedding extractors
- Depends on: PyTorch, torchvision, timm, albumentations
- Used by: Fusion module for visual feature vectors

**Fusion Module (Stage 2b - Diagnóstico):**
- Purpose: Multimodal integration of clinical risk scores and visual embeddings
- Location: `src/fusion/`
- Contains: Late fusion networks, attention mechanisms, final classification heads
- Depends on: PyTorch, clinical module outputs, vision module outputs
- Used by: Inference pipeline, UI layer for final predictions

**Explanability Layer (XAI):**
- Purpose: Generate human-interpretable explanations for model decisions
- Location: `src/xai/`
- Contains: SHAP plot generators for clinical features, Grad-CAM heatmap generators for images, fairness analyzers
- Depends on: shap, pytorch-grad-cam, clinical and fusion modules
- Used by: UI layer for visualization, reports generation

**UI Layer:**
- Purpose: Medical-grade interface for end-to-end inference
- Location: `src/ui/` or `app.py`
- Contains: Streamlit/Gradio web interface, form handlers, image uploaders, result visualizers
- Depends on: All modules above, gradio/streamlit
- Used by: Medical professionals (end users)

**Evaluation & Metrics Layer:**
- Purpose: Comprehensive model evaluation and statistical testing
- Location: `src/evaluation/`
- Contains: Metric calculators (AUC, sensitivity, specificity), DeLong test implementations, fairness auditors
- Depends on: scikit-learn, scipy, predictions from all modules
- Used by: Training scripts, reporting notebooks

## Data Flow

**Training Flow:**

1. Raw Data Ingestion: CBIS-DDSM and CMMD datasets downloaded via `src/data/download.py`
2. Clinical Preprocessing: Tabular features extracted and encoded in `src/data/clinical_preprocessor.py`
3. Image Preprocessing: DICOMs converted, normalized, and augmented via `src/data/image_preprocessor.py`
4. Clinical Training: XGBoost/MLP trained on tabular data only, outputs risk model checkpoint
5. Vision Training: CNN/ViT trained on images, outputs backbone checkpoint and embedding extractor
6. Fusion Training: Combined clinical + visual features trained with late fusion mechanism
7. XAI Generation: SHAP and Grad-CAM explanations computed on validation set
8. Evaluation: Metrics calculated, fairness analysis performed, results logged

**Inference Flow:**

1. User Input: Clinical data entered via web form + mammography image uploaded
2. Clinical Stage: Risk score computed using trained clinical model
3. Decision Point: If risk < 0.3 → "Bajo Riesgo"; If risk ≥ 0.3 → Proceed to vision stage
4. Vision Stage: Image processed, embeddings extracted using trained vision backbone
5. Fusion Stage: Clinical risk vector + visual embeddings combined, final cancer probability computed
6. Explanation Generation: SHAP values for clinical factors + Grad-CAM heatmap on image
7. Result Display: Final prediction + explanations presented to physician in <10 seconds

**State Management:**
- No persistent state between sessions; each inference is stateless
- Model checkpoints loaded once at startup and cached in memory
- Session data managed by Streamlit/Gradio framework

## Key Abstractions

**PatientRecord:**
- Purpose: Unified representation of a patient's clinical and imaging data
- Location: `src/data/patient_record.py`
- Pattern: Data class/dataclass holding clinical features dict + image paths + labels

**RiskScorer (Clinical Model Interface):**
- Purpose: Abstract interface for clinical risk prediction models
- Location: `src/clinical/base.py` or `src/clinical/risk_scorer.py`
- Pattern: Abstract base class with `fit()`, `predict_risk()`, `explain()` methods
- Implementations: XGBoostRiskScorer, MLPRiskScorer

**VisionBackbone (Image Feature Extractor):**
- Purpose: Abstract interface for deep learning image encoders
- Location: `src/vision/base.py` or `src/vision/backbone.py`
- Pattern: PyTorch nn.Module with `forward()`, `extract_embeddings()`, `get_attention_maps()` methods
- Implementations: ResNetBackbone, EfficientNetBackbone, ViTBackbone

**FusionNetwork (Multimodal Integration):**
- Purpose: Combines clinical and visual representations for final prediction
- Location: `src/fusion/base.py` or `src/fusion/late_fusion.py`
- Pattern: PyTorch nn.Module taking two feature vectors, outputting probability
- Implementations: ConcatenationFusion, AttentionFusion

**Explainer (XAI Interface):**
- Purpose: Generate explanations for model predictions
- Location: `src/xai/base.py`
- Pattern: Interface with `explain_clinical()`, `explain_visual()` methods
- Implementations: ShapExplainer, GradCamExplainer

**Dataset (PyTorch Dataset):**
- Purpose: Bridge between raw data and model training
- Location: `src/data/dataset.py`
- Pattern: PyTorch Dataset/DataLoader for batching clinical + image data

## Entry Points

**Training Script:**
- Location: `train.py` or `scripts/train_all.py`
- Triggers: Manual execution or SLURM job submission
- Responsibilities: Orchestrates training of all 3 model stages (clinical, vision, fusion)

**Inference Script:**
- Location: `predict.py` or `scripts/infer.py`
- Triggers: Command-line execution with patient data and image path
- Responsibilities: Loads trained models, runs full pipeline, outputs prediction + explanations

**Web Application:**
- Location: `app.py` or `src/ui/app.py`
- Triggers: `streamlit run app.py` or `python -m src.ui`
- Responsibilities: Serves web interface, handles uploads, displays results

**Evaluation Notebook:**
- Location: `notebooks/evaluate.ipynb`
- Triggers: Jupyter execution
- Responsibilities: Loads trained models, runs full evaluation suite, generates figures for thesis

**Data Preparation Script:**
- Location: `scripts/prepare_data.py`
- Triggers: One-time setup execution
- Responsibilities: Downloads datasets, cleans metadata, splits train/val/test, preprocesses images

## Error Handling

**Strategy:** Fail-fast during training with informative logs; graceful degradation during inference

**Patterns:**
- Missing clinical features: Imputation using median values or model-level handling
- Corrupt/missing images: Skip with warning, log for review
- Model loading failures: Clear error messages with checkpoint path info
- GPU OOM: Automatic batch size reduction or CPU fallback

## Cross-Cutting Concerns

**Logging:** MLflow or Weights & Biases for experiment tracking; Python logging module for debugging

**Validation:** pydantic for input data validation; JSON schemas for configuration files

**Authentication:** None planned for local deployment; UI is single-user medical workstation style

**Reproducibility:** Fixed random seeds (numpy, torch, python), requirement.txt/poetry.lock, Docker support planned

---
*Architecture analysis: 2026-03-20*
