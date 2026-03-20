# External Integrations

**Analysis Date:** 2026-03-20

## APIs & External Services

**No external APIs detected** - This is a self-contained research project using publicly downloadable datasets.

**Potential Future Integrations:**
- MLflow Tracking Server or Weights & Biases cloud - For experiment tracking (optional)

## Data Storage

**Datasets (Public Medical Imaging):**

1. **CBIS-DDSM (Primary Training)**
   - Source: The Cancer Imaging Archive (TCIA)
   - Size: 1,566 patients | 10,239 mammograms
   - Content: DICOM images, ROI annotations, BI-RADS, density, lesion type
   - URL: https://www.cancerimagingarchive.net/collection/cbis-ddsm/
   - Citation: `10.7937/K9/TCIA.2016.7O02S9CY`
   - License: CC BY 3.0

2. **CMMD (External Validation)**
   - Source: TCIA
   - Size: 1,775 patients | 3,728 mammograms
   - Content: Biopsy-confirmed cases, molecular subtypes (CMMD2)
   - URL: https://www.cancerimagingarchive.net/collection/cmmd/
   - Citation: `10.7937/tcia.eqq2-h416`

3. **VinDr-Mammo (Optional)**
   - Source: PhysioNet
   - Size: ~5,000 exams | 20,000 images
   - Content: Full BI-RADS (1-5), Asian population
   - URL: https://physionet.org/content/vindr-mammo/

**Databases:**
- **Local filesystem only** - No external database servers
- Data stored as:
  - DICOM/PNG images in `data/raw/`
  - CSV metadata files
  - Processed numpy arrays in `data/processed/`

**File Storage:**
- Local filesystem - No cloud storage integration

**Caching:**
- Not applicable - Dataset processed once and stored locally

## Authentication & Identity

**Auth Provider:**
- **None** - No user authentication system
- Single-user academic/research application
- If deployed: Simple Gradio/Streamlit interface without auth

## Monitoring & Observability

**Error Tracking:**
- None (standard Python logging to console/files recommended)

**Logs:**
- Standard Python logging
- MLflow or Weights & Biases for experiment metrics

**Experiment Tracking:**
- MLflow (local or remote server) OR
- Weights & Biases (cloud)

## CI/CD & Deployment

**Hosting:**
- Local execution only
- No CI/CD pipelines detected
- Deployment target: Medical workstation/laptop

**CI Pipeline:**
- Not implemented

## Environment Configuration

**Required environment variables:**
- None required (all datasets are public)

**Optional configuration:**
- `MLFLOW_TRACKING_URI` - If using MLflow server
- `WANDB_API_KEY` - If using Weights & Biases
- Dataset paths (prefer relative paths in config file)

**Secrets location:**
- Not applicable (no secrets required)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None (batch processing only)

## Data Download Methods

**Primary Method - Kaggle:**
```python
import kagglehub
path = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
```

**Alternative - TCIA:**
```python
from tcia_utils import nbia
# Download via TCIA REST API
```

**Manual Download:**
- Direct from TCIA website for CBIS-DDSM and CMMD
- PhysioNet for VinDr-Mammo

## Dataset Citation Requirements

**Academic citations required in thesis:**
- CBIS-DDSM: `10.7937/K9/TCIA.2016.7O02S9CY`
- CMMD: `10.7937/tcia.eqq2-h416`
- TCGA-BRCA (if used): `10.7937/K9/TCIA.2016.AB2NAZRP`

---

*Integration audit: 2026-03-20*
