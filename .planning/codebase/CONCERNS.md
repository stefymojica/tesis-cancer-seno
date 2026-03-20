# Codebase Concerns

**Analysis Date:** 2026-03-20

## Tech Debt

**Empty Codebase - Project Not Started:**
- Issue: No source code exists beyond planning documents. The project has extensive documentation (Design Doc, Roadmap, Requirements, Phase Plans) but zero implementation.
- Files: `.planning/PROJECT.md`, `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`
- Impact: All 14 requirements (DATA-01 through UI-02) are marked as pending. Project is in Phase 1 planning but no code written.
- Fix approach: Begin implementation following the documented phases. Start with `requirements.txt` and `src/data/make_dataset.py` per Phase 1 Plan 2.

**Missing Dependency Management:**
- Issue: No `requirements.txt`, `setup.py`, `pyproject.toml`, or `environment.yml` exists.
- Impact: Cannot reproduce development environment. Dependencies must be inferred from planning documents (pandas, scikit-learn, xgboost, pydicom, opencv-python mentioned in research).
- Fix approach: Create `requirements.txt` as specified in `.planning/phases/01-preparacion-y-baseline-clinico/01-preparacion-y-baseline-clinico-02-PLAN.md` Task 1.

**No Project Structure:**
- Issue: No `src/`, `tests/`, `notebooks/`, or `data/` directories exist.
- Impact: Code organization will be ad-hoc when development starts, risking technical debt accumulation.
- Fix approach: Create standard Python ML project structure as outlined in `.planning/phases/01-preparacion-y-baseline-clinico/01-RESEARCH.md`:
  ```
  src/
  ├── data/make_dataset.py
  ├── data/preprocess.py
  └── models/train_clinical.py
  ```

## Known Risks

**Multimodal ML Complexity:**
- Risk: The project combines three complex domains: clinical tabular data, medical computer vision, and multimodal fusion.
- Files: Referenced in `docs/00_design_doc.md`
- Trigger: Each phase (Clinical baseline → Vision model → Fusion → XAI+Fairness) introduces new failure modes.
- Current mitigation: Well-documented phase planning exists.
- Workaround: Follow strict phase boundaries and validate success criteria before proceeding.

**Dataset Access Uncertainty:**
- Risk: CBIS-DDSM and INbreast datasets require registration and may have access restrictions.
- Files: Referenced in `.planning/phases/01-preparacion-y-baseline-clinico/01-RESEARCH.md`
- Impact: Cannot begin Phase 1 (DATA-01) without confirmed dataset access.
- Recommendation: Verify dataset access immediately. Have fallback datasets ready.

**Class Imbalance (Medical Data):**
- Risk: Medical datasets typically have severe class imbalance (many benign cases, few malignant).
- Impact: Models may achieve high accuracy by predicting majority class while failing to detect cancer.
- Current mitigation: Documented in research as "Pitfall 1" with recommendation to use `scale_pos_weight` in XGBoost and focus on AUC/F1 metrics.
- Files: `.planning/phases/01-preparacion-y-baseline-clinico/01-RESEARCH.md` lines 130-135

**Data Leakage Risk:**
- Risk: High probability of data leakage between train/test splits, especially with patient-level data.
- Impact: Inflated performance metrics that don't generalize to real patients.
- Current mitigation: Research document explicitly warns against fitting scalers on full dataset before split.
- Recommendation: Implement strict patient-level splits (not image-level) to prevent leakage between mammograms from same patient.
- Files: `.planning/phases/01-preparacion-y-baseline-clinico/01-RESEARCH.md` lines 113-116

**ID Alignment Errors:**
- Risk: Misalignment between clinical metadata and image files due to improper merging.
- Impact: Wrong image assigned to patient data → catastrophic diagnostic errors.
- Current mitigation: Research recommends strict `pd.merge()` with Patient_ID + View validation and assertions.
- Files: `.planning/phases/01-preparacion-y-baseline-clinico/01-RESEARCH.md` lines 136-140

## Security Considerations

**Minimal .gitignore:**
- Risk: Current `.gitignore` only ignores `docs/guia_tesis_meritoria.md`
- Files: `.gitignore` (3 lines)
- Impact: Risk of committing large datasets, model weights, or environment files with secrets.
- Current mitigation: None
- Recommendations: Add standard Python ML `.gitignore` entries:
  - `data/raw/`, `data/processed/` (large medical images)
  - `*.pkl`, `*.joblib` (model artifacts)
  - `.env`, `*.env` (secrets)
  - `__pycache__/`, `*.pyc`
  - `.ipynb_checkpoints/`

**No Secrets Management:**
- Risk: No mechanism for managing API keys (Kaggle, DICOM servers) or database credentials.
- Impact: Credentials may be hardcoded or committed to git.
- Recommendation: Create `.env.example` template and add python-dotenv to requirements.

## Performance Bottlenecks

**Medical Image Processing:**
- Problem: DICOM images are large (16-bit, high resolution). Loading and preprocessing entire dataset may exhaust memory.
- Files: Referenced in `docs/00_design_doc.md` and `.planning/phases/01-preparacion-y-baseline-clinico/01-RESEARCH.md`
- Cause: Loading full CBIS-DDSM dataset (thousands of high-res mammograms) into memory.
- Improvement path: Implement data generators (`tf.data` or PyTorch `DataLoader`) with batch loading. Use memory-mapped arrays for processed images.

**XGBoost on Large Clinical Data:**
- Problem: If clinical metadata grows (multiple views per patient, temporal data), training may slow.
- Recommendation: Start with XGBoost's `tree_method='hist'` for faster training on larger datasets.

## Fragile Areas

**Hardcoded Paths Risk:**
- Files: To be created per plans (`src/data/make_dataset.py`, `src/data/preprocess.py`)
- Why fragile: Plans explicitly warn against hardcoding local paths like `/Users/nombre/dataset/...`
- Safe modification: Use `pathlib.Path` with project-relative paths from day one.
- Test coverage: Currently none (no tests exist)

**Manual Data Preprocessing:**
- Why fragile: Research highlights that manual DICOM pixel normalization without header metadata can produce incorrect windowing.
- Files: `.planning/phases/01-preparacion-y-baseline-clinico/01-RESEARCH.md` lines 166-163
- Safe modification: Use `pydicom` windowing functions rather than simple min-max scaling.

## Scaling Limits

**Single-Developer Architecture:**
- Current capacity: Single thesis student developer
- Limit: No CI/CD, no code review process, no automated testing framework
- Scaling path: As complexity grows (Phases 2-4), introduce pytest for automated testing and pre-commit hooks for code quality.

**Local-Only Development:**
- Current setup: Laptop-based development intended (per Out of Scope in `REQUIREMENTS.md` line 40)
- Limit: No cloud GPU resources identified for training vision models (ResNet/EfficientNet/ViT).
- Scaling path: May need Google Colab, Kaggle Notebooks, or university HPC resources for Phase 2 (Vision model training).

## Dependencies at Risk

**XGBoost + SHAP Integration:**
- Risk: SHAP values for XGBoost can be slow to compute on large feature sets.
- Impact: May become bottleneck in Phase 3 (XAI) when explaining clinical model.
- Migration plan: Consider faster approximations like `shap.TreeExplainer` specifically designed for tree models.

**Medical Imaging Libraries:**
- Risk: `pydicom` and `opencv-python` can have compatibility issues with certain Python versions.
- Impact: Installation problems for thesis evaluators.
- Recommendation: Pin specific versions in `requirements.txt` after initial testing.

## Missing Critical Features

**Testing Framework:**
- Problem: No testing infrastructure exists.
- Blocks: Cannot verify correctness of data alignment, preprocessing, or model predictions.
- Priority: HIGH - Add pytest and basic unit tests for data pipeline functions.

**Experiment Tracking:**
- Problem: No MLflow, Weights & Biases, or similar experiment tracking configured.
- Blocks: Cannot compare model variants across phases or reproduce results for thesis defense.
- Priority: MEDIUM - Add lightweight tracking (even CSV logs) for model metrics.

**Reproducibility Seeds:**
- Problem: No random seed management specified in plans.
- Blocks: Results cannot be reproduced exactly for thesis validation.
- Priority: HIGH - Add seed setting to all training scripts from day one.

## Test Coverage Gaps

**All Areas Untested:**
- What's not tested: Literally everything (no code exists).
- Files: No `tests/` directory, no `*_test.py` or `test_*.py` files.
- Risk: All 14 requirements lack automated verification.
- Priority: HIGH - Create test infrastructure before writing significant implementation code (test-first approach).

---

*Concerns audit: 2026-03-20*
