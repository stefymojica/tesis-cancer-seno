# Testing Patterns

**Analysis Date:** 2026-03-20

## Test Framework

**Runner:**
- **Primary:** pytest 8.0+
- **Config:** `pyproject.toml` or `pytest.ini`
- **Plugins:**
  - `pytest-cov` (coverage)
  - `pytest-xdist` (parallel execution)
  - `pytest-mock` (mocking utilities)
  - `pytest-randomly` (random test order to catch state issues)

**Assertion Library:**
- Built-in `assert` with pytest enhancements
- NumPy testing: `numpy.testing.assert_array_almost_equal`
- PyTorch testing: `torch.testing.assert_close`
- Pandas testing: `pandas.testing.assert_frame_equal`

**Run Commands:**
```bash
pytest                      # Run all tests
pytest -x                   # Stop on first failure
pytest -v                   # Verbose output
pytest --cov=src --cov-report=html  # Coverage report
pytest -n auto              # Parallel execution
pytest --randomly-seed=42   # Reproducible random order
```

## Test File Organization

**Location:**
- **Co-located:** Not used - tests are separate
- **Test directory:** `tests/` at project root
- **Mirror structure:** `tests/unit/test_models/test_train_clinical.py` mirrors `src/models/train_clinical.py`

**Naming:**
- **Files:** `test_*.py` - e.g., `test_train_clinical.py`, `test_preprocess.py`
- **Functions:** `test_descriptive_name` - e.g., `test_clinical_model_outputs_correct_shape`
- **Classes:** `TestPascalCase` - e.g., `TestClinicalPreprocessor`, `TestMammographyDataset`

**Structure:**
```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests (fast, isolated)
│   ├── test_data/
│   │   ├── test_preprocess.py
│   │   └── test_transforms.py
│   ├── test_models/
│   │   ├── test_train_clinical.py
│   │   └── test_predict_clinical.py
│   ├── test_vision/
│   │   ├── test_backbones.py
│   │   └── test_features.py
│   └── test_fusion/
│       └── test_hierarchical.py
├── integration/                # Integration tests (slower, multi-component)
│   ├── test_pipeline.py        # End-to-end data -> model
│   ├── test_clinical_vision.py # Clinical + vision together
│   └── test_xai_integration.py # SHAP + Grad-CAM integration
├── fixtures/                   # Test data and artifacts
│   ├── sample_clinical.csv     # Minimal clinical dataset
│   ├── sample_dicom/           # Sample DICOM files
│   └── mock_models/            # Pre-trained mock models for testing
└── e2e/                        # End-to-end tests (slowest)
    └── test_full_pipeline.py   # Full system test
```

## Test Structure

**Suite Organization:**
```python
import pytest
import numpy as np
from src.models.train_clinical import train_clinical_model
from src.data.preprocess import ClinicalPreprocessor


class TestClinicalModel:
    """Tests for clinical risk prediction model."""
    
    def test_model_outputs_correct_shape(self, sample_clinical_data, trained_model):
        """Model should return probability vector with correct shape."""
        X = sample_clinical_data.drop('target', axis=1)
        predictions = trained_model.predict_proba(X)
        
        assert predictions.shape == (len(X), 2)  # Binary classification
        assert np.all((predictions >= 0) & (predictions <= 1))  # Valid probabilities
    
    def test_model_handles_missing_values(self, clinical_data_with_missing):
        """Model should raise appropriate error for missing values."""
        with pytest.raises(ValueError, match="Missing required columns"):
            train_clinical_model(clinical_data_with_missing)
    
    def test_reproducibility_with_seed(self, train_data):
        """Same seed should produce identical models."""
        model1 = train_clinical_model(train_data, random_state=42)
        model2 = train_clinical_model(train_data, random_state=42)
        
        # Predictions should be identical
        preds1 = model1.predict_proba(train_data)
        preds2 = model2.predict_proba(train_data)
        np.testing.assert_array_equal(preds1, preds2)


class TestClinicalPreprocessor:
    """Tests for clinical data preprocessing."""
    
    def test_preprocessor_fit_transform(self, raw_clinical_df):
        """Preprocessor should handle fit/transform correctly."""
        preprocessor = ClinicalPreprocessor()
        
        # Fit on training data only
        train_transformed = preprocessor.fit_transform(raw_clinical_df)
        
        # Transform should work on new data
        test_transformed = preprocessor.transform(raw_clinical_df.head(5))
        
        assert train_transformed.shape[1] == test_transformed.shape[1]
    
    def test_no_data_leakage_in_preprocessing(self, train_df, test_df):
        """Preprocessing stats should only come from train set."""
        preprocessor = ClinicalPreprocessor()
        preprocessor.fit(train_df)
        
        # Test that mean from train is used, not combined mean
        train_mean = train_df['age'].mean()
        assert preprocessor.age_mean_ == pytest.approx(train_mean, rel=1e-5)
```

**Patterns:**
- **Setup:** Use fixtures in `conftest.py` for common setup
- **Teardown:** Use `tmp_path` fixture for temporary files
- **Parametrization:** Test multiple scenarios with `@pytest.mark.parametrize`
- **Markers:** Use `@pytest.mark.slow`, `@pytest.mark.gpu` for test categorization

## Mocking

**Framework:** `unittest.mock` (standard library) + `pytest-mock` fixture

**Patterns:**
```python
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch


def test_vision_model_with_mocked_dataloader(mocker):
    """Test vision model without loading real images."""
    # Mock the dataset to return synthetic tensors
    mock_dataset = mocker.Mock()
    mock_dataset.__len__ = Mock(return_value=100)
    mock_dataset.__getitem__ = Mock(
        return_value=(torch.randn(3, 512, 512), torch.tensor(1))  # Fake image, label
    )
    
    # Mock DataLoader
    mocker.patch('torch.utils.data.DataLoader', return_value=mock_dataset)
    
    # Test model training loop without real data
    from src.models.train_vision import train_epoch
    model = Mock()
    optimizer = Mock()
    
    loss = train_epoch(model, mock_dataset, optimizer)
    assert model.forward.called


def test_clinical_prediction_with_mocked_model():
    """Test prediction logic with mocked XGBoost."""
    with patch('xgboost.XGBClassifier') as MockModel:
        # Configure mock
        mock_instance = MockModel.return_value
        mock_instance.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        # Test prediction function
        from src.models.predict_clinical import get_clinical_risk_vector
        risk = get_clinical_risk_vector(patient_data, model_path='fake_path')
        
        assert risk[1] == pytest.approx(0.7)


def test_shap_with_mocked_explainer(mocker):
    """Test SHAP integration without full model."""
    mock_explainer = mocker.patch('shap.TreeExplainer')
    mock_explainer.return_value.shap_values.return_value = np.random.randn(10, 5)
    
    from src.xai.shap_clinical import generate_shap_plot
    fig = generate_shap_plot(model_mock, data_mock)
    
    assert fig is not None
    mock_explainer.assert_called_once()
```

**What to Mock:**
- **External APIs:** Dataset downloads (CBIS-DDSM, CMMD)
- **Heavy computations:** Full model training, SHAP computations on large data
- **File I/O:** DICOM reading, model saving/loading
- **Randomness:** For reproducible tests, mock `np.random` or set seeds

**What NOT to Mock:**
- **Data transformations:** Test actual preprocessing logic
- **Model architecture:** Test layer definitions, forward pass
- **Metric calculations:** Test actual AUC, sensitivity calculations

## Fixtures and Factories

**Test Data:**
```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import torch


@pytest.fixture(scope='session')
def random_seed():
    """Set random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture
def sample_clinical_data():
    """Minimal clinical dataset for testing."""
    return pd.DataFrame({
        'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'age': [45, 55, 65, 35, 50],
        'density': [2, 3, 4, 1, 3],
        'family_history': [0, 1, 0, 0, 1],
        'birads_prev': [1, 2, 3, 1, 4],
        'target': [0, 1, 1, 0, 1]
    })


@pytest.fixture
def clinical_data_with_missing(sample_clinical_data):
    """Clinical data with missing values for edge case testing."""
    df = sample_clinical_data.copy()
    df.loc[0, 'age'] = np.nan
    df.loc[2, 'density'] = np.nan
    return df


@pytest.fixture
def mock_mammogram_tensor():
    """Synthetic mammogram tensor (no real image loading)."""
    return torch.randn(1, 3, 512, 512)  # Batch, channels, height, width


@pytest.fixture
def trained_model(sample_clinical_data, tmp_path):
    """Train a small model for testing (uses tmp_path for artifacts)."""
    from src.models.train_clinical import train_clinical_model
    
    model = train_clinical_model(
        sample_clinical_data,
        output_dir=tmp_path,
        n_estimators=10,  # Small for fast tests
        max_depth=3
    )
    return model


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    """Create temporary directory structure for dataset testing."""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    (data_dir / 'raw').mkdir()
    (data_dir / 'processed').mkdir()
    return data_dir


# Factory fixture for parametrized data
@pytest.fixture
def make_clinical_data():
    """Factory to create clinical data with specific properties."""
    def _make(n_samples=100, n_features=10, seed=42):
        np.random.seed(seed)
        data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        data['target'] = np.random.randint(0, 2, n_samples)
        return data
    return _make
```

**Location:**
- **Primary fixtures:** `tests/conftest.py` (shared across all tests)
- **Module fixtures:** `tests/unit/test_data/conftest.py` (data-specific)
- **Factory functions:** Use closures to generate variations

## Coverage

**Requirements:**
- **Target:** Minimum 80% coverage for `src/`
- **Critical paths:** 100% coverage for:
  - `src/data/preprocess.py`
  - `src/models/predict_clinical.py`
  - `src/models/predict_vision.py`
  - `src/evaluation/metrics.py`

**View Coverage:**
```bash
pytest --cov=src --cov-report=term-missing      # Terminal with missing lines
pytest --cov=src --cov-report=html                # HTML report in htmlcov/
pytest --cov=src --cov-report=xml                 # XML for CI integration
pytest --cov-fail-under=80                        # Fail if coverage < 80%
```

**Coverage Configuration (pyproject.toml):**
```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "src/ui/*",           # UI code tested manually
    "src/__main__.py",    # Entry point
    "*/__init__.py"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:"
]
```

## Test Types

**Unit Tests:**
- **Scope:** Single functions/classes in isolation
- **Speed:** Fast (< 1 second per test)
- **Mocking:** Heavy use of mocks for dependencies
- **Examples:**
  - Preprocessor transform logic
  - Individual metric calculations (AUC, sensitivity)
  - Model architecture layer definitions
  - Data validation functions

**Integration Tests:**
- **Scope:** Multiple components working together
- **Speed:** Medium (1-10 seconds per test)
- **Mocking:** Minimal - use real (small) datasets
- **Examples:**
  - Data pipeline: load → preprocess → train
  - Clinical + Vision fusion
  - SHAP/Grad-CAM generation pipeline
  - End-to-end prediction with all preprocessing

**E2E Tests:**
- **Scope:** Full system from input to output
- **Speed:** Slow (seconds to minutes)
- **Mocking:** None - use real components
- **Examples:**
  - Full training pipeline on small dataset subset
  - UI streamlit app loading and prediction
  - Complete inference: image + clinical → prediction + explanations

## Common Patterns

**Async Testing:**
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_data_loading():
    """Test async data loading if implemented."""
    result = await load_dataset_async()
    assert result is not None
```

**Error Testing:**
```python
def test_preprocessor_raises_on_invalid_data():
    """Preprocessor should raise on invalid input."""
    preprocessor = ClinicalPreprocessor()
    invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
    
    with pytest.raises(ValueError, match="Missing required columns"):
        preprocessor.fit(invalid_data)


def test_model_raises_on_mismatched_features():
    """Model should raise when features don't match training."""
    model = train_model(train_data_with_features_a)
    test_data_wrong_features = pd.DataFrame({'feature_b': [1, 2]})
    
    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict(test_data_wrong_features)
```

**ML-Specific Testing:**
```python
def test_no_data_leakage_between_splits():
    """Critical: No patient should appear in multiple splits."""
    train_ids = set(train_df['patient_id'])
    val_ids = set(val_df['patient_id'])
    test_ids = set(test_df['patient_id'])
    
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_prediction_probabilities_sum_to_one():
    """Binary classification probabilities should sum to 1."""
    probs = model.predict_proba(test_data)
    np.testing.assert_array_almost_equal(probs.sum(axis=1), np.ones(len(probs)))


def test_model_deterministic_with_fixed_seed():
    """Model training should be reproducible."""
    model1 = train_model(data, random_state=42)
    model2 = train_model(data, random_state=42)
    
    preds1 = model1.predict(test_data)
    preds2 = model2.predict(test_data)
    
    np.testing.assert_array_equal(preds1, preds2)


def test_shap_values_explain_prediction():
    """SHAP values should sum to prediction minus baseline."""
    shap_values = explainer.shap_values(test_instance)
    expected_sum = model.predict(test_instance) - explainer.expected_value
    
    np.testing.assert_almost_equal(np.sum(shap_values), expected_sum, decimal=5)


def test_gradcam_highlights_relevant_regions():
    """Grad-CAM should highlight non-empty regions."""
    heatmap = generate_gradcam(model, image, target_layer)
    
    assert heatmap.shape == (image_height, image_width)
    assert heatmap.max() > 0  # At least some activation
    assert not np.all(heatmap == heatmap[0, 0])  # Not uniform
```

**PyTorch-Specific Testing:**
```python
def test_model_forward_pass_output_shape():
    """Vision model should output expected embedding dimension."""
    model = VisionFeatureExtractor()
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    
    output = model(input_tensor)
    
    assert output.shape == (batch_size, 512)  # Expected embedding dim


def test_gradient_flow():
    """Gradients should flow through all layers during backprop."""
    model = HierarchicalFusionModel()
    input_clinical = torch.randn(2, 10)
    input_visual = torch.randn(2, 512)
    target = torch.tensor([1, 0])
    
    output = model(input_clinical, input_visual)
    loss = F.binary_cross_entropy(output, target.float())
    loss.backward()
    
    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no gradient"
        assert not torch.all(param.grad == 0), f"{name} has zero gradient"
```

## Running Tests

**During Development:**
```bash
pytest tests/unit/test_models/test_train_clinical.py -v  # Specific test file
pytest -k "test_clinical" -v                               # Tests matching pattern
pytest --lf                                                # Last failed only
pytest --ff                                                # Failed first, then others
```

**Pre-Commit:**
```bash
pytest tests/unit/ -x --tb=short                          # Quick unit tests only
```

**CI/CD Pipeline:**
```bash
pytest tests/unit/ tests/integration/ -v --cov=src --cov-report=xml --cov-fail-under=80
```

**Before Major Changes:**
```bash
pytest tests/ -v --randomly-seed=42                      # Full suite with fixed seed
```

---

*Testing analysis: 2026-03-20*
