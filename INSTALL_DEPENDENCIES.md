# Installation Guide

## Quick Fix for Missing Dependencies

If you're getting errors about missing modules when running the web application, install all dependencies:

### Option 1: Install All Dependencies (Recommended)
```bash
pip install -r requirements.txt
```

### Option 2: Install Missing Dependencies Individually

#### For XGBoost Models
```bash
pip install xgboost
```

#### For Interactive Charts
```bash
pip install plotly
```

#### For Streamlit Web App
```bash
pip install streamlit
```

## Complete Installation

To ensure everything works correctly, install all dependencies:

```bash
# Activate your virtual environment (if using one)
# For Windows:
.venv\Scripts\activate

# For Linux/Mac:
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

## Verify Installation

After installing, verify the packages are available:

```python
python -c "import xgboost; import plotly; import streamlit; print('All dependencies installed!')"
```

## Troubleshooting

### Issue: "No module named 'xgboost'"
**Solution:** Run `pip install xgboost`

### Issue: "No module named 'plotly'"
**Solution:** Run `pip install plotly`

### Issue: Models won't load
**Solution:** 
1. Make sure you've run the training pipeline: `python scripts/run_all.py`
2. Check that model files exist in `outputs/regression/` and `outputs/classification/`
3. Install xgboost if XGBoost models are failing: `pip install xgboost`

### Issue: Streamlit app won't start
**Solution:** 
1. Install streamlit: `pip install streamlit`
2. Make sure you're in the project root directory
3. Run: `streamlit run app.py`

## Requirements File Contents

The `requirements.txt` includes:
- pandas, numpy (data processing)
- matplotlib, seaborn, plotly (visualization)
- scikit-learn, xgboost (machine learning)
- statsmodels, scipy (statistical analysis)
- streamlit (web application)
- And other supporting libraries

Install everything at once with: `pip install -r requirements.txt`

