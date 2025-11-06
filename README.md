# Hyperliquid Sentiment-Behavior-PnL Analysis

## Project Overview

This project is a comprehensive data science analysis that investigates **how market sentiment (Fear vs. Greed Index) affects trader behavior and profitability on Hyperliquid**, a Web3 decentralized exchange. The project delivers a complete end-to-end workflow from data ingestion to predictive modeling, generating actionable insights suitable for both technical analysis and executive-level presentation.

### Key Questions Answered

- **Does market sentiment significantly impact trader profitability?**
- **Which sentiment conditions (Fear vs. Greed) lead to better trading outcomes?**
- **Can we predict trader profitability based on sentiment and trading features?**
- **What are the behavioral patterns of traders during different sentiment phases?**

---

## 🎯 Project Objectives

1. **Data Integration**: Merge Fear & Greed Index data with Hyperliquid historical trades
2. **Exploratory Analysis**: Discover patterns, correlations, and distributions
3. **Statistical Testing**: Validate hypotheses about sentiment-performance relationships
4. **Feature Engineering**: Create predictive features from raw data
5. **Predictive Modeling**: Build models to forecast profitability
6. **Visualization**: Create interactive dashboards and charts
7. **Insights Generation**: Produce actionable recommendations

---

## 📊 Data Requirements

### Input Files

Place these CSV files in the project root directory:

1. **`fear_greed_index.csv`**
   - **Columns**: `timestamp`, `value`, `classification`, `date`
   - **Description**: Historical Fear & Greed Index data
   - **Format**: Daily sentiment scores (0-100) with classifications (Fear, Greed, Neutral, etc.)

2. **`historical_data.csv`**
   - **Columns**: `Account`, `Coin`, `Execution Price`, `Size Tokens`, `Size USD`, `Side`, `Timestamp IST`, `Start Position`, `Direction`, `Closed PnL`, `Transaction Hash`, `Order ID`, `Crossed`, `Fee`, `Trade ID`, `Timestamp`
   - **Description**: Historical trading data from Hyperliquid
   - **Format**: Individual trade records with PnL, timestamps, and trader accounts

### Data Characteristics

- **Total Trades Analyzed**: 211,224 trades
- **Date Range**: March 2023 to June 2025
- **Sentiment Coverage**: February 2018 to May 2025
- **Time Zone**: All timestamps standardized to UTC

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB+ RAM recommended
- Internet connection (for initial package installation)

### Step-by-Step Installation

#### 1. Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd intern

# Or simply navigate to the project directory
cd D:\intern
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt
python -m venv .venv
.venv\Scripts\activate.bat

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies

**Option A: Install all at once (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Use the helper script**
```bash
python install_deps.py
```

**Option C: Install individually**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly xgboost statsmodels scipy shap streamlit tabulate pytz python-dateutil
```

#### 4. Verify Installation

```bash
python -c "import pandas, numpy, streamlit, xgboost, plotly; print('All dependencies installed successfully!')"
```

### Required Packages

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.1.0 | Data manipulation and analysis |
| numpy | ≥1.26.0 | Numerical computations |
| matplotlib | ≥3.8.0 | Static plotting |
| seaborn | ≥0.13.0 | Statistical visualizations |
| scikit-learn | ≥1.3.0 | Machine learning models |
| plotly | ≥5.18.0 | Interactive visualizations |
| xgboost | ≥2.0.0 | Gradient boosting models |
| statsmodels | ≥0.14.0 | Statistical modeling |
| scipy | ≥1.11.0 | Statistical tests |
| shap | ≥0.43.0 | Model explainability |
| streamlit | ≥1.28.0 | Web application framework |
| tabulate | ≥0.9.0 | Table formatting |

---

## 📁 Project Structure

```
intern/
│
├── 📄 README.md                          # This comprehensive guide
├── 📄 requirements.txt                   # Python dependencies
├── 📄 install_deps.py                   # Dependency installation helper
├── 📄 app.py                            # Streamlit web application
│
├── 📊 Data Files
│   ├── fear_greed_index.csv             # Fear & Greed Index data
│   └── historical_data.csv              # Hyperliquid trading data
│
├── 📂 src/                               # Source code modules
│   ├── __init__.py
│   ├── data_prep.py                     # Data loading and preprocessing
│   ├── eda.py                           # Exploratory data analysis
│   ├── features.py                      # Feature engineering
│   ├── stats_tests.py                   # Statistical hypothesis testing
│   ├── modeling.py                      # Machine learning models
│   ├── viz.py                           # Visualization functions
│   └── report_generator.py             # Report generation
│
├── 📂 scripts/
│   └── run_all.py                       # Main pipeline execution script
│
├── 📂 notebooks/
│   └── Hyperliquid_Sentiment_Analysis.ipynb  # Jupyter notebook for exploration
│
├── 📂 outputs/                           # Generated outputs
│   ├── regression/                      # Regression model files
│   │   ├── LinearRegression.joblib
│   │   ├── RidgeCV.joblib
│   │   ├── XGBRegressor.joblib
│   │   └── regression_results.json
│   ├── classification/                  # Classification model files
│   │   ├── LogisticRegression.joblib
│   │   ├── RandomForest.joblib
│   │   ├── XGBClassifier.joblib
│   │   └── classification_results.json
│   ├── *.csv                            # Statistical summaries
│   ├── *.png                            # Static charts
│   └── *.html                           # Interactive visualizations
│
└── 📂 reports/                           # Generated reports
    ├── executive_summary.md             # Markdown executive summary
    └── comprehensive_analysis_report.txt # Human-readable text report
```

---

## 🔄 Running the Analysis

### Quick Start

1. **Ensure data files are in place**
   ```bash
   # Verify files exist
   ls fear_greed_index.csv historical_data.csv
   ```

2. **Run the complete pipeline**
   ```bash
   python scripts/run_all.py
   ```

3. **Launch the web application**
   ```bash
   streamlit run app.py
   ```

### Detailed Pipeline Execution

The pipeline consists of 6 main steps:

```
[1/6] Loading and preparing data...
[2/6] Running EDA and generating visualizations...
[3/6] Running statistical tests...
[4/6] Training predictive models...
[5/6] Generating executive summary...
[6/6] Generating comprehensive human-readable report...
```

**Expected Runtime**: 5-15 minutes depending on system performance

---

## 🔍 Understanding the Pipeline

### Step 1: Data Preparation (`src/data_prep.py`)

**What it does:**
- Loads and parses CSV files
- Standardizes timestamps to UTC
- Handles missing values and outliers
- Merges sentiment data with trades
- Normalizes trading metrics

**Key Operations:**
- **Timestamp Parsing**: Converts Unix timestamps (milliseconds) and IST timestamps to UTC
- **Data Cleaning**: Removes invalid entries, handles NaN values
- **Outlier Treatment**: Winsorization (clipping at 1st and 99th percentiles)
- **Sentiment Merging**: Aligns daily sentiment scores with trade timestamps

**Output**: Clean, merged dataset ready for analysis

### Step 2: Exploratory Data Analysis (`src/eda.py`)

**What it does:**
- Computes descriptive statistics
- Analyzes PnL distributions by sentiment
- Creates correlation matrices
- Generates time series visualizations

**Key Metrics Generated:**
- Mean, median, standard deviation for PnL, trade value, execution price
- PnL breakdown by sentiment class (Fear, Greed, Neutral)
- Trade direction frequency (Long vs Short)
- Correlation coefficients between variables

**Visualizations Created:**
- PnL distribution plots (KDE)
- Correlation heatmaps
- Daily time series charts
- Direction frequency charts

### Step 3: Statistical Testing (`src/stats_tests.py`)

**What it does:**
- Performs t-tests to compare Fear vs Greed performance
- Calculates correlation tests
- Runs OLS regression analysis

**Key Tests:**
1. **T-Test (Fear vs Greed)**
   - Tests if mean PnL differs significantly between sentiment conditions
   - **Result**: Highly significant (p < 0.001)
   - **Finding**: Greed markets show higher average PnL

2. **Correlation Tests**
   - Spearman correlation between sentiment score and PnL
   - Tests relationships between trade size, price, and profitability

3. **OLS Regression**
   - Estimates linear relationship: `PnL = f(sentiment, trade_value, execution_price, side)`
   - Provides coefficient estimates and significance tests

### Step 4: Feature Engineering (`src/features.py`)

**What it does:**
- Creates rolling averages (7-day sentiment, 3-day PnL)
- Computes trader performance metrics (win rate, profit factor)
- Generates sentiment transition flags
- Builds feature matrix for modeling

**Features Created:**
- `sentiment_ma_7d`: 7-day moving average of sentiment
- `pnl_ma_3d`: 3-day moving average of PnL
- `win_rate_14d`: 14-day rolling win rate per trader
- `profit_factor_14d`: 14-day rolling profit factor
- `sentiment_score`: Normalized sentiment (0-1)
- `trade_value_usd_robust`: Robust-scaled trade value

### Step 5: Predictive Modeling (`src/modeling.py`)

**What it does:**
- Trains regression models to predict PnL amount
- Trains classification models to predict win/loss
- Evaluates model performance
- Saves trained models for deployment

**Regression Models:**
1. **Linear Regression**: Baseline linear model
2. **Ridge Regression**: Regularized linear model
3. **XGBoost Regressor**: Gradient boosting for regression

**Classification Models:**
1. **Logistic Regression**: Baseline classification
2. **Random Forest**: Ensemble tree-based model
3. **XGBoost Classifier**: Gradient boosting for classification

**Evaluation Metrics:**
- **Regression**: R² Score, RMSE (Root Mean Squared Error)
- **Classification**: Accuracy, F1-Score, Precision, Recall

### Step 6: Report Generation (`src/report_generator.py`)

**What it does:**
- Compiles all results into human-readable format
- Generates executive summary
- Creates comprehensive analysis report

---

## 📊 Outputs & Results

### Generated Files Overview

#### 1. Statistical Summaries (CSV Files)

**`outputs/descriptive_stats.csv`**
- Summary statistics for PnL, trade value, execution price
- Includes count, mean, std, min, max, percentiles

**`outputs/pnl_by_sentiment.csv`**
- PnL statistics grouped by sentiment class
- Shows count, mean, median, std, sum for each sentiment

**`outputs/ttest_fear_vs_greed.csv`**
- T-test results comparing Fear vs Greed
- Contains t-statistic, p-value, sample sizes

**`outputs/correlation_matrix.csv`**
- Correlation coefficients between key variables
- Spearman correlation matrix

**`outputs/correlation_tests.csv`**
- Statistical tests for correlations
- P-values for correlation significance

**`outputs/direction_frequency.csv`**
- Frequency of long vs short trades by sentiment

#### 2. Visualizations

**Static Images (PNG):**
- `pnl_distribution.png`: Probability density of PnL by sentiment
- `correlation_heatmap.png`: Correlation matrix visualization

**Interactive Charts (HTML):**
- `daily_timeseries.html`: Daily aggregates over time
- `sentiment_vs_pnl.html`: Sentiment vs total PnL line chart
- `volume_vs_sentiment.html`: Trade volume vs sentiment
- `top_traders_by_sentiment.html`: Top 10 traders per sentiment phase

#### 3. Trained Models

**Regression Models** (`outputs/regression/`):
- `LinearRegression.joblib`
- `RidgeCV.joblib`
- `XGBRegressor.joblib`
- `regression_results.json`: Performance metrics

**Classification Models** (`outputs/classification/`):
- `LogisticRegression.joblib`
- `RandomForest.joblib`
- `XGBClassifier.joblib`
- `classification_results.json`: Performance metrics

#### 4. Reports

**`reports/executive_summary.md`**
- Markdown format summary
- Key findings and metrics
- Suitable for presentations

**`reports/comprehensive_analysis_report.txt`**
- Detailed human-readable report
- Complete analysis with explanations
- Strategic recommendations

---

## 🌐 Web Application Guide

### Launching the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Application Features

#### 🏠 Home Page
- **Key Metrics Dashboard**: Overview of average PnL, sentiment performance
- **Model Performance**: Summary of best models
- **Statistical Test Results**: T-test and correlation results

#### 🔮 Model Predictions
- **Interactive Prediction Form**: Input trade features
  - Sentiment Score (0-1 slider)
  - Trade Value (USD)
  - Execution Price
  - Sentiment Class (radio buttons)
  - Trade Side (Long/Short)
- **Real-time Predictions**: 
  - Regression: Predicted PnL amount
  - Classification: Win/Loss probability with confidence

#### 📈 Data Explorer
Organized into 5 tabs:

1. **Overview**: Descriptive statistics with bar charts
2. **PnL Analysis**: 
   - PnL by sentiment tables
   - Interactive bar and pie charts
   - Distribution plots
3. **Correlations**: 
   - Correlation matrix tables
   - Interactive heatmaps
   - Test results
4. **Distributions**: 
   - Trade direction frequency
   - Stacked bar charts
5. **Statistics**: 
   - T-test results with metrics
   - Significance indicators

#### 📊 Visualizations
Organized into 4 tabs:

1. **Time Series**: Daily aggregates, sentiment vs PnL
2. **Performance**: Volume vs sentiment, model comparisons
3. **Top Traders**: Top 10 traders by sentiment
4. **All Charts**: Complete visualization gallery

#### 📄 Report Viewer
- View comprehensive analysis report
- Download report as text file
- Generate report on-demand

---

## 📈 Results Interpretation

### Key Findings

#### 1. Statistical Significance

**T-Test Result: Fear vs Greed**
- **T-Statistic**: -16.407
- **P-Value**: 2.27e-60 (extremely significant)
- **Interpretation**: There is overwhelming statistical evidence that trader performance differs significantly between Fear and Greed market conditions.

**Sample Sizes:**
- Fear periods: 133,871 trades
- Greed periods: 43,251 trades

#### 2. Performance by Sentiment

Based on the analysis of 211,224 trades:

| Sentiment | Average PnL | Total PnL | Number of Trades |
|-----------|-------------|-----------|------------------|
| **Greed** | $43.89 | $1.90M | 43,251 |
| **Fear** | $29.97 | $4.01M | 133,871 |
| **Neutral** | $29.78 | $0.21M | 7,141 |
| **Unknown** | $18.18 | $0.49M | 26,961 |

**Key Insight**: 
- **Greed markets show 46% higher average PnL** than Fear markets ($43.89 vs $29.97)
- However, Fear markets have more total volume, resulting in higher aggregate PnL
- Traders tend to perform better during optimistic (Greed) market conditions

#### 3. Model Performance

**Best Regression Model: XGBoost Regressor**
- **R² Score**: 0.998 (99.8% variance explained)
- **RMSE**: $6.12
- **Interpretation**: The model can predict PnL with very high accuracy

**Best Classification Model: XGBoost Classifier**
- **Accuracy**: 99.9%
- **F1-Score**: 0.999
- **Interpretation**: The model can predict win/loss with near-perfect accuracy

#### 4. Correlation Insights

- **Sentiment Score vs PnL**: Positive correlation (higher sentiment = higher PnL)
- **Trade Value vs PnL**: Positive correlation (larger trades = higher PnL)
- **Execution Price**: Negative correlation with PnL (price impact)

### Strategic Recommendations

1. **Dynamic Risk Management**
   - Reduce position sizes during Fear periods
   - Increase activity during Greed periods when average PnL is higher

2. **Sentiment-Driven Strategy**
   - Use Fear & Greed Index as a feature in algorithmic trading
   - Optimize entry/exit timing based on sentiment phases

3. **Portfolio Optimization**
   - Allocate capital differently based on sentiment
   - Consider sentiment-based position sizing

4. **Model Deployment**
   - Deploy XGBoost models for real-time profitability forecasting
   - Use predictions to guide trading decisions

---

## 🔧 Technical Details

### Data Processing Pipeline

1. **Timestamp Standardization**
   - All timestamps converted to UTC
   - Handles multiple input formats (Unix milliseconds, IST strings)
   - Daily aggregation for sentiment alignment

2. **Data Cleaning**
   - Missing value handling: Forward/backward fill for sentiment
   - Outlier treatment: Winsorization at 1st and 99th percentiles
   - Type conversion: Numeric columns coerced, categorical encoded

3. **Feature Engineering**
   - Rolling windows: 3-day PnL, 7-day sentiment
   - Trader-level metrics: 14-day win rate, profit factor
   - Robust scaling: Median and IQR-based normalization

### Model Architecture

**Regression Pipeline:**
```
Input Features → Robust Scaling → Model Training → Evaluation
```

**Classification Pipeline:**
```
Input Features → Robust Scaling → Model Training → Evaluation → Probability Calibration
```

**Feature Set:**
- Numeric: sentiment_score, trade_value_usd, execution_price, rolling averages
- Categorical: side_norm (long/short), sentiment_class (one-hot encoded)
- Engineered: win_rate_14d, profit_factor_14d, normalized metrics

### Statistical Methods

- **T-Test**: Independent samples t-test (Welch's t-test for unequal variances)
- **Correlation**: Spearman rank correlation (non-parametric)
- **Regression**: Ordinary Least Squares (OLS) with robust standard errors
- **Cross-Validation**: 5-fold cross-validation for model evaluation

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'xgboost'
ModuleNotFoundError: No module named 'plotly'
```

**Solution:**
```bash
pip install -r requirements.txt
# Or individually:
pip install xgboost plotly streamlit
```

#### Issue 2: Models Won't Load

**Symptoms:**
```
Could not load XGBRegressor: No module named 'xgboost'
```

**Solution:**
1. Install xgboost: `pip install xgboost`
2. Re-run pipeline: `python scripts/run_all.py`
3. Verify model files exist in `outputs/regression/` and `outputs/classification/`

#### Issue 3: Timestamp Parsing Errors

**Symptoms:**
```
Date ranges - Trades: 1970-01-01 to 1970-01-01
```

**Solution:**
- The code automatically handles Unix timestamps in milliseconds
- If issues persist, check CSV format matches expected structure

#### Issue 4: Empty DataFrames

**Symptoms:**
```
Warning: Account column not found
Warning: Sentiment merge: 0/211224 trades matched
```

**Solution:**
- Verify CSV files are in the project root
- Check column names match expected format
- Ensure data files are not corrupted

#### Issue 5: Streamlit App Errors

**Symptoms:**
```
SyntaxError: invalid syntax
Progress Value has invalid type: float32
```

**Solution:**
- Update to latest code version
- Restart Streamlit: `streamlit run app.py`
- Clear cache: `streamlit cache clear`

### Getting Help

1. **Check Logs**: Review terminal output for detailed error messages
2. **Verify Dependencies**: Run `pip list` to check installed packages
3. **Test Installation**: Run `python -c "import streamlit; print('OK')"`
4. **Review Documentation**: See `INSTALL_DEPENDENCIES.md` and `WEB_APP_GUIDE.md`

---

## 🚀 Future Enhancements

### Potential Improvements

1. **Real-time Data Integration**
   - API connections to live Fear & Greed Index
   - Real-time Hyperliquid trade data streaming

2. **Advanced Models**
   - Deep learning models (LSTM, Transformer)
   - Ensemble methods combining multiple models
   - Time series forecasting models

3. **Enhanced Features**
   - Technical indicators (RSI, MACD, Bollinger Bands)
   - Market volatility metrics
   - Liquidity indicators

4. **Deployment**
   - Docker containerization
   - Cloud deployment (AWS, GCP, Azure)
   - REST API for model serving

5. **Additional Analysis**
   - Trader clustering analysis
   - Market regime detection
   - Risk-adjusted performance metrics (Sharpe ratio, Sortino ratio)

---

## 📚 Additional Resources

### Documentation Files

- **`INSTALL_DEPENDENCIES.md`**: Detailed dependency installation guide
- **`WEB_APP_GUIDE.md`**: Complete web application user guide
- **`reports/comprehensive_analysis_report.txt`**: Full analysis report
- **`reports/executive_summary.md`**: Executive summary

### Code Documentation

- All source code includes docstrings
- Type hints provided for function signatures
- Comments explain complex logic

### External Resources

- **Fear & Greed Index**: https://alternative.me/crypto/fear-and-greed-index/
- **Hyperliquid**: https://hyperliquid.xyz/
- **Streamlit Documentation**: https://docs.streamlit.io/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/

---

## 📝 License & Credits

### Data Sources

- **Fear & Greed Index**: Alternative.me
- **Hyperliquid Trading Data**: Provided dataset

### Technologies Used

- Python 3.8+
- pandas, numpy for data processing
- scikit-learn, xgboost for machine learning
- plotly, matplotlib, seaborn for visualization
- streamlit for web application
- statsmodels, scipy for statistical analysis

---

## ✅ Summary

This project provides a **complete end-to-end data science solution** for analyzing the relationship between market sentiment and trader performance. It includes:

✅ **Data Integration**: Seamless merging of sentiment and trading data  
✅ **Comprehensive Analysis**: Statistical tests, EDA, and modeling  
✅ **Predictive Models**: High-accuracy regression and classification  
✅ **Interactive Dashboard**: User-friendly web application  
✅ **Detailed Reports**: Human-readable insights and recommendations  
✅ **Production Ready**: Modular code, error handling, documentation  

**Ready to use for:**
- Quantitative trading research
- Market sentiment analysis
- Trader behavior studies
- Executive presentations
- Technical interviews

---

**For questions or issues, refer to the troubleshooting section or review the code documentation.**

**Happy Analyzing! 📊🚀**
# Hyperliquid-Sentiment-Behavior-PnL-Analysis
