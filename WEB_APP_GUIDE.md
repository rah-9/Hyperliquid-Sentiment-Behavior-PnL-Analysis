# Web Application Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Analysis Pipeline

First, run the complete analysis pipeline to generate all outputs and train models:

```bash
python scripts/run_all.py
```

This will:
- Load and prepare the data
- Run exploratory data analysis
- Perform statistical tests
- Train predictive models
- Generate visualizations
- Create comprehensive reports

### 3. Launch the Web Application

After the pipeline completes, launch the Streamlit web app:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Web Application Features

### 🏠 Home Page
- Overview of key metrics
- Model performance summaries
- Statistical test results
- Quick insights dashboard

### 🔮 Model Predictions
- Interactive form to input trade features
- Real-time predictions from trained models
- Regression predictions (PnL amount)
- Classification predictions (Win/Loss probability)

**Input Features:**
- Sentiment Score (0-1 slider)
- Trade Value (USD)
- Execution Price
- Sentiment Class (Fear/Greed/Neutral)
- Trade Side (Long/Short)

### 📈 Data Explorer
- Browse descriptive statistics
- View PnL by sentiment class
- Interactive charts and tables
- Statistical test results

### 📊 Visualizations
- Sentiment vs PnL interactive chart
- Volume vs Sentiment analysis
- Top traders by sentiment
- Daily time series trends

### 📄 Report Viewer
- View the comprehensive human-readable text report
- Download the report as a .txt file
- Generate report on-demand if missing

## Generated Reports

### Comprehensive Analysis Report (`reports/comprehensive_analysis_report.txt`)

This human-readable text report includes:

1. **Executive Summary**
   - Key findings and statistical significance
   - Best performing models

2. **Trader Performance Overview**
   - Overall trading statistics
   - Average PnL, trade values, execution prices

3. **Sentiment-Based Performance Analysis**
   - Performance breakdown by sentiment class
   - Best and worst performing market conditions

4. **Statistical Tests & Significance**
   - T-test results (Fear vs Greed)
   - Correlation analysis

5. **Predictive Models Performance**
   - Regression models (R², RMSE)
   - Classification models (Accuracy, F1-Score)

6. **Regression Analysis (OLS)**
   - Detailed OLS regression summary

7. **Key Insights & Strategic Recommendations**
   - Actionable insights for trading strategies
   - Recommendations for risk management

## Output Files

All outputs are saved in the `outputs/` directory:

- `regression/` - Trained regression models and results
- `classification/` - Trained classification models and results
- `*.csv` - Statistical summaries and test results
- `*.html` - Interactive visualizations
- `*.png` - Static charts

Reports are saved in the `reports/` directory:

- `executive_summary.md` - Markdown executive summary
- `comprehensive_analysis_report.txt` - Human-readable text report

## Troubleshooting

### Models Not Loading
If models fail to load in the web app, ensure you've run the training pipeline first:
```bash
python scripts/run_all.py
```

### Missing Visualizations
If visualizations don't appear, check that the HTML files exist in `outputs/` directory. Re-run the pipeline if needed.

### Report Not Found
The comprehensive report is automatically generated after training. If missing, click "Generate Report Now" in the Report Viewer page.

## Next Steps

1. **Explore the Data**: Use the Data Explorer to understand the dataset
2. **Test Predictions**: Try different input combinations in Model Predictions
3. **Review Insights**: Read the comprehensive report for strategic recommendations
4. **Analyze Visualizations**: Explore interactive charts to identify patterns

## Support

For issues or questions, check:
- The comprehensive analysis report for detailed findings
- The executive summary for quick insights
- The output CSV files for raw statistics

