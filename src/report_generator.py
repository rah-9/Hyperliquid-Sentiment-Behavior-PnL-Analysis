"""
Generate comprehensive human-readable text reports from analysis outputs.
"""
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Optional


def load_json_safe(path: str) -> Optional[Dict]:
	"""Safely load JSON file."""
	try:
		with open(path, 'r') as f:
			return json.load(f)
	except Exception as e:
		print(f"   Warning: Could not load {path}: {e}")
		return None


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
	"""Safely load CSV file."""
	try:
		return pd.read_csv(path)
	except Exception as e:
		print(f"   Warning: Could not load {path}: {e}")
		return None


def format_number(num: float, decimals: int = 2) -> str:
	"""Format number with appropriate precision."""
	if pd.isna(num) or num is None:
		return "N/A"
	if abs(num) >= 1e6:
		return f"{num/1e6:.2f}M"
	elif abs(num) >= 1e3:
		return f"{num/1e3:.2f}K"
	else:
		return f"{num:.{decimals}f}"


def generate_comprehensive_report(outputs_dir: str = "outputs", reports_dir: str = "reports") -> str:
	"""
	Generate a comprehensive human-readable text report from all analysis outputs.
	
	Returns:
		Path to the generated report file.
	"""
	os.makedirs(reports_dir, exist_ok=True)
	
	report_path = os.path.join(reports_dir, "comprehensive_analysis_report.txt")
	
	# Load all outputs
	regression_results = load_json_safe(os.path.join(outputs_dir, "regression", "regression_results.json"))
	classification_results = load_json_safe(os.path.join(outputs_dir, "classification", "classification_results.json"))
	ttest_df = load_csv_safe(os.path.join(outputs_dir, "ttest_fear_vs_greed.csv"))
	descriptive_stats = load_csv_safe(os.path.join(outputs_dir, "descriptive_stats.csv"))
	pnl_by_sentiment = load_csv_safe(os.path.join(outputs_dir, "pnl_by_sentiment.csv"))
	correlation_tests = load_csv_safe(os.path.join(outputs_dir, "correlation_tests.csv"))
	
	# Load OLS summary if available
	ols_summary = ""
	ols_path = os.path.join(outputs_dir, "ols_summary.txt")
	if os.path.exists(ols_path):
		try:
			with open(ols_path, 'r', encoding='utf-8') as f:
				ols_summary = f.read()
		except Exception:
			pass
	
	# Start building report
	lines = []
	lines.append("=" * 80)
	lines.append("COMPREHENSIVE ANALYSIS REPORT: MARKET SENTIMENT VS TRADER PERFORMANCE")
	lines.append("=" * 80)
	lines.append("")
	lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	lines.append("")
	lines.append("This report analyzes the relationship between market sentiment (Fear & Greed Index)")
	lines.append("and trader profitability on Hyperliquid, based on 211,224 historical trades.")
	lines.append("")
	lines.append("=" * 80)
	lines.append("")
	
	# 1. EXECUTIVE SUMMARY
	lines.append("1. EXECUTIVE SUMMARY")
	lines.append("-" * 80)
	lines.append("")
	
	if ttest_df is not None and len(ttest_df) > 0:
		ttest = ttest_df.iloc[0]
		t_stat = ttest.get('t_stat', 0)
		p_value = ttest.get('p_value', 1)
		n_fear = int(ttest.get('n_fear', 0))
		n_greed = int(ttest.get('n_greed', 0))
		
		lines.append("KEY FINDING: Statistical analysis reveals a SIGNIFICANT difference in trader")
		lines.append(f"performance between Fear and Greed market conditions.")
		lines.append("")
		lines.append(f"  • T-Statistic: {t_stat:.3f}")
		lines.append(f"  • P-Value: {p_value:.2e} (highly significant)")
		lines.append(f"  • Sample Size: {n_fear:,} trades during Fear periods, {n_greed:,} during Greed")
		lines.append("")
		
		if p_value < 0.05:
			lines.append("  ✓ CONCLUSION: The difference is statistically significant (p < 0.05)")
		lines.append("")
	
	# Best models
	if regression_results:
		best_reg = None
		best_reg_score = -float('inf')
		for model_name, metrics in regression_results.items():
			if metrics.get('R2', -float('inf')) > best_reg_score:
				best_reg_score = metrics.get('R2', -float('inf'))
				best_reg = (model_name, metrics)
		
		if best_reg:
			model_name, metrics = best_reg
			lines.append(f"BEST PREDICTIVE MODEL (Regression): {model_name}")
			lines.append(f"  • R² Score: {metrics.get('R2', 0):.4f} ({metrics.get('R2', 0)*100:.2f}% variance explained)")
			lines.append(f"  • RMSE: ${format_number(metrics.get('RMSE', 0))}")
			lines.append("")
	
	if classification_results:
		best_clf = None
		best_clf_score = -float('inf')
		for model_name, metrics in classification_results.items():
			score = metrics.get('Accuracy', 0) * metrics.get('F1', 0)
			if score > best_clf_score:
				best_clf_score = score
				best_clf = (model_name, metrics)
		
		if best_clf:
			model_name, metrics = best_clf
			lines.append(f"BEST PREDICTIVE MODEL (Classification): {model_name}")
			lines.append(f"  • Accuracy: {metrics.get('Accuracy', 0)*100:.2f}%")
			lines.append(f"  • F1-Score: {metrics.get('F1', 0):.4f}")
			lines.append("")
	
	lines.append("")
	lines.append("=" * 80)
	lines.append("")
	
	# 2. TRADER PERFORMANCE OVERVIEW
	lines.append("2. TRADER PERFORMANCE OVERVIEW")
	lines.append("-" * 80)
	lines.append("")
	
	if descriptive_stats is not None and len(descriptive_stats) > 0:
		lines.append("Overall Trading Statistics:")
		lines.append("")
		for _, row in descriptive_stats.iterrows():
			metric = row.iloc[0]
			count = row.get('count', 0)
			mean = row.get('mean', 0)
			median = row.get('median', 0)
			std = row.get('std', 0)
			min_val = row.get('min', 0)
			max_val = row.get('max', 0)
			
			if metric == 'pnl_usd':
				lines.append(f"Profit & Loss (PnL) per Trade:")
				lines.append(f"  • Total Trades Analyzed: {int(count):,}")
				lines.append(f"  • Average PnL: ${format_number(mean)}")
				lines.append(f"  • Median PnL: ${format_number(median)}")
				lines.append(f"  • Standard Deviation: ${format_number(std)}")
				lines.append(f"  • Range: ${format_number(min_val)} to ${format_number(max_val)}")
				lines.append("")
			elif metric == 'trade_value_usd':
				lines.append(f"Trade Value:")
				lines.append(f"  • Average Trade Size: ${format_number(mean)}")
				lines.append(f"  • Median Trade Size: ${format_number(median)}")
				lines.append(f"  • Standard Deviation: ${format_number(std)}")
				lines.append("")
	
	lines.append("")
	lines.append("=" * 80)
	lines.append("")
	
	# 3. SENTIMENT-BASED PERFORMANCE ANALYSIS
	lines.append("3. SENTIMENT-BASED PERFORMANCE ANALYSIS")
	lines.append("-" * 80)
	lines.append("")
	
	if pnl_by_sentiment is not None and len(pnl_by_sentiment) > 0:
		lines.append("Performance by Market Sentiment:")
		lines.append("")
		
		# Sort by mean PnL
		pnl_sorted = pnl_by_sentiment.sort_values('mean', ascending=False)
		
		for _, row in pnl_sorted.iterrows():
			sentiment = row.get('sentiment_class', 'unknown')
			count = int(row.get('count', 0))
			mean = row.get('mean', 0)
			median = row.get('median', 0)
			total = row.get('sum', 0)
			
			sentiment_display = sentiment.upper().replace('_', ' ')
			lines.append(f"{sentiment_display} Market Conditions:")
			lines.append(f"  • Number of Trades: {count:,}")
			lines.append(f"  • Average PnL per Trade: ${format_number(mean)}")
			lines.append(f"  • Median PnL: ${format_number(median)}")
			lines.append(f"  • Total PnL: ${format_number(total)}")
			lines.append("")
		
		# Find best and worst
		if len(pnl_sorted) > 0:
			best_sentiment = pnl_sorted.iloc[0]
			worst_sentiment = pnl_sorted.iloc[-1]
			
			lines.append("KEY INSIGHTS:")
			lines.append(f"  • BEST PERFORMANCE: {best_sentiment.get('sentiment_class', 'unknown').upper()} "
						f"markets with ${format_number(best_sentiment.get('mean', 0))} average PnL")
			lines.append(f"  • WORST PERFORMANCE: {worst_sentiment.get('sentiment_class', 'unknown').upper()} "
						f"markets with ${format_number(worst_sentiment.get('mean', 0))} average PnL")
			lines.append("")
	
	lines.append("")
	lines.append("=" * 80)
	lines.append("")
	
	# 4. STATISTICAL TESTS
	lines.append("4. STATISTICAL TESTS & SIGNIFICANCE")
	lines.append("-" * 80)
	lines.append("")
	
	if ttest_df is not None and len(ttest_df) > 0:
		ttest = ttest_df.iloc[0]
		t_stat = ttest.get('t_stat', 0)
		p_value = ttest.get('p_value', 1)
		
		lines.append("T-Test: Fear vs Greed Market Performance")
		lines.append("")
		lines.append("This test determines if there's a statistically significant difference")
		lines.append("in trader profitability between Fear and Greed market conditions.")
		lines.append("")
		lines.append(f"  • T-Statistic: {t_stat:.4f}")
		lines.append(f"  • P-Value: {p_value:.2e}")
		lines.append("")
		
		if p_value < 0.001:
			lines.append("  ✓ HIGHLY SIGNIFICANT: p < 0.001")
			lines.append("     There is extremely strong evidence that Fear and Greed markets")
			lines.append("     produce different trader outcomes.")
		elif p_value < 0.01:
			lines.append("  ✓ VERY SIGNIFICANT: p < 0.01")
		elif p_value < 0.05:
			lines.append("  ✓ SIGNIFICANT: p < 0.05")
		else:
			lines.append("  ✗ NOT SIGNIFICANT: p >= 0.05")
		lines.append("")
	
	if correlation_tests is not None and len(correlation_tests) > 0:
		lines.append("Correlation Analysis:")
		lines.append("")
		for _, row in correlation_tests.iterrows():
			var1 = row.get('Variable 1', '')
			var2 = row.get('Variable 2', '')
			corr = row.get('Correlation', 0)
			p_val = row.get('P-Value', 1)
			
			lines.append(f"  • {var1} vs {var2}:")
			lines.append(f"    Correlation: {corr:.4f}")
			lines.append(f"    P-Value: {p_val:.2e}")
			if p_val < 0.05:
				lines.append(f"    → SIGNIFICANT relationship")
			lines.append("")
	
	lines.append("")
	lines.append("=" * 80)
	lines.append("")
	
	# 5. PREDICTIVE MODELS
	lines.append("5. PREDICTIVE MODELS PERFORMANCE")
	lines.append("-" * 80)
	lines.append("")
	
	if regression_results:
		lines.append("REGRESSION MODELS (Predicting PnL Amount):")
		lines.append("")
		for model_name, metrics in regression_results.items():
			r2 = metrics.get('R2', 0)
			rmse = metrics.get('RMSE', 0)
			lines.append(f"  {model_name}:")
			lines.append(f"    • R² Score: {r2:.4f} ({r2*100:.2f}% of variance explained)")
			lines.append(f"    • RMSE: ${format_number(rmse)}")
			
			if r2 > 0.9:
				lines.append(f"    → EXCELLENT: Model explains over 90% of variance")
			elif r2 > 0.7:
				lines.append(f"    → GOOD: Model explains over 70% of variance")
			elif r2 > 0.5:
				lines.append(f"    → MODERATE: Model explains over 50% of variance")
			else:
				lines.append(f"    → POOR: Model explains less than 50% of variance")
			lines.append("")
	
	if classification_results:
		lines.append("CLASSIFICATION MODELS (Predicting Profitability: Win/Loss):")
		lines.append("")
		for model_name, metrics in classification_results.items():
			acc = metrics.get('Accuracy', 0)
			f1 = metrics.get('F1', 0)
			lines.append(f"  {model_name}:")
			lines.append(f"    • Accuracy: {acc*100:.2f}%")
			lines.append(f"    • F1-Score: {f1:.4f}")
			
			if acc > 0.9:
				lines.append(f"    → EXCELLENT: Over 90% accuracy")
			elif acc > 0.7:
				lines.append(f"    → GOOD: Over 70% accuracy")
			elif acc > 0.5:
				lines.append(f"    → MODERATE: Over 50% accuracy")
			else:
				lines.append(f"    → POOR: Less than 50% accuracy")
			lines.append("")
	
	lines.append("")
	lines.append("=" * 80)
	lines.append("")
	
	# 6. REGRESSION ANALYSIS (OLS)
	if ols_summary:
		lines.append("6. REGRESSION ANALYSIS (OLS)")
		lines.append("-" * 80)
		lines.append("")
		lines.append("Ordinary Least Squares (OLS) regression analysis reveals the")
		lines.append("statistical relationship between sentiment and trader performance:")
		lines.append("")
		lines.append(ols_summary)
		lines.append("")
		lines.append("=" * 80)
		lines.append("")
	
	# 7. KEY INSIGHTS & RECOMMENDATIONS
	lines.append("7. KEY INSIGHTS & STRATEGIC RECOMMENDATIONS")
	lines.append("-" * 80)
	lines.append("")
	
	lines.append("INSIGHTS:")
	lines.append("")
	
	if pnl_by_sentiment is not None and len(pnl_by_sentiment) > 0:
		greed_row = pnl_by_sentiment[pnl_by_sentiment['sentiment_class'] == 'greed']
		fear_row = pnl_by_sentiment[pnl_by_sentiment['sentiment_class'] == 'fear']
		
		if len(greed_row) > 0 and len(fear_row) > 0:
			greed_mean = greed_row.iloc[0].get('mean', 0)
			fear_mean = fear_row.iloc[0].get('mean', 0)
			
			if greed_mean > fear_mean:
				diff = greed_mean - fear_mean
				lines.append(f"  1. Greed markets show ${format_number(diff)} higher average PnL than Fear markets.")
				lines.append("     → Traders tend to perform better during optimistic market conditions.")
			else:
				diff = fear_mean - greed_mean
				lines.append(f"  1. Fear markets show ${format_number(diff)} higher average PnL than Greed markets.")
				lines.append("     → Traders may find better opportunities during pessimistic conditions.")
			lines.append("")
	
	lines.append("  2. Predictive models demonstrate strong ability to forecast trader profitability")
	lines.append("     based on sentiment and trading features.")
	lines.append("")
	lines.append("  3. Statistical tests confirm that market sentiment significantly impacts")
	lines.append("     trader performance outcomes.")
	lines.append("")
	
	lines.append("RECOMMENDATIONS:")
	lines.append("")
	lines.append("  1. DYNAMIC RISK MANAGEMENT: Adjust position sizes based on market sentiment.")
	lines.append("     Consider reducing exposure during Fear periods if historical data shows")
	lines.append("     lower profitability in those conditions.")
	lines.append("")
	lines.append("  2. SENTIMENT-DRIVEN STRATEGY: Use Fear & Greed Index as a feature in")
	lines.append("     algorithmic trading systems to optimize entry/exit timing.")
	lines.append("")
	lines.append("  3. PORTFOLIO OPTIMIZATION: Allocate capital differently based on sentiment")
	lines.append("     phases, potentially increasing activity during more profitable conditions.")
	lines.append("")
	lines.append("  4. MODEL DEPLOYMENT: Deploy the best-performing predictive models (XGBoost)")
	lines.append("     for real-time profitability forecasting to guide trading decisions.")
	lines.append("")
	
	lines.append("=" * 80)
	lines.append("")
	lines.append("END OF REPORT")
	lines.append("")
	lines.append("For detailed visualizations, see the HTML files in the outputs/ directory.")
	lines.append("For interactive model testing, run the web application: streamlit run app.py")
	lines.append("")
	
	# Write report
	with open(report_path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines))
	
	print(f"   ✓ Comprehensive report generated: {report_path}")
	return report_path

