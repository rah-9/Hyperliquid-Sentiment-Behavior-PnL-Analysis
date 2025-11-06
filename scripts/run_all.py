import os
import sys
import json
import pandas as pd

# Add project root to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_prep import prepare_datasets
from src.features import rolling_features, trader_performance_features, build_feature_matrix
from src.eda import descriptive_stats, pnl_by_sentiment, direction_frequency, plot_pnl_distribution, plot_time_series_agg, correlation_heatmap
from src.stats_tests import ttest_pnl_fear_vs_greed, correlation_tests, ols_sentiment_on_pnl
from src.modeling import train_regressors, train_classifiers
from src import viz as viz_helpers


DATA_FG = os.path.join(ROOT, 'fear_greed_index.csv')
DATA_TRADES = os.path.join(ROOT, 'historical_data.csv')
OUTPUTS = os.path.join(ROOT, 'outputs')
REPORTS = os.path.join(ROOT, 'reports')

os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)


def main():
	print("=" * 60)
	print("Hyperliquid Sentiment Analysis Pipeline")
	print("=" * 60)
	
	try:
		print("\n[1/5] Loading and preparing data...")
		sentiment_daily, trades_raw, trades = prepare_datasets(DATA_FG, DATA_TRADES)
		print(f"   Loaded {len(trades)} trades")
		
		# Diagnostic: Check Account column
		if 'Account' in trades.columns:
			account_count = trades['Account'].notna().sum()
			print(f"   Account column found: {account_count}/{len(trades)} non-null values")
		else:
			print(f"   Warning: Account column missing after load. Available columns: {list(trades.columns)[:15]}")
		
		trades = rolling_features(trades)
		if 'Account' not in trades.columns:
			print(f"   Warning: Account column lost after rolling_features")
		
		trades = trader_performance_features(trades)
		if 'Account' not in trades.columns:
			print(f"   Warning: Account column lost after trader_performance_features")
		else:
			print(f"   Account column preserved: {trades['Account'].notna().sum()}/{len(trades)} non-null")
		
		print("   ✓ Data preparation complete")
	except Exception as e:
		print(f"   ✗ Error in data preparation: {e}")
		raise

	try:
		print("\n[2/5] Running EDA and generating visualizations...")
		desc = descriptive_stats(trades)
		desc.to_csv(os.path.join(OUTPUTS, 'descriptive_stats.csv'))
		pnl_sent = pnl_by_sentiment(trades)
		pnl_sent.to_csv(os.path.join(OUTPUTS, 'pnl_by_sentiment.csv'))
		dir_freq = direction_frequency(trades)
		dir_freq.to_csv(os.path.join(OUTPUTS, 'direction_frequency.csv'))
		plot_pnl_distribution(trades, out_png=os.path.join(OUTPUTS, 'pnl_distribution.png'))
		plot_time_series_agg(trades, out_html=os.path.join(OUTPUTS, 'daily_timeseries.html'))
		corr = correlation_heatmap(trades, out_png=os.path.join(OUTPUTS, 'correlation_heatmap.png'))
		if len(corr) > 0:
			corr.to_csv(os.path.join(OUTPUTS, 'correlation_matrix.csv'))
		else:
			print("   (Skipped correlation matrix CSV - insufficient data)")
		viz_helpers.plot_sentiment_vs_total_pnl(trades, out_html=os.path.join(OUTPUTS, 'sentiment_vs_pnl.html'))
		result = viz_helpers.plot_top_traders_by_sentiment(trades, out_html=os.path.join(OUTPUTS, 'top_traders_by_sentiment.html'))
		if result is None:
			print("   (Skipped top traders plot - Account column not available)")
		viz_helpers.plot_volume_vs_sentiment(trades, out_html=os.path.join(OUTPUTS, 'volume_vs_sentiment.html'))
		print("   ✓ EDA complete")
	except Exception as e:
		print(f"   ✗ Error in EDA: {e}")
		desc = None
		pnl_sent = None

	try:
		print("\n[3/5] Running statistical tests...")
		ttest = ttest_pnl_fear_vs_greed(trades)
		pd.DataFrame([ttest]).to_csv(os.path.join(OUTPUTS, 'ttest_fear_vs_greed.csv'), index=False)
		corr_tests = correlation_tests(trades)
		if len(corr_tests) > 0:
			corr_tests.to_csv(os.path.join(OUTPUTS, 'correlation_tests.csv'), index=False)
		else:
			print("   (Skipped correlation tests - insufficient data)")
		
		# Diagnostic: Check data availability before OLS
		pnl_count = trades['pnl_usd'].notna().sum() if 'pnl_usd' in trades.columns else 0
		sentiment_count = trades['sentiment_score'].notna().sum() if 'sentiment_score' in trades.columns else 0
		print(f"   Data check: pnl_usd valid={pnl_count}/{len(trades)}, sentiment_score valid={sentiment_count}/{len(trades)}")
		
		try:
			ols = ols_sentiment_on_pnl(trades)
			with open(os.path.join(OUTPUTS, 'ols_summary.txt'), 'w') as f:
				f.write(ols.summary().as_text())
		except ValueError as ve:
			print(f"   (Skipped OLS regression: {ve})")
			ols = None
		except Exception as e:
			print(f"   (Skipped OLS regression: {e})")
			ols = None
		
		print("   ✓ Statistical tests complete")
	except Exception as e:
		print(f"   ✗ Error in statistical tests: {e}")
		ttest = {}
		ols = None

	try:
		print("\n[4/5] Training predictive models...")
		X, (y_reg, y_clf), feature_cols = build_feature_matrix(trades)
		print(f"   Features: {len(feature_cols)}")
		reg_results = train_regressors(X, y_reg, out_dir=os.path.join(OUTPUTS, 'regression'))
		clf_results = train_classifiers(X, y_clf, out_dir=os.path.join(OUTPUTS, 'classification'))
		with open(os.path.join(OUTPUTS, 'features.json'), 'w') as f:
			json.dump({'features': feature_cols}, f, indent=2)
		print("   ✓ Modeling complete")
	except Exception as e:
		print(f"   ✗ Error in modeling: {e}")
		reg_results = {}
		clf_results = {}

	try:
		print("\n[5/6] Generating executive summary...")
		ols_text = ols.summary().as_text() if ols else "OLS model not available"
		build_executive_summary(REPORTS, desc, pnl_sent, reg_results, clf_results, ttest, ols_text)
		print("   ✓ Executive summary generated")
	except Exception as e:
		print(f"   ✗ Error generating summary: {e}")

	try:
		print("\n[6/6] Generating comprehensive human-readable report...")
		from src.report_generator import generate_comprehensive_report
		report_path = generate_comprehensive_report(outputs_dir=OUTPUTS, reports_dir=REPORTS)
		print(f"   ✓ Comprehensive report generated: {report_path}")
	except Exception as e:
		print(f"   ✗ Error generating comprehensive report: {e}")

	print("\n" + "=" * 60)
	print("Pipeline complete! Check outputs/ and reports/ directories.")
	print("=" * 60)
	print("\nTo launch the web application, run:")
	print("  streamlit run app.py")
	print("=" * 60)



def build_executive_summary(reports_dir: str,
							desc: pd.DataFrame,
							pnl_sent: pd.DataFrame,
							reg: dict,
							clf: dict,
							ttest: dict,
							ols_text: str):
	path = os.path.join(reports_dir, 'executive_summary.md')
	lines = []
	lines.append('# Executive Summary: Sentiment vs Hyperliquid Trader Performance')
	lines.append('')
	lines.append('## Key Findings')
	
	if ttest and 't_stat' in ttest and not pd.isna(ttest.get('t_stat')):
		lines.append(f"- T-Test Fear vs Greed PnL: t={ttest.get('t_stat'):.3f}, p={ttest.get('p_value'):.3f} (n_fear={ttest.get('n_fear')}, n_greed={ttest.get('n_greed')})")
	else:
		lines.append("- T-Test Fear vs Greed PnL: Insufficient data")
	
	if reg and 'XGBRegressor' in reg:
		lines.append(f"- Best Regression (XGB): R2={reg['XGBRegressor']['R2']:.3f}, RMSE={reg['XGBRegressor']['RMSE']:.3f}")
	if clf and 'XGBClassifier' in clf:
		lines.append(f"- Best Classifier (XGB): Acc={clf['XGBClassifier']['Accuracy']:.3f}, F1={clf['XGBClassifier']['F1']:.3f}")
	lines.append('')
	
	if desc is not None:
		lines.append('## Descriptive Highlights')
		try:
			lines.append(desc.to_markdown())
		except Exception:
			# Fallback to CSV format if tabulate not available
			lines.append(desc.to_string())
		lines.append('')
	
	if pnl_sent is not None:
		lines.append('## PnL by Sentiment')
		try:
			lines.append(pnl_sent.to_markdown())
		except Exception:
			# Fallback to CSV format if tabulate not available
			lines.append(pnl_sent.to_string())
		lines.append('')
	
	lines.append('## OLS Regression Summary')
	lines.append('```')
	lines.append(ols_text)
	lines.append('```')
	
	with open(path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines))


if __name__ == '__main__':
	main()
