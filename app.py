"""
Streamlit Web Application for Hyperliquid Sentiment Analysis
Interactive interface to test models and explore results.
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
from pathlib import Path

# Try importing optional dependencies
try:
	import plotly.express as px
	PLOTLY_AVAILABLE = True
except ImportError:
	PLOTLY_AVAILABLE = False
	st.sidebar.warning("⚠️ Plotly not installed. Charts will be disabled. Run: pip install plotly")

try:
	import xgboost
	XGBOOST_AVAILABLE = True
except ImportError:
	XGBOOST_AVAILABLE = False
	st.sidebar.warning("⚠️ XGBoost not installed. XGBoost models won't load. Run: pip install xgboost")

# Page config
st.set_page_config(
	page_title="Hyperliquid Sentiment Analysis",
	page_icon="📊",
	layout="wide"
)

# Title
st.title("📊 Hyperliquid Sentiment Analysis Dashboard")
st.markdown("**Interactive tool to explore market sentiment effects on trader performance**")

# Dependency check banner
if not XGBOOST_AVAILABLE or not PLOTLY_AVAILABLE:
	missing = []
	if not XGBOOST_AVAILABLE:
		missing.append("xgboost")
	if not PLOTLY_AVAILABLE:
		missing.append("plotly")
	
	st.warning(f"⚠️ Missing dependencies: {', '.join(missing)}. Install with: `pip install {' '.join(missing)}` or `pip install -r requirements.txt`")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("**Choose a page:**")
# Show all pages as radio buttons so they're all visible
page = st.sidebar.radio(
	"Select Page",
	options=["🏠 Home", "🔮 Model Predictions", "📈 Data Explorer", "📊 Visualizations", "📄 Report Viewer"],
	label_visibility="collapsed"
)

# Load data function
@st.cache_data
def load_data():
	"""Load all analysis outputs."""
	data = {}
	
	# Load regression results
	reg_path = "outputs/regression/regression_results.json"
	if os.path.exists(reg_path):
		with open(reg_path, 'r') as f:
			data['regression'] = json.load(f)
	
	# Load classification results
	clf_path = "outputs/classification/classification_results.json"
	if os.path.exists(clf_path):
		with open(clf_path, 'r') as f:
			data['classification'] = json.load(f)
	
	# Load descriptive stats
	desc_path = "outputs/descriptive_stats.csv"
	if os.path.exists(desc_path):
		data['descriptive'] = pd.read_csv(desc_path)
	
	# Load PnL by sentiment
	pnl_path = "outputs/pnl_by_sentiment.csv"
	if os.path.exists(pnl_path):
		data['pnl_sentiment'] = pd.read_csv(pnl_path)
	
	# Load t-test results
	ttest_path = "outputs/ttest_fear_vs_greed.csv"
	if os.path.exists(ttest_path):
		data['ttest'] = pd.read_csv(ttest_path)
	
	# Load correlation matrix
	corr_path = "outputs/correlation_matrix.csv"
	if os.path.exists(corr_path):
		data['correlation'] = pd.read_csv(corr_path, index_col=0)
	
	# Load direction frequency
	dir_path = "outputs/direction_frequency.csv"
	if os.path.exists(dir_path):
		data['direction_freq'] = pd.read_csv(dir_path, index_col=0)
	
	# Load correlation tests
	corr_test_path = "outputs/correlation_tests.csv"
	if os.path.exists(corr_test_path):
		data['correlation_tests'] = pd.read_csv(corr_test_path)
	
	return data

# Load models function
@st.cache_resource
def load_models():
	"""Load trained models."""
	models = {}
	
	# Regression models
	reg_models = {
		'XGBRegressor': 'outputs/regression/XGBRegressor.joblib',
		'LinearRegression': 'outputs/regression/LinearRegression.joblib',
		'RidgeCV': 'outputs/regression/RidgeCV.joblib'
	}
	
	for name, path in reg_models.items():
		if os.path.exists(path):
			# Skip XGBoost models if xgboost is not available
			if 'XGB' in name and not XGBOOST_AVAILABLE:
				continue
			try:
				models[name] = joblib.load(path)
			except Exception as e:
				# Only show warning if it's not a missing dependency issue
				if 'xgboost' not in str(e).lower() or XGBOOST_AVAILABLE:
					st.sidebar.warning(f"Could not load {name}: {e}")
	
	# Classification models
	clf_models = {
		'XGBClassifier': 'outputs/classification/XGBClassifier.joblib',
		'RandomForest': 'outputs/classification/RandomForest.joblib',
		'LogisticRegression': 'outputs/classification/LogisticRegression.joblib'
	}
	
	for name, path in clf_models.items():
		if os.path.exists(path):
			# Skip XGBoost models if xgboost is not available
			if 'XGB' in name and not XGBOOST_AVAILABLE:
				continue
			try:
				models[name] = joblib.load(path)
			except Exception as e:
				# Only show warning if it's not a missing dependency issue
				if 'xgboost' not in str(e).lower() or XGBOOST_AVAILABLE:
					st.sidebar.warning(f"Could not load {name}: {e}")
	
	return models

# Home page
if page == "🏠 Home":
	st.header("Welcome to the Hyperliquid Sentiment Analysis Dashboard")
	
	data = load_data()
	
	# Key metrics
	col1, col2, col3, col4 = st.columns(4)
	
	if 'descriptive' in data and len(data['descriptive']) > 0:
		pnl_row = data['descriptive'][data['descriptive'].iloc[:, 0] == 'pnl_usd']
		if len(pnl_row) > 0:
			mean_pnl = pnl_row.iloc[0]['mean']
			col1.metric("Average PnL per Trade", f"${mean_pnl:.2f}")
	
	if 'pnl_sentiment' in data and len(data['pnl_sentiment']) > 0:
		greed_row = data['pnl_sentiment'][data['pnl_sentiment']['sentiment_class'] == 'greed']
		if len(greed_row) > 0:
			greed_pnl = greed_row.iloc[0]['mean']
			col2.metric("Greed Market Avg PnL", f"${greed_pnl:.2f}")
		
		fear_row = data['pnl_sentiment'][data['pnl_sentiment']['sentiment_class'] == 'fear']
		if len(fear_row) > 0:
			fear_pnl = fear_row.iloc[0]['mean']
			col3.metric("Fear Market Avg PnL", f"${fear_pnl:.2f}")
	
	if 'regression' in data and 'XGBRegressor' in data['regression']:
		r2 = data['regression']['XGBRegressor']['R2']
		col4.metric("Best Model R²", f"{r2:.4f}")
	
	st.markdown("---")
	
	# Summary
	st.subheader("Analysis Summary")
	
	if 'ttest' in data and len(data['ttest']) > 0:
		ttest = data['ttest'].iloc[0]
		st.info(f"""
		**Statistical Test Results:**
		- T-Statistic: {ttest.get('t_stat', 0):.4f}
		- P-Value: {ttest.get('p_value', 0):.2e}
		- Sample Size: {int(ttest.get('n_fear', 0)):,} Fear trades, {int(ttest.get('n_greed', 0)):,} Greed trades
		""")
	
	# Model performance
	if 'regression' in data:
		st.subheader("Regression Models Performance")
		reg_df = pd.DataFrame(data['regression']).T
		st.dataframe(reg_df, width='stretch')
	
	if 'classification' in data:
		st.subheader("Classification Models Performance")
		clf_df = pd.DataFrame(data['classification']).T
		st.dataframe(clf_df, width='stretch')

# Model Predictions page
elif page == "🔮 Model Predictions":
	st.header("Make Predictions with Trained Models")
	
	models = load_models()
	
	if not models:
		st.error("⚠️ No trained models found or available.")
		st.info("""
		**To fix this:**
		1. Run the training pipeline: `python scripts/run_all.py`
		2. Install missing dependencies: `pip install -r requirements.txt`
		3. If XGBoost models are missing, install: `pip install xgboost`
		""")
		st.stop()
	
	# Show available models
	available_reg = [k for k in models.keys() if 'Regressor' in k]
	available_clf = [k for k in models.keys() if 'Classifier' in k]
	
	if not available_reg and not available_clf:
		st.warning("⚠️ No models are currently loaded. Check the sidebar for dependency warnings.")
	else:
		st.success(f"✓ Loaded {len(available_reg)} regression and {len(available_clf)} classification models")
	
	# Input form
	st.subheader("Input Trade Features")
	
	col1, col2 = st.columns(2)
	
	with col1:
		sentiment_score = st.slider("Sentiment Score (0-1)", 0.0, 1.0, 0.5, 0.01)
		trade_value_usd = st.number_input("Trade Value (USD)", min_value=0.0, value=1000.0, step=100.0)
		execution_price = st.number_input("Execution Price", min_value=0.0, value=100.0, step=1.0)
	
	with col2:
		sentiment_class = st.selectbox("Sentiment Class", ["fear", "greed", "neutral", "extreme_fear", "extreme_greed"])
		side = st.selectbox("Trade Side", ["long", "short"])
	
	# Create feature vector (simplified - in production, use the same preprocessing)
	st.markdown("---")
	
	# Regression predictions
	if any('Regressor' in name for name in models.keys()):
		st.subheader("Regression Predictions (PnL Amount)")
		
		reg_cols = st.columns(len([k for k in models.keys() if 'Regressor' in k]))
		
		idx = 0
		for name, model in models.items():
			if 'Regressor' in name:
				with reg_cols[idx]:
					try:
						# Simplified feature vector - in production, use proper feature engineering
						# This is a placeholder - actual features depend on the trained model
						features = np.array([[sentiment_score, trade_value_usd, execution_price, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
						prediction = model.predict(features)[0]
						st.metric(name, f"${prediction:.2f}")
					except Exception as e:
						st.error(f"Error: {e}")
				idx += 1
	
	# Classification predictions
	if any('Classifier' in name for name in models.keys()):
		st.subheader("Classification Predictions (Win/Loss)")
		
		clf_cols = st.columns(len([k for k in models.keys() if 'Classifier' in k]))
		
		idx = 0
		for name, model in models.items():
			if 'Classifier' in name:
				with clf_cols[idx]:
					try:
						# Simplified feature vector
						features = np.array([[sentiment_score, trade_value_usd, execution_price, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
						prediction = model.predict(features)[0]
						proba = model.predict_proba(features)[0]
						st.metric(name, "Win" if prediction == 1 else "Loss")
						# Convert to float to avoid float32 type error
						win_prob = float(proba[1]) if len(proba) > 1 else 0.5
						st.progress(win_prob)
						st.caption(f"Confidence: {win_prob*100:.1f}%")
					except Exception as e:
						st.error(f"Error: {e}")
				idx += 1
	
	st.info("⚠️ Note: These predictions use simplified features. For accurate predictions, use the full feature engineering pipeline.")

# Data Explorer page
elif page == "📈 Data Explorer":
	st.header("Explore Analysis Results")
	
	data = load_data()
	
	# Create tabs for different sections
	tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "💰 PnL Analysis", "📈 Correlations", "📉 Distributions", "📋 Statistics"])
	
	with tab1:
		st.subheader("Descriptive Statistics")
		if 'descriptive' in data:
			st.dataframe(data['descriptive'], width='stretch')
			
			# Visualize descriptive stats
			if PLOTLY_AVAILABLE and len(data['descriptive']) > 0:
				desc_df = data['descriptive'].T
				desc_df.columns = desc_df.iloc[0]
				desc_df = desc_df[1:]
				
				# Mean values chart
				if 'mean' in desc_df.columns:
					fig_mean = px.bar(
						x=desc_df.index,
						y=desc_df['mean'],
						title='Mean Values by Metric',
						labels={'x': 'Metric', 'y': 'Mean Value'}
					)
					st.plotly_chart(fig_mean, width='stretch')
		else:
			st.info("Run the analysis pipeline to generate descriptive statistics.")
	
	with tab2:
		st.subheader("PnL Analysis by Sentiment")
		if 'pnl_sentiment' in data:
			col1, col2 = st.columns(2)
			
			with col1:
				st.dataframe(data['pnl_sentiment'], width='stretch')
			
			with col2:
				if PLOTLY_AVAILABLE:
					# Bar chart
					fig_bar = px.bar(
						data['pnl_sentiment'],
						x='sentiment_class',
						y='mean',
						title='Average PnL by Sentiment Class',
						labels={'mean': 'Average PnL (USD)', 'sentiment_class': 'Sentiment Class'},
						color='mean',
						color_continuous_scale='RdYlGn'
					)
					st.plotly_chart(fig_bar, width='stretch')
					
					# Pie chart for total PnL
					fig_pie = px.pie(
						data['pnl_sentiment'],
						values='sum',
						names='sentiment_class',
						title='Total PnL Distribution by Sentiment'
					)
					st.plotly_chart(fig_pie, width='stretch')
				else:
					st.info("📊 Install plotly to view interactive charts: `pip install plotly`")
		else:
			st.info("Run the analysis pipeline to generate PnL sentiment analysis.")
		
		# Show saved PnL distribution image
		if os.path.exists("outputs/pnl_distribution.png"):
			st.subheader("PnL Distribution by Sentiment")
			st.image("outputs/pnl_distribution.png", caption="Probability Density of PnL by Sentiment Class")
	
	with tab3:
		st.subheader("Correlation Analysis")
		
		# Correlation matrix
		if 'correlation' in data and len(data['correlation']) > 0:
			col1, col2 = st.columns(2)
			
			with col1:
				st.write("**Correlation Matrix**")
				st.dataframe(data['correlation'], width='stretch')
			
			with col2:
				if PLOTLY_AVAILABLE:
					# Create interactive heatmap
					fig_heatmap = px.imshow(
						data['correlation'],
						text_auto=".2f",
						aspect="auto",
						title="Correlation Heatmap",
						color_continuous_scale="RdBu",
						color_continuous_midpoint=0
					)
					st.plotly_chart(fig_heatmap, width='stretch')
			
			# Show saved correlation heatmap image
			if os.path.exists("outputs/correlation_heatmap.png"):
				st.image("outputs/correlation_heatmap.png", caption="Correlation Heatmap")
		else:
			st.info("Run the analysis pipeline to generate correlation analysis.")
		
		# Correlation tests
		if 'correlation_tests' in data and len(data['correlation_tests']) > 0:
			st.subheader("Correlation Test Results")
			st.dataframe(data['correlation_tests'], width='stretch')
	
	with tab4:
		st.subheader("Trade Direction Frequency")
		if 'direction_freq' in data and len(data['direction_freq']) > 0:
			col1, col2 = st.columns(2)
			
			with col1:
				st.dataframe(data['direction_freq'], width='stretch')
			
			with col2:
				if PLOTLY_AVAILABLE:
					# Stacked bar chart
					fig_dir = px.bar(
						data['direction_freq'].reset_index(),
						x='sentiment_class',
						y=data['direction_freq'].columns.tolist(),
						title='Trade Direction Frequency by Sentiment',
						barmode='stack',
						labels={'value': 'Number of Trades', 'sentiment_class': 'Sentiment Class'}
					)
					st.plotly_chart(fig_dir, width='stretch')
		else:
			st.info("Run the analysis pipeline to generate direction frequency analysis.")
	
	with tab5:
		st.subheader("Statistical Test Results")
		if 'ttest' in data:
			st.dataframe(data['ttest'], width='stretch')
			
			if len(data['ttest']) > 0:
				ttest = data['ttest'].iloc[0]
				st.markdown("### T-Test: Fear vs Greed")
				st.metric("T-Statistic", f"{ttest.get('t_stat', 0):.4f}")
				st.metric("P-Value", f"{ttest.get('p_value', 0):.2e}")
				st.metric("Fear Sample Size", f"{int(ttest.get('n_fear', 0)):,}")
				st.metric("Greed Sample Size", f"{int(ttest.get('n_greed', 0)):,}")
				
				# Significance indicator
				p_val = ttest.get('p_value', 1)
				if p_val < 0.001:
					st.success("✅ **Highly Significant** (p < 0.001)")
				elif p_val < 0.01:
					st.success("✅ **Very Significant** (p < 0.01)")
				elif p_val < 0.05:
					st.success("✅ **Significant** (p < 0.05)")
				else:
					st.warning("⚠️ **Not Significant** (p >= 0.05)")
		else:
			st.info("Run the analysis pipeline to generate statistical tests.")

# Visualizations page
elif page == "📊 Visualizations":
	st.header("Interactive Visualizations")
	
	# Create tabs for different visualization categories
	tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "💰 Performance", "👥 Top Traders", "📊 All Charts"])
	
	with tab1:
		st.subheader("Time Series Visualizations")
		
		# Daily Time Series
		if os.path.exists("outputs/daily_timeseries.html"):
			st.markdown("### Daily Aggregates: PnL, Volume, Sentiment")
			with open("outputs/daily_timeseries.html", 'r', encoding='utf-8') as f:
				html_content = f.read()
			st.components.v1.html(html_content, height=600, scrolling=True)
		else:
			st.info("Daily time series visualization will be generated after running the analysis pipeline.")
		
		# Sentiment vs PnL
		if os.path.exists("outputs/sentiment_vs_pnl.html"):
			st.markdown("### Sentiment vs Total PnL")
			with open("outputs/sentiment_vs_pnl.html", 'r', encoding='utf-8') as f:
				html_content = f.read()
			st.components.v1.html(html_content, height=600, scrolling=True)
		else:
			st.info("Sentiment vs PnL visualization will be generated after running the analysis pipeline.")
	
	with tab2:
		st.subheader("Performance Visualizations")
		
		# Volume vs Sentiment
		if os.path.exists("outputs/volume_vs_sentiment.html"):
			st.markdown("### Trade Volume vs Sentiment")
			with open("outputs/volume_vs_sentiment.html", 'r', encoding='utf-8') as f:
				html_content = f.read()
			st.components.v1.html(html_content, height=600, scrolling=True)
		else:
			st.info("Volume vs sentiment visualization will be generated after running the analysis pipeline.")
		
		# Model Performance Comparison
		data = load_data()
		if 'regression' in data and PLOTLY_AVAILABLE:
			st.markdown("### Model Performance Comparison")
			
			# Regression models
			reg_df = pd.DataFrame(data['regression']).T
			if len(reg_df) > 0:
				fig_reg = px.bar(
					reg_df.reset_index(),
					x='index',
					y='R2',
					title='Regression Models: R² Score',
					labels={'index': 'Model', 'R2': 'R² Score'},
					color='R2',
					color_continuous_scale='Viridis'
				)
				st.plotly_chart(fig_reg, width='stretch')
			
			# Classification models
			if 'classification' in data:
				clf_df = pd.DataFrame(data['classification']).T
				if len(clf_df) > 0:
					fig_clf = px.bar(
						clf_df.reset_index(),
						x='index',
						y='Accuracy',
						title='Classification Models: Accuracy',
						labels={'index': 'Model', 'Accuracy': 'Accuracy'},
						color='Accuracy',
						color_continuous_scale='Plasma'
					)
					st.plotly_chart(fig_clf, width='stretch')
	
	with tab3:
		st.subheader("Top Traders Analysis")
		
		# Top Traders by Sentiment
		if os.path.exists("outputs/top_traders_by_sentiment.html"):
			st.markdown("### Top 10 Traders by Sentiment Phase")
			with open("outputs/top_traders_by_sentiment.html", 'r', encoding='utf-8') as f:
				html_content = f.read()
			st.components.v1.html(html_content, height=600, scrolling=True)
		else:
			st.info("Top traders visualization will be generated after running the analysis pipeline.")
	
	with tab4:
		st.subheader("All Available Visualizations")
		
		viz_files = {
			"📈 Daily Time Series": "outputs/daily_timeseries.html",
			"💰 Sentiment vs PnL": "outputs/sentiment_vs_pnl.html",
			"📊 Volume vs Sentiment": "outputs/volume_vs_sentiment.html",
			"👥 Top Traders by Sentiment": "outputs/top_traders_by_sentiment.html"
		}
		
		selected_viz = st.radio(
			"Select a visualization to view:",
			options=list(viz_files.keys()),
			horizontal=False
		)
		
		viz_path = viz_files[selected_viz]
		if os.path.exists(viz_path):
			with open(viz_path, 'r', encoding='utf-8') as f:
				html_content = f.read()
			st.components.v1.html(html_content, height=700, scrolling=True)
		else:
			st.warning(f"Visualization file not found: {viz_path}")
			st.info("Run the analysis pipeline to generate all visualizations.")
		
		# Show static images
		st.markdown("---")
		st.subheader("Static Charts")
		
		col1, col2 = st.columns(2)
		
		with col1:
			if os.path.exists("outputs/pnl_distribution.png"):
				st.image("outputs/pnl_distribution.png", caption="PnL Distribution by Sentiment")
		
		with col2:
			if os.path.exists("outputs/correlation_heatmap.png"):
				st.image("outputs/correlation_heatmap.png", caption="Correlation Heatmap")

# Report Viewer page
elif page == "📄 Report Viewer":
	st.header("Comprehensive Analysis Report")
	
	report_path = "reports/comprehensive_analysis_report.txt"
	
	if os.path.exists(report_path):
		with open(report_path, 'r', encoding='utf-8') as f:
			report_content = f.read()
		
		st.text_area("Report Content", report_content, height=600)
		
		# Download button
		st.download_button(
			label="Download Report",
			data=report_content,
			file_name="comprehensive_analysis_report.txt",
			mime="text/plain"
		)
	else:
		st.warning("Report not found. Please run the analysis pipeline to generate the report.")
		if st.button("Generate Report Now"):
			try:
				from src.report_generator import generate_comprehensive_report
				report_path = generate_comprehensive_report()
				st.success(f"Report generated: {report_path}")
				st.rerun()
			except Exception as e:
				st.error(f"Error generating report: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Hyperliquid Sentiment Analysis**")
st.sidebar.markdown("Built with Streamlit")

