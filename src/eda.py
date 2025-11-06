import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
	cols = [c for c in ['pnl_usd', 'trade_value_usd', 'Execution Price'] if c in df.columns]
	desc = df[cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
	return desc


def pnl_by_sentiment(df: pd.DataFrame) -> pd.DataFrame:
	grouped = df.groupby('sentiment_class', dropna=False)['pnl_usd'].agg(['count', 'mean', 'median', 'std', 'sum']).sort_values('mean')
	return grouped


def direction_frequency(df: pd.DataFrame) -> pd.DataFrame:
	return df.groupby(['sentiment_class', 'side_norm'], observed=True).size().unstack(fill_value=0)


def plot_pnl_distribution(df: pd.DataFrame, out_png: str = None):
	# Check if we have the required columns and data
	if 'pnl_usd' not in df.columns or 'sentiment_class' not in df.columns:
		print("   Warning: Missing required columns for PnL distribution plot")
		return
	
	# Filter out NaN values
	plot_df = df[['pnl_usd', 'sentiment_class']].dropna()
	if len(plot_df) == 0:
		print("   Warning: No valid data for PnL distribution plot")
		return
	
	plt.figure(figsize=(9, 5))
	try:
		sns.kdeplot(data=plot_df, x='pnl_usd', hue='sentiment_class', common_norm=False)
		plt.title('PnL Distribution by Sentiment')
		plt.tight_layout()
		if out_png:
			plt.savefig(out_png, dpi=160)
	except Exception as e:
		print(f"   Warning: Could not create PnL distribution plot: {e}")
	finally:
		plt.close()


def plot_time_series_agg(df: pd.DataFrame, out_html: str = None):
	agg = df.set_index('timestamp_utc').sort_index().resample('1D').agg({
		'pnl_usd': 'sum',
		'trade_value_usd': 'sum',
		'sentiment_score': 'mean'
	}).reset_index()
	fig = px.line(agg, x='timestamp_utc', y=['pnl_usd', 'trade_value_usd', 'sentiment_score'],
				 title='Daily Aggregates: PnL, Volume, Sentiment')
	if out_html:
		fig.write_html(out_html)
	return fig


def correlation_heatmap(df: pd.DataFrame, out_png: str = None) -> pd.DataFrame:
	cols = [c for c in ['pnl_usd', 'trade_value_usd', 'Execution Price', 'sentiment_score'] if c in df.columns]
	if len(cols) < 2:
		print("   Warning: Insufficient columns for correlation heatmap")
		return pd.DataFrame()
	
	# Filter to numeric columns with valid data
	corr_df = df[cols].select_dtypes(include=[np.number])
	if len(corr_df.columns) < 2:
		print("   Warning: Insufficient numeric columns for correlation heatmap")
		return pd.DataFrame()
	
	try:
		corr = corr_df.corr(method='spearman')
		if corr.isna().all().all():
			print("   Warning: All correlations are NaN")
			return pd.DataFrame()
		
		plt.figure(figsize=(6, 5))
		sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0)
		plt.title('Correlation Heatmap')
		plt.tight_layout()
		if out_png:
			plt.savefig(out_png, dpi=160)
		plt.close()
		return corr
	except Exception as e:
		print(f"   Warning: Could not create correlation heatmap: {e}")
		plt.close()
		return pd.DataFrame()


def pairplot_sample(df: pd.DataFrame, out_png: str = None, max_rows: int = 3000):
	sample = df.sample(min(len(df), max_rows), random_state=42)
	cols = [c for c in ['pnl_usd', 'trade_value_usd', 'Execution Price', 'sentiment_score'] if c in df.columns]
	g = sns.pairplot(sample[cols], corner=True)
	if out_png:
		g.savefig(out_png, dpi=160)
	plt.close('all')
