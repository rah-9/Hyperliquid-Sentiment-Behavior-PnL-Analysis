import os
import pandas as pd
import plotly.express as px


def plot_sentiment_vs_total_pnl(df: pd.DataFrame, out_html: str):
	agg = df.set_index('timestamp_utc').sort_index().resample('1D').agg({
		'pnl_usd': 'sum',
		'sentiment_score': 'mean'
	}).reset_index()
	fig = px.line(agg, x='timestamp_utc', y=['pnl_usd', 'sentiment_score'], title='Sentiment vs Total PnL (Daily)')
	fig.write_html(out_html)
	return fig


def plot_top_traders_by_sentiment(df: pd.DataFrame, out_html: str, top_k: int = 10):
	# Check if Account column exists
	if 'Account' not in df.columns:
		# Try alternative column names
		if 'account' in df.columns:
			df = df.rename(columns={'account': 'Account'})
		else:
			# If no account column, skip this plot
			print(f"   Warning: 'Account' column not found, skipping top traders plot")
			return None
	
	grp = df.groupby(['sentiment_class', 'Account'], observed=True)['pnl_usd'].sum().reset_index()
	if len(grp) == 0:
		print(f"   Warning: No data for top traders plot")
		return None
	
	top = grp.sort_values(['sentiment_class', 'pnl_usd'], ascending=[True, False]).groupby('sentiment_class').head(top_k)
	if len(top) == 0:
		print(f"   Warning: No top traders data after filtering")
		return None
	
	fig = px.bar(top, x='Account', y='pnl_usd', color='sentiment_class', barmode='group', title=f'Top {top_k} Profitable Traders per Sentiment Phase')
	fig.write_html(out_html)
	return fig


def plot_volume_vs_sentiment(df: pd.DataFrame, out_html: str):
	agg = df.set_index('timestamp_utc').sort_index().resample('1D').agg({
		'trade_value_usd': 'sum',
		'sentiment_score': 'mean'
	}).reset_index()
	fig = px.scatter(agg, x='sentiment_score', y='trade_value_usd', trendline='ols', title='Trade Volume vs Fear/Greed Index')
	fig.write_html(out_html)
	return fig
