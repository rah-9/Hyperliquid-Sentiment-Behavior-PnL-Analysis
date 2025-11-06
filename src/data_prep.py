import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from dateutil import parser
from typing import Tuple

IST = pytz.timezone('Asia/Kolkata')
UTC = pytz.UTC


def _to_utc(ts: pd.Series) -> pd.Series:
	"""Convert a timestamp series with unknown/IST tz to UTC."""
	def parse_one(x):
		if pd.isna(x):
			return pd.NaT
		if isinstance(x, (pd.Timestamp, datetime)):
			if x.tzinfo is None:
				return x.replace(tzinfo=UTC)
			return x.astimezone(UTC)
		try:
			dt = parser.parse(str(x))
			if dt.tzinfo is None:
				return dt.replace(tzinfo=UTC)
			return dt.astimezone(UTC)
		except Exception:
			return pd.NaT
	return pd.to_datetime(pd.Series([parse_one(v) for v in ts]), errors='coerce', utc=True)


def load_fear_greed(csv_path: str) -> pd.DataFrame:
	"""Load Fear-Greed index; return dataframe with UTC date index and normalized sentiment.

	Expected columns: timestamp, value, classification, date
	"""
	df = pd.read_csv(csv_path)
	if 'date' in df.columns:
		df['date'] = pd.to_datetime(df['date'], errors='coerce')
		# Only localize if not already timezone-aware
		if df['date'].dt.tz is None:
			df['date'] = df['date'].dt.tz_localize('UTC')
		date_col = 'date'
	elif 'timestamp' in df.columns:
		df['date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce', utc=True)
		date_col = 'date'
	else:
		raise ValueError('Fear-Greed CSV missing date/timestamp')

	if 'value' in df.columns:
		df['sentiment_value_raw'] = pd.to_numeric(df['value'], errors='coerce')
		max_v = 100.0
		min_v = 0.0
		df['sentiment_score'] = (df['sentiment_value_raw'] - min_v) / (max_v - min_v)
	else:
		df['sentiment_score'] = np.nan

	if 'classification' in df.columns:
		cls = df['classification'].astype(str).str.lower()
		df['sentiment_class'] = np.select(
			[
				cls.str.contains('extreme fear'),
				cls.str.contains('fear'),
				cls.str.contains('neutral'),
				cls.str.contains('greed'),
				cls.str.contains('extreme greed'),
			],
			['extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'],
			default='unknown',
		)
	else:
		df['sentiment_class'] = 'unknown'

	df['date'] = pd.to_datetime(df[date_col]).dt.floor('D')
	df = df.sort_values('date').drop_duplicates('date', keep='last')
	df = df.set_index('date').sort_index()
	return df[['sentiment_score', 'sentiment_class']]


def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
	for c in cols:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors='coerce')
	return df


def load_hyperliquid_trades(csv_path: str) -> pd.DataFrame:
	"""Load Hyperliquid historical trades CSV and standardize types."""
	df = pd.read_csv(csv_path)
	# Strip whitespace from column names
	df.columns = df.columns.str.strip()
	
	# Debug: Print first few column names to verify Account is there
	if 'Account' not in df.columns:
		account_variants = [c for c in df.columns if 'account' in c.lower() or 'Account' in c]
		if account_variants:
			print(f"   Debug: Found account-like columns: {account_variants}, will map to 'Account'")
		else:
			print(f"   Debug: Account column not found. First 10 columns: {list(df.columns)[:10]}")
	
	# Parse timestamp - prioritize Unix timestamp (in milliseconds) if available
	if 'Timestamp' in df.columns:
		# Check if Timestamp is numeric (Unix timestamp in milliseconds)
		if pd.api.types.is_numeric_dtype(df['Timestamp']):
			# Unix timestamp in milliseconds
			df['timestamp_utc'] = pd.to_datetime(df['Timestamp'], unit='ms', errors='coerce', utc=True)
		else:
			# Try parsing as datetime string
			df['timestamp_utc'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
	elif 'Timestamp IST' in df.columns:
		# Handle DD-MM-YYYY HH:MM format (common in IST timezone)
		# Try explicit format first
		dt = pd.to_datetime(df['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
		if dt.isna().sum() > len(dt) * 0.1:  # If more than 10% fail, try alternative
			# Fallback: try without format (auto-detect)
			dt = pd.to_datetime(df['Timestamp IST'], errors='coerce')
		df['timestamp_utc'] = dt.dt.tz_localize(IST, nonexistent='shift_forward', ambiguous='NaT').dt.tz_convert('UTC')
	else:
		df['timestamp_utc'] = _to_utc(df.iloc[:, 0])

	num_cols = ['Execution Price', 'Size Tokens', 'Size USD', 'Closed PnL', 'Fee']
	df = _coerce_numeric(df, num_cols)

	if 'Side' in df.columns:
		s = df['Side'].astype(str).str.lower()
		df['side_norm'] = np.where(s.str.contains('buy') | s.str.contains('long'), 'long',
							   np.where(s.str.contains('sell') | s.str.contains('short'), 'short', 'unknown'))
	elif 'Direction' in df.columns:
		d = df['Direction'].astype(str).str.lower()
		df['side_norm'] = np.where(d.str.contains('long'), 'long', np.where(d.str.contains('short'), 'short', 'unknown'))
	else:
		df['side_norm'] = 'unknown'

	# Ensure Account column exists (case-insensitive check, handle whitespace)
	account_col = None
	if 'Account' in df.columns:
		account_col = 'Account'
		print(f"   Account column found: {df['Account'].notna().sum()}/{len(df)} non-null")
	else:
		# Try case-insensitive match
		account_cols = [c for c in df.columns if c.strip().lower() == 'account']
		if account_cols:
			account_col = account_cols[0]
			if account_col != 'Account':
				df['Account'] = df[account_col]
				print(f"   Account column mapped from '{account_col}'")
		else:
			# Create placeholder if missing
			print(f"   Warning: Account column not found in CSV. Available columns: {list(df.columns)[:15]}")
	
	if account_col or 'Account' in df.columns:
		# Ensure Account is not all NaN and convert to category
		if 'Account' in df.columns:
			if df['Account'].isna().all():
				print(f"   Warning: Account column exists but is all NaN")
			df['Account'] = df['Account'].astype('category')
	else:
		# Create placeholder if missing
		df['Account'] = pd.Series(index=df.index, dtype='object').astype('category')
	
	# Ensure Coin column exists
	if 'Coin' not in df.columns:
		coin_cols = [c for c in df.columns if c.lower() == 'coin']
		if coin_cols:
			df['Coin'] = df[coin_cols[0]].astype('category')
		else:
			df['Coin'] = pd.Series(index=df.index, dtype='object').astype('category')
	else:
		df['Coin'] = df['Coin'].astype('category')

	for c in ['Execution Price', 'Size Tokens', 'Size USD', 'Closed PnL', 'Fee']:
		if c in df.columns:
			mask_inf = ~np.isfinite(df[c])
			df.loc[mask_inf, c] = np.nan

	for c in ['Execution Price', 'Size USD', 'Closed PnL']:
		if c in df.columns:
			low, high = df[c].quantile([0.01, 0.99])
			df[c] = df[c].clip(lower=low, upper=high)

	df['trade_value_usd'] = df.get('Size USD', np.nan)
	df['pnl_usd'] = df.get('Closed PnL', np.nan)
	df['fee_usd'] = df.get('Fee', np.nan)
	df['is_profitable'] = (df['pnl_usd'] > 0).astype('Int64')

	df['date_utc'] = df['timestamp_utc'].dt.floor('D')
	return df


def align_and_merge(trades: pd.DataFrame, sentiment_daily: pd.DataFrame) -> pd.DataFrame:
	"""Attach sentiment to each trade by date (UTC)."""
	# Ensure date_utc is datetime
	if 'date_utc' not in trades.columns:
		if 'timestamp_utc' in trades.columns:
			trades['date_utc'] = trades['timestamp_utc'].dt.floor('D')
		else:
			raise ValueError("Cannot merge: missing date_utc or timestamp_utc in trades")
	
	# Ensure date_utc is timezone-naive for merge (or both timezone-aware)
	trades['date_utc'] = pd.to_datetime(trades['date_utc'])
	if trades['date_utc'].dt.tz is not None:
		trades['date_utc'] = trades['date_utc'].dt.tz_localize(None)
	
	# Prepare sentiment for merge
	sentiment_for_merge = sentiment_daily.reset_index().rename(columns={'date': 'date_utc'})
	sentiment_for_merge['date_utc'] = pd.to_datetime(sentiment_for_merge['date_utc'])
	if sentiment_for_merge['date_utc'].dt.tz is not None:
		sentiment_for_merge['date_utc'] = sentiment_for_merge['date_utc'].dt.tz_localize(None)
	
	# Diagnostic: Check date ranges
	trade_date_min = trades['date_utc'].min()
	trade_date_max = trades['date_utc'].max()
	sentiment_date_min = sentiment_for_merge['date_utc'].min()
	sentiment_date_max = sentiment_for_merge['date_utc'].max()
	
	print(f"   Date ranges - Trades: {trade_date_min.date()} to {trade_date_max.date()}")
	print(f"   Date ranges - Sentiment: {sentiment_date_min.date()} to {sentiment_date_max.date()}")
	
	# Merge
	# Explicitly preserve Account and other important columns
	cols_to_preserve = ['Account'] if 'Account' in trades.columns else []
	merged = trades.merge(
		sentiment_for_merge[['date_utc', 'sentiment_score', 'sentiment_class']],
		how='left', on='date_utc'
	)
	
	# Ensure Account is preserved (should be automatic, but double-check)
	if 'Account' not in merged.columns and cols_to_preserve:
		merged['Account'] = trades['Account'].values
	
	# Check merge success
	matched = merged['sentiment_score'].notna().sum()
	print(f"   Sentiment merge: {matched}/{len(merged)} trades matched with sentiment data")
	
	# Forward/backward fill sentiment scores
	merged['sentiment_score'] = merged['sentiment_score'].ffill().bfill()
	merged['sentiment_class'] = merged['sentiment_class'].fillna('unknown')
	
	final_matched = merged['sentiment_score'].notna().sum()
	if final_matched < len(merged):
		print(f"   After fill: {final_matched}/{len(merged)} trades have sentiment (filled from nearby dates)")
	
	return merged


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Add robust-normalized versions of key numeric metrics."""
	for c in ['pnl_usd', 'trade_value_usd', 'Execution Price']:
		if c in df.columns:
			med = df[c].median(skipna=True)
			iqr = df[c].quantile(0.75) - df[c].quantile(0.25)
			if iqr == 0 or not np.isfinite(iqr):
				iqr = 1.0
			df[f'{c}_robust'] = (df[c] - med) / iqr
	return df


def prepare_datasets(fg_path: str, trades_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Full preparation pipeline returning (sentiment_daily, trades_raw, trades_enriched)."""
	sentiment_daily = load_fear_greed(fg_path)
	trades = load_hyperliquid_trades(trades_path)
	merged = align_and_merge(trades, sentiment_daily)
	merged = add_normalized_columns(merged)
	return sentiment_daily, trades, merged
