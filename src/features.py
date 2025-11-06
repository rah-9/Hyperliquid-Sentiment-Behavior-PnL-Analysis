import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List


def add_sentiment_transitions(sentiment_daily: pd.DataFrame) -> pd.DataFrame:
	df = sentiment_daily.copy()
	df['sentiment_prev'] = df['sentiment_class'].shift(1)
	df['transition'] = df['sentiment_prev'].astype(str) + '->' + df['sentiment_class'].astype(str)
	return df


def rolling_features(trades: pd.DataFrame) -> pd.DataFrame:
	"""Add rolling 3D PnL and 7D sentiment over trades indexed by timestamp."""
	df = trades.copy()
	# Preserve Account column if it exists
	account_preserved = 'Account' in df.columns
	if account_preserved:
		account_data = df['Account'].copy()
	
	df = df.sort_values('timestamp_utc')
	
	# Ensure date_utc exists and is timezone-naive for merge
	if 'date_utc' not in df.columns:
		df['date_utc'] = df['timestamp_utc'].dt.floor('D')
	# Ensure date_utc is timezone-naive
	df['date_utc'] = pd.to_datetime(df['date_utc'])
	if df['date_utc'].dt.tz is not None:
		df['date_utc'] = df['date_utc'].dt.tz_localize(None)
	
	# Daily aggregates for rolling stats
	daily = df.set_index('timestamp_utc').resample('1D').agg({
		'pnl_usd': 'sum',
		'sentiment_score': 'mean'
	}).rename(columns={'pnl_usd': 'pnl_usd_daily', 'sentiment_score': 'sentiment_daily'})
	daily['pnl_ma_3d'] = daily['pnl_usd_daily'].rolling(3, min_periods=1).mean()
	daily['sentiment_ma_7d'] = daily['sentiment_daily'].rolling(7, min_periods=1).mean()
	daily = daily.reset_index()
	daily['date_utc'] = daily['timestamp_utc'].dt.floor('D')
	# Ensure date_utc is timezone-naive for merge
	daily['date_utc'] = pd.to_datetime(daily['date_utc'])
	if daily['date_utc'].dt.tz is not None:
		daily['date_utc'] = daily['date_utc'].dt.tz_localize(None)
	daily = daily.drop(columns=['timestamp_utc'])
	
	df = df.merge(daily, on='date_utc', how='left')
	
	# Restore Account if it was lost
	if account_preserved and 'Account' not in df.columns:
		df['Account'] = account_data
	elif account_preserved:
		# Ensure Account values are preserved (merge might have changed them)
		df['Account'] = df['Account'].fillna(account_data)
	
	return df


def trader_performance_features(trades: pd.DataFrame) -> pd.DataFrame:
	"""Compute per-account rolling performance ratios: profit factor and win rate (14D)."""
	df = trades.copy()
	
	# Check if Account exists
	if 'Account' not in df.columns:
		print("   Warning: Account column missing in trader_performance_features, creating placeholder")
		df['Account'] = pd.Series(index=df.index, dtype='object').astype('category')
	
	# Ensure Account is not all NaN
	if df['Account'].isna().all():
		print("   Warning: Account column is all NaN, cannot compute per-account features")
		df['win_rate_14d'] = np.nan
		df['profit_factor_14d'] = np.nan
		return df
	
	df = df.sort_values(['Account', 'timestamp_utc'])
	# Store original index to restore order later
	original_index = df.index.copy()
	
	def perf(group: pd.DataFrame) -> pd.DataFrame:
		# Account should be in the group (same for all rows) - preserve it explicitly
		account_val = group['Account'].iloc[0] if 'Account' in group.columns else None
		# Store original index of this group
		group_index = group.index.copy()
		# Make sure Account is a column before setting index
		if 'Account' not in group.columns and account_val is not None:
			group = group.copy()
			group['Account'] = account_val
		g = group.set_index('timestamp_utc')
		# Ensure Account is still a column (not in index)
		if 'Account' not in g.columns and account_val is not None:
			g['Account'] = account_val
		wins = (g['pnl_usd'] > 0).astype(int)
		g['win_rate_14d'] = wins.rolling('14D').mean()
		profits = g['pnl_usd'].clip(lower=0)
		losses = -g['pnl_usd'].clip(upper=0)
		pf = profits.rolling('14D').sum() / (losses.rolling('14D').sum() + 1e-6)
		g['profit_factor_14d'] = pf
		result = g.reset_index()
		# Ensure Account is preserved (double-check)
		if 'Account' not in result.columns and account_val is not None:
			result['Account'] = account_val
		# Restore original index
		result.index = group_index
		return result
	
	try:
		# Use group_keys=False to avoid Account in index
		# Try with include_groups if available (pandas >= 1.5), otherwise fall back
		try:
			feat = df.groupby('Account', group_keys=False, observed=True).apply(perf, include_groups=True)
		except TypeError:
			# Fallback for older pandas versions that don't support include_groups
			feat = df.groupby('Account', group_keys=False, observed=True).apply(perf)
		# Ensure Account column exists
		if 'Account' not in feat.columns:
			# Try to restore from original dataframe by matching index
			if len(feat) == len(df) and feat.index.equals(original_index):
				feat['Account'] = df['Account'].values
			else:
				# Last resort: merge back Account from original
				feat = feat.reset_index()
				df_temp = df[['Account']].reset_index()
				feat = feat.merge(df_temp[['Account']], left_index=True, right_index=True, how='left', suffixes=('', '_orig'))
				if 'Account_orig' in feat.columns:
					feat['Account'] = feat['Account_orig'].fillna(feat.get('Account', ''))
					feat = feat.drop(columns=['Account_orig'])
				feat = feat.set_index('index') if 'index' in feat.columns else feat
	except Exception as e:
		print(f"   Warning: Error computing trader performance features: {e}")
		df['win_rate_14d'] = np.nan
		df['profit_factor_14d'] = np.nan
		return df
	
	return feat


def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
	"""Construct X, y for modeling with robust scaling and one-hot encoding of side and sentiment."""
	d = df.copy()
	# Encodings
	if 'side_norm' in d.columns:
		d = pd.get_dummies(d, columns=['side_norm'], drop_first=True)
	if 'sentiment_class' in d.columns:
		d = pd.get_dummies(d, columns=['sentiment_class'], drop_first=True)

	# Feature columns
	numeric_cols = [c for c in [
		'trade_value_usd', 'Execution Price', 'pnl_ma_3d', 'sentiment_ma_7d',
		'win_rate_14d', 'profit_factor_14d', 'sentiment_score',
		'pnl_usd_robust', 'trade_value_usd_robust', 'Execution Price_robust'
	] if c in d.columns]
	cat_cols = [c for c in d.columns if c.startswith('side_norm_') or c.startswith('sentiment_class_')]
	feature_cols = numeric_cols + cat_cols

	X = d[feature_cols].fillna(0.0)
	y_reg = d['pnl_usd'].fillna(0.0)
	y_clf = (d['pnl_usd'] > 0).astype(int)

	# Scale numeric only
	scaler = RobustScaler()
	if numeric_cols:
		X_numeric = pd.DataFrame(scaler.fit_transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
		X = pd.concat([X_numeric, X[cat_cols]], axis=1)

	return X, (y_reg, y_clf), feature_cols
