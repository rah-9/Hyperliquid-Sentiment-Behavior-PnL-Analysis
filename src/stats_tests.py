import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict


def ttest_pnl_fear_vs_greed(df: pd.DataFrame) -> Dict[str, float]:
	fear = df.loc[df['sentiment_class'].isin(['extreme_fear', 'fear']), 'pnl_usd'].dropna()
	greed = df.loc[df['sentiment_class'].isin(['extreme_greed', 'greed']), 'pnl_usd'].dropna()
	if len(fear) < 3 or len(greed) < 3:
		return {'t_stat': np.nan, 'p_value': np.nan, 'n_fear': len(fear), 'n_greed': len(greed)}
	t_stat, p_val = stats.ttest_ind(fear, greed, equal_var=False)
	return {'t_stat': float(t_stat), 'p_value': float(p_val), 'n_fear': int(len(fear)), 'n_greed': int(len(greed))}


def correlation_tests(df: pd.DataFrame) -> pd.DataFrame:
	cols = [c for c in ['pnl_usd', 'trade_value_usd', 'Execution Price', 'sentiment_score'] if c in df.columns]
	if len(cols) < 2:
		return pd.DataFrame(columns=['var1', 'var2', 'rho', 'p_value'])
	
	out = []
	for i, c1 in enumerate(cols):
		for c2 in cols[i+1:]:
			# Check if both columns have valid data
			valid_mask = df[[c1, c2]].notna().all(axis=1)
			if valid_mask.sum() < 3:  # Need at least 3 valid pairs
				continue
			try:
				r, p = stats.spearmanr(df[c1], df[c2], nan_policy='omit')
				if np.isfinite(r) and np.isfinite(p):
					out.append({'var1': c1, 'var2': c2, 'rho': r, 'p_value': p})
			except Exception:
				continue
	
	if len(out) == 0:
		return pd.DataFrame(columns=['var1', 'var2', 'rho', 'p_value'])
	return pd.DataFrame(out).sort_values('p_value')


def ols_sentiment_on_pnl(df: pd.DataFrame):
	"""OLS: pnl_usd ~ sentiment_score + trade_value_usd + execution_price + C(side_norm)."""
	# Diagnostic: Check data availability before dropping
	pnl_valid = df['pnl_usd'].notna().sum() if 'pnl_usd' in df.columns else 0
	sentiment_valid = df['sentiment_score'].notna().sum() if 'sentiment_score' in df.columns else 0
	both_valid = df[['pnl_usd', 'sentiment_score']].notna().all(axis=1).sum() if 'pnl_usd' in df.columns and 'sentiment_score' in df.columns else 0
	
	if both_valid == 0:
		raise ValueError(
			f"Insufficient data for OLS regression: "
			f"pnl_usd valid: {pnl_valid}/{len(df)}, "
			f"sentiment_score valid: {sentiment_valid}/{len(df)}, "
			f"both valid: {both_valid}/{len(df)}"
		)
	
	d = df.dropna(subset=['pnl_usd', 'sentiment_score']).copy()
	
	# Check if we have enough data
	if len(d) < 10:
		raise ValueError(f"Insufficient data for OLS regression: only {len(d)} valid observations (need at least 10)")
	
	# Rename column with space to avoid formula parsing issues
	if 'Execution Price' in d.columns:
		d['execution_price'] = d['Execution Price']
	
	# Ensure side_norm exists and has multiple levels
	if 'side_norm' not in d.columns:
		d['side_norm'] = 'unknown'
	
	# Check if side_norm has sufficient levels for categorical encoding
	side_levels = d['side_norm'].nunique()
	if side_levels < 2:
		# If only one level, use numeric only
		if 'execution_price' in d.columns:
			formula = 'pnl_usd ~ sentiment_score + trade_value_usd + execution_price'
		else:
			formula = 'pnl_usd ~ sentiment_score + trade_value_usd'
	else:
		# Use categorical encoding
		if 'execution_price' in d.columns:
			formula = 'pnl_usd ~ sentiment_score + trade_value_usd + execution_price + C(side_norm)'
		else:
			formula = 'pnl_usd ~ sentiment_score + trade_value_usd + C(side_norm)'
	
	try:
		model = smf.ols(formula=formula, data=d).fit(cov_type='HC3')
	except Exception as e:
		# Fallback to simpler model if formula fails
		try:
			formula = 'pnl_usd ~ sentiment_score + trade_value_usd'
			model = smf.ols(formula=formula, data=d).fit(cov_type='HC3')
		except Exception as e2:
			# Last resort: simplest model
			formula = 'pnl_usd ~ sentiment_score'
			model = smf.ols(formula=formula, data=d).fit(cov_type='HC3')
	
	return model
