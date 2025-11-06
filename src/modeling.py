import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import shap


def train_regressors(X: pd.DataFrame, y: pd.Series, out_dir: str) -> Dict[str, Any]:
	os.makedirs(out_dir, exist_ok=True)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

	models = {
		'LinearRegression': LinearRegression(),
		'RidgeCV': RidgeCV(alphas=(0.1, 1.0, 10.0)),
		'XGBRegressor': XGBRegressor(
			n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
			random_state=42, n_jobs=4, objective='reg:squarederror')
	}

	results = {}
	for name, model in models.items():
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		res = {
			'R2': float(r2_score(y_test, y_pred)),
			'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred)))
		}
		results[name] = res
		joblib.dump(model, os.path.join(out_dir, f'{name}.joblib'))

	# SHAP for XGB if available
	if 'XGBRegressor' in models:
		try:
			explainer = shap.TreeExplainer(models['XGBRegressor'])
			shap_values = explainer.shap_values(X_test)
			shap.summary_plot(shap_values, X_test, show=False)
			export_path = os.path.join(out_dir, 'shap_reg_summary.png')
			import matplotlib.pyplot as plt
			plt.tight_layout()
			plt.savefig(export_path, dpi=160)
			plt.close()
		except Exception:
			pass

	with open(os.path.join(out_dir, 'regression_results.json'), 'w') as f:
		json.dump(results, f, indent=2)
	return results


def train_classifiers(X: pd.DataFrame, y: pd.Series, out_dir: str) -> Dict[str, Any]:
	os.makedirs(out_dir, exist_ok=True)
	# Check if stratification is possible
	try:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
	except ValueError:
		# If stratification fails (e.g., only one class), use regular split
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

	models = {
		'LogisticRegression': LogisticRegression(max_iter=1000, solver='liblinear'),
		'RandomForest': RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42),
		'XGBClassifier': XGBClassifier(
			n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
			random_state=42, n_jobs=4, objective='binary:logistic')
	}

	results = {}
	for name, model in models.items():
		model.fit(X_train, y_train)
		y_pred = model.predict(X_test)
		res = {
			'Accuracy': float(accuracy_score(y_test, y_pred)),
			'F1': float(f1_score(y_test, y_pred))
		}
		results[name] = res
		joblib.dump(model, os.path.join(out_dir, f'{name}.joblib'))

	# SHAP for XGB
	if 'XGBClassifier' in models:
		try:
			explainer = shap.TreeExplainer(models['XGBClassifier'])
			shap_values = explainer.shap_values(X_test)
			shap.summary_plot(shap_values, X_test, show=False)
			import matplotlib.pyplot as plt
			plt.tight_layout()
			plt.savefig(os.path.join(out_dir, 'shap_clf_summary.png'), dpi=160)
			plt.close()
		except Exception:
			pass

	with open(os.path.join(out_dir, 'classification_results.json'), 'w') as f:
		json.dump(results, f, indent=2)
	return results
