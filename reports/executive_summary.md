# Executive Summary: Sentiment vs Hyperliquid Trader Performance

## Key Findings
- T-Test Fear vs Greed PnL: t=-16.407, p=0.000 (n_fear=133871, n_greed=43251)
- Best Regression (XGB): R2=0.998, RMSE=6.119
- Best Classifier (XGB): Acc=0.999, F1=0.999

## Descriptive Highlights
|                 |   count |       mean |      std |          min |           1% |       5% |     50% |       95% |     99% |      max |
|:----------------|--------:|-----------:|---------:|-------------:|-------------:|---------:|--------:|----------:|--------:|---------:|
| pnl_usd         |  211224 |    31.3061 |   134.81 | -140.767     | -140.73      | -5.32101 |   0     |   165.789 |  1023.2 |  1023.32 |
| trade_value_usd |  211224 |  4281.76   | 12439.7  |   11.07      |   11.07      | 17.5115  | 597.045 | 20023     | 88887.2 | 88887.2  |
| Execution Price |  211224 | 11383.4    | 29351.6  |    0.0125305 |    0.0125308 |  0.14872 |  18.28  | 93259     | 99011   | 99011    |

## PnL by Sentiment
| sentiment_class   |   count |    mean |   median |     std |              sum |
|:------------------|--------:|--------:|---------:|--------:|-----------------:|
| unknown           |   26961 | 18.1768 |        0 | 111.675 | 490064           |
| neutral           |    7141 | 29.7834 |        0 | 131.313 | 212683           |
| fear              |  133871 | 29.9672 |        0 | 129.844 |      4.01174e+06 |
| greed             |   43251 | 43.886  |        0 | 160.247 |      1.89811e+06 |

## OLS Regression Summary
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                pnl_usd   R-squared:                       0.096
Model:                            OLS   Adj. R-squared:                  0.096
Method:                 Least Squares   F-statistic:                     1114.
Date:                Thu, 06 Nov 2025   Prob (F-statistic):               0.00
Time:                        23:51:16   Log-Likelihood:            -1.3249e+06
No. Observations:              211224   AIC:                         2.650e+06
Df Residuals:                  211219   BIC:                         2.650e+06
Df Model:                           4                                         
Covariance Type:                  HC3                                         
=========================================================================================
                            coef    std err          z      P>|z|      [0.025      0.975]
-----------------------------------------------------------------------------------------
Intercept               -12.0050      1.305     -9.198      0.000     -14.563      -9.447
C(side_norm)[T.short]    21.0680      0.550     38.299      0.000      19.990      22.146
sentiment_score          44.9919      2.438     18.455      0.000      40.214      49.770
trade_value_usd           0.0034   6.42e-05     52.796      0.000       0.003       0.004
execution_price          -0.0004   9.96e-06    -42.381      0.000      -0.000      -0.000
==============================================================================
Omnibus:                   206488.825   Durbin-Watson:                   1.117
Prob(Omnibus):                  0.000   Jarque-Bera (JB):          9401107.253
Skew:                           4.911   Prob(JB):                         0.00
Kurtosis:                      34.172   Cond. No.                     2.83e+05
==============================================================================

Notes:
[1] Standard Errors are heteroscedasticity robust (HC3)
[2] The condition number is large, 2.83e+05. This might indicate that there are
strong multicollinearity or other numerical problems.
```