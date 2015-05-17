# -*- coding: utf-8 -*-
"""
DAT5 Project Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

import statsmodels.formula.api as smf

df = pd.read_csv('data/fulldataset.csv')

'''
LINEAR REGRESSION
'''

# Regression for PVI score and Incumbent/Open/Challenger status of GOP candidate

lm1 = smf.ols(formula='per_gop ~ gopvi + inc_gop', data=df).fit()
lm1.rsquared

'''
R Squared
0.8520277791837757     gopvi + inc_gop
'''

feature_cols = ['gopvi','inc_gop']
X = df[feature_cols]
y = df.per_gop
lm2 = LinearRegression()
lm2.fit(X, y)
print lm2.intercept_
print lm2.coef_

zip(feature_cols, lm2.coef_)
'''
[('gopvi',                    0.92660532238700444), 
 ('can_inc_cha_ope_sea_gop', -3.8554427911707645)]
'''

lm1.conf_int()

'''
                                    0          1
Intercept                   44.516323  46.145216
seat_stat_gop[T.INCUMBENT]   7.017446   9.599684
seat_stat_gop[T.OPEN]        0.468559   3.613564
gopvi                        0.762018   0.865181
'''

lm1.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                per_gop   R-squared:                       0.852
Model:                            OLS   Adj. R-squared:                  0.851
Method:                 Least Squares   F-statistic:                     1416.
Date:                Sun, 17 May 2015   Prob (F-statistic):          7.32e-205
Time:                        11:48:20   Log-Likelihood:                -1519.5
No. Observations:                 495   AIC:                             3045.
Df Residuals:                     492   BIC:                             3058.
Df Model:                           2                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     45.8695      0.359    127.899      0.000        45.165    46.574
gopvi          0.8297      0.026     32.385      0.000         0.779     0.880
inc_gop        7.6326      0.605     12.623      0.000         6.445     8.821
==============================================================================
Omnibus:                       12.740   Durbin-Watson:                   1.884
Prob(Omnibus):                  0.002   Jarque-Bera (JB):               24.658
Skew:                          -0.030   Prob(JB):                     4.42e-06
Kurtosis:                       4.092   Cond. No.                         33.6
==============================================================================
"""

# K-FOLD CROSS VALIDATION FOR LINEAR REGRESSION

feature_cols = ['gopvi','inc_gop','gop_cash_adv_beg']
# 5.11210444695

X = df[feature_cols]
y = df.per_gop
lm = LinearRegression()
scores = cross_val_score(lm, X, y, cv=10, scoring='mean_squared_error')
scores_sqrt = np.sqrt(-scores)
print np.mean(scores_sqrt)


'''

DECISION TREE

The decision tree will provide the best features.

'''

feature_cols = ['gopvi','gop_cash_adv_beg','tot_rec_gop','gop_fund_adv','inc_gop']

X = df[feature_cols]
y = df.per_gop

# Determine Best Maximum Depth
max_depth_range = range(1, 11)
RMSE_scores = []
for depth in max_depth_range:
    treereg = DecisionTreeRegressor(max_depth=depth, random_state=1, min_samples_leaf=5)
    MSE_scores = cross_val_score(treereg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
RMSE_scores
plt.plot(max_depth_range, RMSE_scores)
# Best RMSE is 5.7996253874630179 at depth 5

# Determine Best Minimum Leaves
max_leaf_range = range(5, 20)
RMSE_scores = []
for leaf in max_leaf_range:
    treereg = DecisionTreeRegressor(max_depth=5, random_state=1, min_samples_leaf=leaf)
    MSE_scores = cross_val_score(treereg, X, y, cv=5, scoring='mean_squared_error')
    RMSE_scores.append(np.mean(np.sqrt(-MSE_scores)))
RMSE_scores
plt.plot(max_leaf_range, RMSE_scores)
# Best RMSE is 5.5127146937992295 at minmum 7 leaves (shoots up)

# Fit The Model
treereg = DecisionTreeRegressor(max_depth=5, random_state=1, min_samples_leaf=7)
treereg.fit(X, y)

# Print Out Important Features
pd.DataFrame({'feature':feature_cols, 'importance':treereg.feature_importances_})

'''
Importance of features for depth 5, min 7 samples per leaf

            feature  importance
0             gopvi    0.888646
1  gop_cash_adv_beg    0.080406
2       tot_rec_gop    0.023629
3      gop_fund_adv    0.002363
4           inc_gop    0.004956

'''

# PRINT THE TREE

from sklearn.tree import export_graphviz
with open("tree.dot", 'wb') as f:
    f = export_graphviz(treereg, out_file=f, feature_names=feature_cols)
# At the command line, run this to convert to PNG:
# dot -Tpng tree.dot -o tree.png

'''

RANDOM FOREST

Using the features from the decision tree model, create a random forest with 1000 trees.

'''

feature_cols = ['gopvi','gop_cash_adv_beg','tot_rec_gop','gop_fund_adv','inc_gop']

rfclf = RandomForestRegressor(n_estimators=1000, max_features=3, oob_score=True, random_state=1)

rfclf.fit(df[feature_cols], df.per_gop)

pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})

'''
            feature  importance
0             gopvi    0.571315
1  gop_cash_adv_beg    0.243209
2       tot_rec_gop    0.076087
3      gop_fund_adv    0.068150
4           inc_gop    0.041238
'''

rfclf.oob_score_
# 0.86704165734478567

scores = cross_val_score(rfclf, X, y, cv=10, scoring='mean_squared_error')
scores_sqrt = np.sqrt(-scores)
print np.mean(scores_sqrt)

# RSME 5.0166672203


