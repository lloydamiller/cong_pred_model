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

lm1 = smf.ols(formula='per_gop ~ gopvi + inc_gop + gop_cash_adv_end', data=df).fit()
lm1.rsquared

'''
R Squared
0.8597148606005196     gopvi + inc_gop + gop_cash_adv_end
'''

feature_cols = ['gopvi','inc_gop','gop_cash_adv_end']
X = df[feature_cols]
y = df.per_gop
lm2 = LinearRegression()
lm2.fit(X, y)
print lm2.intercept_
print lm2.coef_

zip(feature_cols, lm2.coef_)
'''
[('gopvi',   0.82784430761145078), 
('inc_gop',  5.5514614012928236)]
'''

lm1.conf_int()

'''
                          0          1
Intercept         45.958534  47.474664
gopvi              0.778774   0.876915
inc_gop            4.150680   6.952243
gop_cash_adv_end   0.000001   0.000002
'''

lm1.summary()

"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                per_gop   R-squared:                       0.860
Model:                            OLS   Adj. R-squared:                  0.859
Method:                 Least Squares   F-statistic:                     1003.
Date:                Fri, 29 May 2015   Prob (F-statistic):          6.41e-209
Time:                        11:35:07   Log-Likelihood:                -1506.3
No. Observations:                 495   AIC:                             3021.
Df Residuals:                     491   BIC:                             3037.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------------
Intercept           46.7166      0.386    121.083      0.000        45.959    47.475
gopvi                0.8278      0.025     33.147      0.000         0.779     0.877
inc_gop              5.5515      0.713      7.787      0.000         4.151     6.952
gop_cash_adv_end  1.813e-06   3.49e-07      5.187      0.000      1.13e-06   2.5e-06
==============================================================================
Omnibus:                       22.876   Durbin-Watson:                   1.948
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               58.509
Skew:                          -0.133   Prob(JB):                     1.97e-13
Kurtosis:                       4.663   Cond. No.                     3.00e+06
==============================================================================
"""

# K-FOLD CROSS VALIDATION FOR LINEAR REGRESSION

feature_cols = ['gopvi','inc_gop']
# 5.26368908322

feature_cols = ['gopvi','inc_gop','gop_cash_adv_beg']
# 5.11210444695

feature_cols = ['gopvi','inc_gop','gop_cash_adv_end']
# 5.1040030191

feature_cols = ['gopvi','inc_gop','gop_cash_adv_beg','gop_cash_adv_end']
# 5.12151421201

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

feature_cols = ['gopvi','gop_cash_adv_beg','tot_rec_gop','inc_gop','gop_fund_adv']

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
3           inc_gop    0.004956
4      gop_fund_adv    0.002363

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

feature_cols = ['gopvi','gop_cash_adv_beg','tot_rec_gop','inc_gop']
# RSME 5.01590013856


rfclf = RandomForestRegressor(n_estimators=1000, max_features=3, oob_score=True, random_state=1)

rfclf.fit(df[feature_cols], df.per_gop)

pd.DataFrame({'feature':feature_cols, 'importance':rfclf.feature_importances_})

'''
            feature  importance
0             gopvi    0.675347
1  gop_cash_adv_beg    0.227494
2       tot_rec_gop    0.077886
3           inc_gop    0.019273
'''

rfclf.oob_score_
# 0.86704165734478567

scores = cross_val_score(rfclf, X, y, cv=10, scoring='mean_squared_error')
scores_sqrt = np.sqrt(-scores)
print np.mean(scores_sqrt)

# RSME 5.01590013856


