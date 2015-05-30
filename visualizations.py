# -*- coding: utf-8 -*-
"""
Created on Thu May 14 13:34:49 2015

@author: lloydamiller
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/fulldataset.csv')
df.to_csv('data/fulldataset.csv', index=False)

'''
CREATE NEW FEATURES FROM ORIGINAL DATA
'''

# GOP Candidate's Cash Advantage

df['gop_cash_adv_beg'] = 0
i = 0
while i <= (len(df)-1):
    df['gop_cash_adv_beg'][i] = df['cas_on_han_beg_of_per_gop'][i] - df['cas_on_han_beg_of_per_dem'][i]
    i += 1

df['gop_cash_adv_end'] = 0
i = 0
while i <= (len(df)-1):
    df['gop_cash_adv_end'][i] = df['cas_on_han_clo_of_per_gop'][i] - df['cas_on_han_clo_of_per_dem'][i]
    i += 1

df['gop_fund_adv'] = 0
i = 0
while i <= (len(df)-1):
    df['gop_fund_adv'][i] = df['tot_rec_gop'][i] - df['tot_rec_dem'][i]
    i += 1

# Define Groups of Urban, Suburban, Rural
# 0 Urban = density > 1000
# 1 Suburban = density < 1000 & > 500
# 2 Rural = density < 500

df['urb_cat'] = 0
i = 0
while i <= (len(df)-1):
    if df['density'][i] < 500.0:
        df['urb_cat'][i] = 2
    elif df['density'][i] < 1000.0:
        df['urb_cat'][i] = 1
    else:
        df['urb_cat'][i] = 0
    i += 1

# Define Swing Seats
# 0 = Dem favorable seat
# 1 = Swing Seat
# 2 = GOP favorable seat

df['swing_seat'] = 0
i = 0
while i <= (len(df)-1):
    if df['gopvi'][i] < -10:
        df['swing_seat'][i] = 0
    elif df['gopvi'][i] > 10:
        df['swing_seat'][i] = 2
    else:
        df['swing_seat'][i] = 1
    i += 1

# Map Candidate Status
df['can_inc_cha_ope_sea_dem'] = df.can_inc_cha_ope_sea_dem.map({'INCUMBENT':0, 'CHALLENGER':1, 'OPEN':2})
df['can_inc_cha_ope_sea_gop'] = df.seat_stat_gop.map({'INCUMBENT':0, 'CHALLENGER':1, 'OPEN':2})
df['inc_gop'] = df.seat_stat_gop.map({'INCUMBENT':1, 'CHALLENGER':0, 'OPEN':0})

'''
'year'
'label'
'can_nam_dem'
'can_par_aff_dem'
'can_inc_cha_ope_sea_dem'
'ind_con_dem'
'tot_con_dem'
'tot_rec_dem'
'tot_dis_dem'
'cas_on_han_beg_of_per_dem'
'cas_on_han_clo_of_per_dem'
'tot_rec_final_dem'
'can_nam_gop'
'can_par_aff_gop'
'can_inc_cha_ope_sea_gop'
'per_dem'
'per_gop'
'ind_con_gop'
'tot_con_gop'
'tot_rec_gop'
'tot_dis_gop'
'cas_on_han_beg_of_per_gop'
'cas_on_han_clo_of_per_gop'
'tot_rec_final_gop'
'gopwin'
'population'
'income'
'homeval'
'rentval'
'gopvi'
'density'
'white'
'gop_cash_adv_beg'
'gop_cash_adv_end'
'urb_cat'
'urb_cat_nam'
'''


# Color Blue for Dem win and Red for Rep win
colors = np.where(df.gopwin == 0, 'b', 'r')

# How Powerful Is District Partisanship?
df.groupby('gopvi').gopwin.sum().hist()

pvi = df[['gopvi','gopwin']]

pvi = pvi.groupby('gopvi').gopwin.mean()
pvi.plot(kind='bar')

# Charting The Correlation between Cash on Hand and Percentage of Vote
df.plot(kind='scatter',x='per_gop', y='gop_cash_adv_end', c='gopvi', colormap='Reds')
plt.title('Cash on Hand Indicator')
plt.xlabel('GOP Percentage')
plt.ylabel('Cash on Hand Advantage');

# Charting Individual Contribution Relevance
df.plot(kind='scatter',x='ind_con_gop', y='ind_con_dem', c=colors, xlim=(0,750000), ylim=(0,750000))
plt.title('Individual Contributions')
plt.xlabel('GOP Indiv. Contributions')
plt.ylabel('DEM Indiv. Contributions');

# Charting Spending Relevance
df.plot(kind='scatter',x='tot_dis_gop', y='tot_dis_dem', c=colors, xlim=(0,750000), ylim=(0,750000))
plt.title('Campaign Spending')
plt.xlabel('GOP Disbursements')
plt.ylabel('DEM Disbursements');

# Charting Cash On Hand Relevance
df.plot(kind='scatter',x='cas_on_han_beg_of_per_gop', y='cas_on_han_beg_of_per_dem', c=colors, xlim=(0,3000000), ylim=(0,3000000))
plt.title('Cash On Hand At Start Of Period (April 1st)')
plt.xlabel('GOP Cash on Hand')
plt.ylabel('DEM Cash on Hand');

df.plot(kind='scatter',x='cas_on_han_clo_of_per_gop', y='cas_on_han_clo_of_per_dem', c=colors, xlim=(0,3000000), ylim=(0,3000000))
plt.title('Cash On Hand At End Of Period (June 30)')
plt.xlabel('GOP Cash on Hand')
plt.ylabel('DEM Cash on Hand');

# Charting The Correlations for PVI
sns.pairplot(df, x_vars=['gopvi'], y_vars='per_gop', size=7, aspect=0.7, kind='reg')
plt.title('The Correlations for PVI')
plt.xlabel('PVI')
plt.ylabel('Percent of Vote for GOP');

# Charting The Correlations for Beginning Cash on Hand (April 1)

sns.pairplot(df, x_vars=['gop_cash_adv_beg'], y_vars='per_gop', size=7, aspect=1, kind='reg')
plt.title('The Correlations for Cash on Hand (April 1)')
plt.ylabel('Percent of vote for GOP')
plt.xlabel('Cash on Hand Advantage');

# Charting The Correlations for Beginning Cash on Hand (April 1)

sns.pairplot(df, x_vars=['gop_cash_adv_end'], y_vars='per_gop', size=7, aspect=1, kind='reg')
plt.title('The Correlations for Cash on Hand (June 30)')
plt.ylabel('Percent of vote for GOP')
plt.xlabel('Cash on Hand Advantage');

# The Power Of Incumbency

df.boxplot(column='per_gop', by='can_inc_cha_ope_sea_gop')
plt.title('The Power Of Incumbency')
plt.xlabel('Seat Status (Incumbent, Challenger, Open)')
plt.ylabel('Percent of Vote for GOP');


df[df.can_inc_cha_ope_sea_gop == 0].per_gop.mean()
df[df.can_inc_cha_ope_sea_gop == 1].per_gop.mean()
df[df.can_inc_cha_ope_sea_gop == 2].per_gop.mean()
# Challenger Seat Mean GOP: 
#     38.938755980861245
# Open Seat Mean GOP: 
#     46.957894736842114
# Incumbent Seat Mean GOP: 
#     60.560262008733631

# The Power Of Partisanship

df[df.can_inc_cha_ope_sea_gop == 1].boxplot(column='per_gop', by='swing_seat')
plt.title('The Power Of Partisanship (Open Seats)')
plt.xlabel('Partisan Lean (Dem, Swing, GOP)')
plt.ylabel('Percent of Vote for GOP');

'''