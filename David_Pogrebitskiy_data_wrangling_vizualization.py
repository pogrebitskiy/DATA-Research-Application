#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 15:14:53 2021

@author: dpogrebitskiy
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from yellowbrick.target import FeatureCorrelation
from yellowbrick.features.rankd import Rank2D
from yellowbrick.features import rank1d, rank2d


# Reads in the main data csv file
df = pd.read_csv('lab_assignment.csv')

# Exports all the stock's tickers as a txt file
subset_df = df.groupby('TICKER', as_index = False)['GVKEY'].mean()

# Reads in the stock_info file, removes the unnecessary columns, drops the rows
# with NaN values, and resets the index.
stock_info = pd.read_csv('stock_info.csv')
stock_info = stock_info.drop(columns=['indfmt', 'consol', 'popsrc', 'datafmt',
                                      'curcd', 'costat', 'datadate', 'tic'],
                             inplace=False)
stock_info =stock_info.rename(columns = {'epspx':'eps'})
stock_info=stock_info.dropna()
stock_info = stock_info.reset_index()
stock_list = list(subset_df['TICKER'])

# Creates a new column for the stock price of the company that year.
# Creates a dictionary mapping each (GVKEY, YEAR) tuple to its correspoinding
# year's high stock price. Appends that value to each executive in the main
# dataframe.
df['stock_price'] = np.nan
stock_price_dict = {}
for i in range(0, stock_info.shape[0]):
    stock_price_dict[(stock_info['gvkey'][i], int(stock_info['fyear'][i]))] = stock_info['prch_c'][i] 
for i in range(0, df.shape[0]):
    df['stock_price'][i] = stock_price_dict.get((df['GVKEY'][i], df['YEAR'][i]))




# Does the same thing as the stock price except for Earnings per share
df['eps'] = np.nan
eps_dict = {}
for i in range(0, stock_info.shape[0]):
    eps_dict[(stock_info['gvkey'][i], int(stock_info['fyear'][i]))] = stock_info['eps'][i] 
for i in range(0, df.shape[0]):
    df['eps'][i] = eps_dict.get((df['GVKEY'][i], df['YEAR'][i]))

# Calculates the Price to earnings ratio for each stock
df['pe_ratio'] = np.nan
for i in range(0, df.shape[0]):
    df.loc[i, 'pe_ratio'] = round(df.loc[i, 'stock_price'] / df.loc[i, 'eps'], 5)
    
    

stock_info['roa'] = np.nan
df['roa'] = np.nan
for i in range(0, stock_info.shape[0]):
    stock_info.loc[i, 'roa'] = round(stock_info.loc[i, 'ni'] / stock_info.loc[i, 'at'], 5)
roa_dict = {}
for i in range(0, stock_info.shape[0]):
    roa_dict[(stock_info['gvkey'][i], int(stock_info['fyear'][i]))] = stock_info['roa'][i] 
for i in range(0, df.shape[0]):
    df['roa'][i] = roa_dict.get((df['GVKEY'][i], df['YEAR'][i]))
    
stock_info['roe'] = np.nan
df['roe'] = np.nan
for i in range(0, stock_info.shape[0]):
    stock_info.loc[i, 'roe'] = round(stock_info.loc[i, 'ni'] / stock_info.loc[i, 'seq'], 5)
roe_dict = {}
for i in range(0, stock_info.shape[0]):
    roa_dict[(stock_info['gvkey'][i], int(stock_info['fyear'][i]))] = stock_info['roe'][i] 
for i in range(0, df.shape[0]):
    df['roe'][i] = roa_dict.get((df['GVKEY'][i], df['YEAR'][i]))


clean_df = pd.read_csv('clean_lab_assignment.csv')
'''
# Create a dictionary with each executive tied to their unique ID
id_df = df.groupby('EXEC_FULLNAME', as_index = False)['EXECID'].mean()
execs = {}
for i in range(0, id_df.shape[0]):
    execs[int(id_df.loc[i, 'EXECID'])] = id_df.loc[i, 'EXEC_FULLNAME']



# Creates a dictionary mapping each company to its corresponding GVKEY
company_df = df.groupby('CONAME', as_index=False)['GVKEY'].mean()
companies ={}
for i in range(0, company_df.shape[0]):
    companies[int(company_df.loc[i, 'GVKEY'])] = company_df.loc[i, 'CONAME']
'''

clean_df = pd.read_csv('clean_lab_assignment.csv')
clean_df = pd.get_dummies(clean_df, columns=['GENDER'])
clean_df = clean_df.drop(columns=['EXEC_FULLNAME', 'CONAME', 'YEAR',
                                  'TICKER', 'CO_PER_ROL', 'GVKEY', 'EXECID'])
clean_df.replace([np.inf], np.nan, inplace=True)
clean_df = clean_df.dropna()


plt.figure(figsize=(12,8))
sns.regplot(x = 'roa', y = 'stock_price', data = clean_df)
plt.show()

plt.figure(figsize=(12,8))
sns.regplot(x = 'roe', y = 'stock_price', data = clean_df)
plt.show()

plt.figure(figsize=(12,8))
sns.regplot(x = 'roa', y = 'TDC1', data = clean_df)
plt.show()

plt.figure(figsize=(12,8))
sns.regplot(x = 'roe', y = 'TDC1', data = clean_df)
plt.show()

'''
execpay_corr=clean_df[['TDC1', 'TDC1_PCT','EXECRANKANN', 'stock_price',
 'eps','pe_ratio']].corr()
sns.heatmap(execpay_corr, square=True, annot=True, fmt='.2f', cmap='viridis')
'''


target = clean_df['TDC1']
features = clean_df.drop(columns=['TDC1', 'TDC1_PCT', 'SALARY'])
feature_names = list(features.columns)
visualizer = FeatureCorrelation(labels = feature_names)
visualizer.fit(features, target)
plt.title("Features Correlation with Total Compensation")
plt.tight_layout()
plt.show()
visualizer.poof()

target = clean_df['stock_price']
features = clean_df.drop(columns=['pe_ratio', 'stock_price'])
feature_names = list(features.columns)
visualizer = FeatureCorrelation(labels = feature_names)
visualizer.fit(features, target)
plt.title('Features Correlation with stock price')
plt.tight_layout()
plt.show()
visualizer.poof()
      
target = clean_df['stock_price']
features = clean_df.drop(columns=['SALARY', 'TDC1', 'TDC1_PCT'])
feature_names = list(features.columns)
visualizer = FeatureCorrelation(labels = feature_names)
visualizer.fit(features, target)
plt.title('Features Correlation with Salary')
plt.tight_layout()
plt.show() 
visualizer.poof()

clean_df = clean_df.rename(columns={'EXECRANKANN' : 'Rank', 'GENDER_MALE' : 'Male', 'GENDER_FEMALE':'Female'})
data = clean_df
feat = ['SALARY', 'AGE', 'Rank', 'TDC1', 'TDC1_PCT', 'stock_price',
        'eps', 'pe_ratio', 'roa', 'roe', 'Male', 'Female']
X = data[feat].to_numpy()
y = data.to_numpy()
plt.figure(figsize=(12,8))
viz = Rank2D(features=feat, algorithm = 'pearson')
viz.fit(X, y)
viz.transform(X)
plt.title('Pearson Ranking of 12 Features')
plt.tight_layout()
viz.poof()
plt.show()


roa_df = df.groupby('YEAR', as_index=False)['roa', 'TDC1', 'eps', 'roe'].mean().drop(columns=['YEAR'])
sns.regplot(x='TDC1', y='eps', data = roa_df)


total_corr=roa_df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(total_corr, vmax=1.0, vmin=0.0, square=True, annot=True,
            fmt='.2f', cmap = 'Oranges')
plt.title('Executive Data')
plt.show()


