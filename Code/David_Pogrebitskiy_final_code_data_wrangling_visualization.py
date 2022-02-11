#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research Problem: 
   A publicly traded company is trying to determine various aspects of 
   executive pay and its impact on firm performance, as well as other
   indicators. This company has compiled a collection of data regarding
   executive pay across various publicly traded companies. They would like you
   to analyze this data set, and offer insights and recommendations. 
   
Data Wrangling:
    Pair the provided dataset with other data (weather,stock,economicdata,etc).
    Explain how the addition of your chosen data set will help an analyst
    provide additional insights into the problem above. You must also merge 
    the dataset you found with the one provided to you in this assignment to
    create a single dataset.
    
Data Visualization:
    Design a few visualizations that capture the story behind executive pay.
    These could be any plots of your choosing, or a full on dashboard.
    
By: David Pogrebitskiy
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def export_gvkey(df):
    '''
    This function extracts the unique stocks from the provided data and exports
    it as a txt file to be used to gather data in the future.

    Parameters
    ----------
    df : Pandas DataFrame
        The datafram that you are extracting the stock names from

    Returns
    -------
    None

    '''
    # Groups the dataframe by ticker and then exports the data as a txt
    subset_df = df.groupby('GVKEY', as_index = False)['GVKEY'].mean()
    subset_df['GVKEY'].to_csv('stock_keys.txt', header= False, index=False)



def plot_regression(df, ind_var, dep_var):
    '''
    This function takes in data and two variable names a plots their corresponding
    linear regression

    Parameters
    ----------
    df : Pandas DataFrame
        This is the main dataframe that contains the data you're plotting
    ind_var : String
        The is the name of the Independent variable in your visualization
    dep_var : String
        This is the name of the Dependant variable in your visualization

    Returns
    -------
    None. Creates a visualization

    '''
    # Plots the regression plot and creates a corresponding title to the
    # parameters given
    plt.figure(figsize=(12,8))
    sns.regplot(x = ind_var, y = dep_var, data = df)
    title = ind_var + ' vs. ' + dep_var
    plt.title(title)
    plt.savefig(title + '.png')
    plt.show()
    


if __name__ == '__main__':
    
    '''
    This is the data wrangling portion of the assignment
    '''
    
    # Reads in the csv file provided for the assignment
    exec_df = pd.read_csv('lab_assignment.csv')
    
    # Extracts each unique stock ticker and exports it to a txt file
    gv_keys = pd.unique(exec_df['GVKEY'])
    np.savetxt('gv_keys.txt', gv_keys, fmt='%i')
    
    # Reads in the csv file exported from WRDS
    stock_info = pd.read_csv('David_Pogrebitskiy_stock_info_data_wrangling.csv')
    
    # Cleans up the new stock info dataframe by removing unnecessary columns,
    # renaming columns for legibility, dropping rows with NaN values,
    # and resetting the index to account for the deleted rows.
    stock_info = stock_info.drop(columns=['indfmt', 'consol', 'popsrc',
                                          'datafmt','curcd', 'costat',
                                          'datadate'], inplace=False)
    stock_info =stock_info.rename(columns = {'epspx':'eps'})

    # Merges the stock_info file with the main exec file by gvkey and year pair
    exec_df = exec_df.merge(stock_info, left_on=['GVKEY', 'YEAR'],
                       right_on=['gvkey', 'fyear'], how = 'left')
    exec_df = exec_df.drop(columns=['gvkey', 'fyear'])
    exec_df = exec_df.rename(columns={'prch_c' : 'stock_price'})
    
    # Calculates Return on Assets, Price to Earnings Ratio, Return on 
    # Equity, and bonuses for each row and makes a new column for each
    exec_df['roa'] = exec_df['ni'].div(exec_df['at'].values)
    exec_df['roe'] = exec_df['ni'].div(exec_df['seq'].values)
    exec_df['pe_ratio'] = exec_df['stock_price'].div(exec_df['eps'].values)
    exec_df['bonus_etc'] = exec_df['TDC1'] - exec_df['SALARY']
    
    
    # Exports the newly cleaned and merged data to a csv file, marking the 
    # success of the data wrangling portion of this assignment
    exec_df.to_csv('cleanedAndMergedData.csv', index=False)
    clean_df = pd.read_csv('David_Pogrebitskiy_cleanedAndMergedData_data_wrangling.csv')
    
    '''
    This is the data visualization portion of the assignment
    '''
    
    # Converts the dataset to a form that is ready to be visualized by
    # converting the categorical data to a one-hot representation and dropping
    # columns that have no significance in visualization
    numerical_df = pd.get_dummies(clean_df, columns=['GENDER'])
    numerical_df = numerical_df.drop(columns=['EXEC_FULLNAME', 'CONAME',
                                  'TICKER', 'CO_PER_ROL', 'GVKEY', 'EXECID'])
    numerical_df = numerical_df.rename(columns={'EXECRANKANN' : 'Rank',
                                                'GENDER_MALE' : 'Male',
                                                'GENDER_FEMALE':'Female',
                                                'TDC1' : 'Compensation',
                                                'TDC1_PCT' : 'Comp Change'})

    
    # Creates two new dataframes grouped by year and gender, then calcultes the mean
    # of each varible by those groupings
    year_data = numerical_df.groupby('YEAR', as_index=False).mean()
    
    
    # Plots the regression between year and pct of females in the given data set
    # showing that the percentage is growing year to year
    plot_regression(year_data, 'YEAR', 'Female')
    plot_regression(year_data, 'YEAR', 'Male')

    
    # Plots the regression between Year vs total comp and salary showing
    # that the average salary is growing year to year
    plot_regression(year_data, 'YEAR', 'Compensation')
    plot_regression(year_data, 'YEAR', 'SALARY')
    plot_regression(year_data, 'YEAR', 'bonus_etc')

    # Creates a categorical point plot that shows the breakdown of salary vs rank
    # based on gender. Shows that while rank has a large infuence on salary, gender
    # does not
    sns.catplot(x='EXECRANKANN', y='SALARY', hue='GENDER', data=clean_df,
                kind='point', height=6, aspect=2, palette ='CMRmap')
    plt.title('Rank vs Salary by Gender')
    plt.xlabel('Rank')
    plt.tight_layout()
    plt.savefig('salary_rank_catplot.png')
    plt.show()
    
    # Creates a regression plot that pairs with the above categorical bar plot
    # To show that rank and salary are negatively correlated
    plot_regression(numerical_df, 'Rank', 'SALARY')
    
    # Plots a bar plot showing the average salary of a man vs a woman on average
    # and during just 2020. Shows
    # that although there seems to be no clear conclusion in the catplot, 
    # the difference is still there in the averages.
    plt.figure(figsize=(12,8))
    sns.barplot(x='GENDER', y='SALARY', data=clean_df.query('YEAR == 2010'),
                palette='nipy_spectral')
    plt.title('Average Salary by Gender in 2010')
    plt.ylim(0,900)
    plt.tight_layout()
    plt.savefig('mean_sal_gender_2010.png')
    plt.show()
    
    plt.figure(figsize=(12,8))
    sns.barplot(x='GENDER', y='SALARY', data=clean_df.query('YEAR == 2020'),
                palette='cubehelix')
    plt.title('Average Salary by Gender in 2020')
    plt.ylim(0,900)
    plt.tight_layout()
    plt.savefig('mean_sal_gender_2020.png')
    plt.show()
    
    # plots a KDE and regression jointplot between roa and compensation
    plt.figure(figsize=(12,8))
    sns.jointplot(x='Compensation', y = 'roa', data=numerical_df, kind='reg')
    plt.title('ROA vs Compensation')
    plt.tight_layout()
    plt.savefig('roa_comp_join.png')
    plt.show()
    
    # plots a KDE and regression jointplot between eps and compensation
    plt.figure(figsize=(12,8))
    sns.jointplot(x='Compensation', y = 'eps', data=numerical_df, kind='reg')
    plt.title('EPS vs Compensation')
    plt.tight_layout()
    plt.savefig('eps_comp_join.png')
    plt.show()
    
    # plots a KDE and regression jointplot between roa and salary
    plt.figure(figsize=(12,8))
    sns.jointplot(x='SALARY', y = 'roa', data=numerical_df, kind='reg')
    plt.title('ROA vs SALARY')
    plt.tight_layout()
    plt.savefig('roa_salary_join.png')
    plt.show()
    
    # plots a KDE and regression jointplot between eps and salary
    plt.figure(figsize=(12,8))
    sns.jointplot(x='SALARY', y = 'eps', data=numerical_df, kind='reg')
    plt.title('EPS vs SALARY')
    plt.tight_layout()
    plt.savefig('eps_salary_join.png')
    plt.show()
    
    
    # Plots a heatmap that represents all the pairwise correlations between
    # all the variables. Sums up findings from assignment.
    plt.figure(figsize=(12,8))
    feature_corr = numerical_df[['SALARY', 'Rank', 'Compensation', 'eps',
                                 'stock_price', 'roa', 'roe', 'pe_ratio',
                                 'bonus_etc' ]].corr()
    sns.heatmap(feature_corr, square=True, annot=True, fmt='.2f')
    plt.title('Pairwise Correlations between Variables')
    plt.savefig('Heatmap.png')
    plt.show()



    
    
    
    
    
    
    