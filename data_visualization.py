'''
Implement generic methods for the following:
- Correlation matrix of the S&P closing price vs all other closing prices
- Charts of returns calculated in feature_engineering.py

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

#Function to generate decomposition graph
#index: source from Data file, eg. 'SP_1d_10y.csv'
#col: 'Close','Open','Volume',...
def draw_time_series(index, col):
    data = pd.read_csv('data/'+index)
    name = index.split('.')[0]
    data = data.set_index('Date')
    decom = seasonal_decompose(data[col], freq=int(len(data)/10))
    decom.plot()
    plt.savefig('Charts/'+name+'_'+col+'_decom.png')
    
