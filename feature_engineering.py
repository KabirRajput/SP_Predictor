'''
- May be interesting: Interest rates: https://quant.stackexchange.com/questions/20911/how-to-predict-daily-range-of-forex
Feature transformation:
- For every S&P day make Close = next Open
- Generate Y: BUY, SELL, HOLD at time t signals if p(t + 1) is higher, lower or the same as p(t), lets call it "Signal"
- Put together the S&P OHLC, S&P vol and closing prices of all the potential features in one file
- Normalize the data: Calculate daily log returns on the price data
- Put all of the above in one file

Data integration:
- It integrates all data S&P500, DJI, FTSE, N225, GOLD, and exchange rates as a single csv file.
- Keeps in mind that it's just raw data, meaning that missing data has not been handled. The missing data can be due to different rest days across different markets. (US, UK, Japan)

Feature selection:
- Choose the top 4 indeces using:
- Perform correlation matrix analysis
- PCA
- Make a call in case of conflict between the above 2 methods i.e. pick what we think is better


'''

import pandas as pd
import download_data as dd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
# error for mac: https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import ta
from sklearn.preprocessing import MinMaxScaler


class Signals:

    def BUY(self): return int(1)

    def SELL(self): return int(2)

    def SIGNAL(self):
        return "Signal"


SIGNALS = Signals()


def generate_y(df, col_name):
    diff = df[col_name].diff(periods=-1)
    diff.values[diff.values > 0] = SIGNALS.SELL()
    diff.values[diff.values <= 0] = SIGNALS.BUY()
    return diff


def generate_y_reg(df, col_name):
    y = df[col_name].shift(-1)
    df2 = y.to_frame(name='Y')
    y = log_returns(df2, 'Y')
    return y


def log_returns(df, col_name):
    ratio = df[col_name] / df[col_name].shift(1)
    return np.log(ratio)


def standardize(df, col_name):
    col = df[col_name]
    mean = col.mean()
    std = col.std()
    return ((col - mean) / std)


def data_integrate():
    data_1d_5y = pd.read_csv('data_1d_5y.csv')  # Please type the path of the original data by Danish
    data_1d_5y['Date'] = pd.to_datetime(data_1d_5y['Date'], dayfirst=True)

    data_1d_5y.set_index('Date', inplace=True)
    data_1d_5y.drop(['Unnamed: 0'], axis=1, inplace=True)
    # data_1d_5y.head(5)

    dji = pd.read_csv('DJI_1d_5y.csv')  # US data
    dji['Date'] = pd.to_datetime(dji['Date'], dayfirst=True)
    dji = dji[['Date', 'Close']]
    dji.rename(columns={'Close': 'DJI_Close'}, inplace=True)
    dji.set_index('Date', inplace=True)
    # dji.head(10)
    data_1d_5y = data_1d_5y.join(dji)
    # data_1d_5y.head(5)

    ftse = pd.read_csv('FTSE_1d_5y.csv')  # UK data
    ftse['Date'] = pd.to_datetime(ftse['Date'], dayfirst=True)
    ftse = ftse[['Date', 'Close']]
    ftse.rename(columns={'Close': 'FTSE_Close'}, inplace=True)
    ftse.set_index('Date', inplace=True)
    # ftse.head(10)
    data_1d_5y = data_1d_5y.join(ftse)
    # data_1d_5y.head(5)

    gold = pd.read_csv('GOLD_1d_5y.csv')  # Gold Price
    gold['Date'] = pd.to_datetime(gold['Dates'], dayfirst=True)
    gold = gold[['Date', 'Close']]
    gold.rename(columns={'Close': 'GOLD_Close'}, inplace=True)
    gold.set_index('Date', inplace=True)
    # gold.head(5)
    data_1d_5y = data_1d_5y.join(gold)
    # data_1d_5y.head(5)

    N225 = pd.read_csv('N225_1d_5y.csv')  # Japan Data
    N225['Date'] = pd.to_datetime(N225['Date'], dayfirst=True)
    N225 = N225[['Date', 'Close']]
    N225.rename(columns={'Close': 'N225_Close'}, inplace=True)
    N225.set_index('Date', inplace=True)
    # N225.head(5)
    data_1d_5y = data_1d_5y.join(N225)
    # data_1d_5y.head(5)

    currency = ['CNY', 'EUR', 'GBP', 'JPY']
    for cur in currency:
        currency_df = pd.read_csv(f"USD{cur}_1d_5y.csv")  # Currency
        currency_df['Date'] = pd.to_datetime(currency_df['Date'], dayfirst=True)
        currency_df = currency_df[['Date', 'Close']]
        currency_df.rename(columns={'Close': f"USD{cur}_Close"}, inplace=True)
        currency_df.set_index('Date', inplace=True)
        data_1d_5y = data_1d_5y.join(currency_df)
    # data_1d_5y.head(5)
    data_1d_5y[data_1d_5y.isna().any(axis=1)]
    data_1d_5y.reset_index(inplace=True)
    data_1d_5y.to_csv('data_1d_5y.csv', index=True)


def generate_data_reg():
    data = pd.read_csv(dd.file_name("data"))
    data_norm = pd.read_csv(dd.file_name("data_normalized"))

    data_norm = data_norm.drop(["Signal"], axis=1)
    y = generate_y_reg(data, 'Close').shift(-1)
    data_norm.insert(data_norm.columns.get_loc('Date') + 1, 'Y', y)
    data_norm = data_norm.drop(["Unnamed: 0"], axis=1)  # data_normalizaed data somehow got an Index row
    data_norm.dropna().to_csv(dd.file_name("data_reg"), index=False, float_format='%.9f')


# correlation matrix
def corr_matrix():
    df = pd.read_csv('data_1d_5y.csv', index_col=0)
    df = df.ffill(axis=0)
    # all close prices
    df = df.filter(regex="Close")
    matrix = df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
    return (matrix)


def pca(df):
    # it will be more useful when we have set the number of PC or explain_variance_ratio
    # this input df should be filled df after df = df.ffill(axis=0)
    pca = PCA()  # there is a parameter to decide the number of PCs or explain variance ratio
    pca.fit(df)

    explained_variance_ratio = pca.explained_variance_ratio_
    eigen_values = pca.explained_variance_
    eigen_vectors = pca.components_

    df1 = pd.DataFrame({
        'eigen_values': eigen_values,
        'explained_variance_ratio': explained_variance_ratio})
    df2 = pd.DataFrame(eigen_vectors)
    df3 = pd.concat([df1, df2], axis=1)
    df3.to_csv(r'eigen_values.csv', index=False)


def feature_importance(df):
    # this input df should be filled df after df = df.ffill(axis=0)    
    y = df['Signal']
    X = df.drop(['Date', 'Signal'], axis=1)

    model = ExtraTreesClassifier(n_estimators=50)  # default is 10 and will change to 100 in the new version
    model.fit(X, y)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # plot graph of feature importances for better visualization
    feature_importance.plot(kind='barh')
    plt.title('feature_importance')
    plt.show()

    # write feature_importances.csv to data directory
    df = pd.DataFrame(feature_importance, columns=['importance'])
    df['rank'] = df['importance'].rank(ascending=False)
    df.to_csv(r'feature_importance.csv')


# Non-normalized data should be fed to this function
def add_technical_indicators(raw_data):
    return ta.add_all_ta_features(raw_data, "Open", "High", "Low", "Close", "Volume", fillna=True)


def normalize_data(file_name, normal_file_name):
    data = pd.read_csv("data/" + file_name)
    data = data.ffill(axis=0)
    for col in data.columns:
        if col != "Date" and col != "Volume" and col != "Signal":
            data[col] = log_returns(data, col)
            data[col] = data[col].replace(to_replace=0, method='ffill')

    data.to_csv("data/" + normal_file_name)


def ordered_train_test_split(data, y_col_name, ratio=80):
    if (ratio > 99 or ratio < 1):
        print("ratio must be between 1 and 99")
        return -1
    train = data.head(int(len(data) * (ratio / 100)))
    test = data.iloc[-(int(len(data) * ((100 - ratio) / 100))):]
    return train.drop([y_col_name], axis=1), test.drop([y_col_name], axis=1), train[y_col_name], test[y_col_name]

# Apply the MinMaxScaler to normalize features, but not label
# Save it. Example: file_name="data_1d_10y"  normal_file_name="data_minmax_1d_10y"
def apply_min_max_scaler(file_name, normal_file_name):
    data = pd.read_csv("data/" + file_name)
    data = data.drop(['Volume', 'Unnamed: 0', 'Signal', 'Date'], axis=1)
    data = data.ffill(axis=0)
    for col in data.columns:
        if col != 'Close':
            scaler=MinMaxScaler()
            data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
            data[col] = data[col].replace(to_replace=0, method='ffill')
    data.to_csv("data/" + normal_file_name, index=False)
    
#Function to produce dataframe with data-related columns
#data: dataframe with 'Date' column
#var: "year","month","weekday","week","day"
def get_date_value(data, var):
    date = {'year':'%Y','month':'%B','weekday':'%A','day':'%d','week':'%U'}
    l = []
    for i in range(len(data)):
        l.append(datetime.strptime(data.Date[i],'%d-%m-%y').strftime(date[var]))
    l = pd.DataFrame(l, columns=[var])
    data_date = l[var].str.get_dummies()
    data = pd.concat([data,data_date],axis=1)
    return(data)


