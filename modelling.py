import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from statsmodels.tsa.arima_model import ARIMA
import feature_engineering as fe
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
'''
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


def fit_ada_boost(features, labels,reg=False):
    param_dist = {
        'n_estimators': [50, 100, 300, 1000, 1500, 2000],
        'learning_rate': [0.01, 0.05, 0.1, 0.3, 1]
        # 'loss': ['linear', 'square', 'exponential']
    }
    if reg:
        param_dist['loss'] = ['linear', 'square', 'exponential']
        pre_gs_inst = RandomizedSearchCV(AdaBoostRegressor(),
                                         param_distributions=param_dist,
                                         cv=3,
                                         n_iter=10,
                                         n_jobs=-1)
    else:
        pre_gs_inst = RandomizedSearchCV(AdaBoostClassifier(),
                                     param_distributions=param_dist,
                                     cv=3,
                                     n_iter=10,
                                     n_jobs=-1)

    model = pre_gs_inst.fit(features, labels)
    return model


def fit_random_forest(features, labels, reg=False):
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    {'bootstrap': [True, False],
     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'n_estimators': [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
    rf = RandomForestClassifier()
    if reg:
        rf = RandomForestRegressor()
    model = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                               random_state=42,
                               n_jobs=-1)
    # RandomForestRegressor(n_estimators=1000, random_state=42, min_impurity_decrease=5, max_depth=5000)

    model = model.fit(features, labels)
    return model


def fit_SVM(features, labels):
    svm = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(svm, param_grid, n_jobs=8, verbose=1)
    model = grid_search.fit(features, labels)
    return model


def fit_KNN(features, labels, withTuning):
    if withTuning:
        param_grid = {'n_neighbors': list(range(1, 30)), 'p': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        model = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='accuracy', n_jobs=8, verbose=1, cv=2)

    else:
        model = KNeighborsClassifier(n_neighbors=14)

    model.fit(features, labels)

    return model

def fit_gradient_boosting(features, labels, withTuning):
    if withTuning:
        param_grid = {'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
                      'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750],
                      'max_depth': [2, 3, 4, 5, 6, 7]}

        model = GridSearchCV(
            estimator=GradientBoostingClassifier(min_samples_split=2,
                                                 min_samples_leaf=1, subsample=1, max_features='sqrt', random_state=10),
            param_grid=param_grid, scoring='accuracy', n_jobs=4, iid=False, cv=5)
    else:
        model = GradientBoostingClassifier(learning_rate=0.001, n_estimators=500, max_depth=3, min_samples_split=2,
                                           min_samples_leaf=2, subsample=1, max_features='sqrt', random_state=10)

    model.fit(features, labels)

    return model


def fit_arima(all_data, col_name, test_lines_num, classification, ord=[2, 1, 2]):
    data = pd.DataFrame()
    data["Date"] = pd.date_range(start='1/1/1979', periods=len(all_data), freq='D')
    data[col_name] = all_data[col_name]
    data = data.set_index('Date')
    model = ARIMA(data[:-test_lines_num], order=(ord[0], ord[1], ord[2]))
    result = model.fit()
    print(result.summary())
    test = data.tail(test_lines_num)
    test["prediction"] = result.forecast(test_lines_num)[0]
    if classification:
        test.to_csv("arima_results.csv")
        return fe.generate_y(test, "prediction")
    return test["prediction"].tolist()


def fit_logistic_regression(features, labels):    
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
    grid_search = GridSearchCV(LogisticRegression(), param_grid, n_jobs = 8, verbose=1)
    model = grid_search.fit(features, labels)
    return model


def get_results(y_true,pred):
    return (y_true == pred).value_counts()

def get_accuracy_score(y_true, pred):
    return accuracy_score(y_true, pred)

# Scale back and return the true mean square error between label and prediction
def get_mse(y_true, pred, scaler_label):
    y_true_scale_back, pred_scale_back = scaler_label.inverse_transform(y_true.values.reshape(-1, 1)), scaler_label.inverse_transform(pred.values.reshape(-1, 1))
    return mean_squared_error(y_true_scale_back, pred_scale_back)

# It prepares the training and testing data for FNN by 
# 1. Remove the additional column ['Unnamed: 0']
# 2. Replace the 2 with 0 in label since the output layer of neural network only accept [0, 1]
def reconstruct(xTrain, xTest, yTrain, yTest):
    return xTrain.drop(columns=['Unnamed: 0'], axis=1), xTest.drop(columns=['Unnamed: 0'], axis=1), yTrain.replace(2, 0), yTest.replace(2, 0)
'''
def fit_FNN(features, labels, withTuning=False):
    # Some common parameters for models
    epochs = 100
    
    optimizer = keras.optimizers.Adam(0.001)

    if withTuning:
        def create_FNN(first_layer_neurons, second_layer_neurons):
            model = keras.Sequential([
                    layers.Dense(first_layer_neurons, activation='relu', input_shape=[len(features.columns)]),
                    layers.Dense(second_layer_neurons, activation='relu'),
                    layers.Dense(1, activation='sigmoid')])
            optimizer = keras.optimizers.Adam(0.001)
            model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
            return model
        
        param_grid = {'first_layer_neurons': [8, 16, 32, 64], 
                      'second_layer_neurons': [8, 16, 32, 64]}
    
        model = KerasClassifier(build_fn=create_FNN, epochs=epochs, batch_size=10, initial_epoch=0, verbose=0)
        model = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=-1, random_state=42, cv=3, n_iter=5)
        model.fit(features, labels)
        return model
    else:
        model = keras.Sequential([
                layers.Dense(16, activation='relu', input_shape=[len(features.columns)]),
                layers.Dense(32, activation='relu'),
                layers.Dense(1, activation='sigmoid')])
        
        model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])
        # Apply early stop to prevent overfitting
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        model.fit(features, labels, epochs = epochs, validation_split = 0.2, verbose = 0, callbacks=[early_stop])
        return model

# Mainly for Recurrent Neural Network
# Repack the features into shape [Samples, History, Features]
# Repack the labels into shape [Labels, 1]
def univariate_data(features, label, start_index, end_index, history_size, target_size, test_size):
    features_list = []
    labels_list = []
    num_of_features = len(features.columns)
    num_of_timestamp = features.shape[0]
    
    # If the end index is not defined, simply pick the last one
    if end_index is None:
        end_index = num_of_timestamp-target_size
    
    # Start iterate from the end item of first pack
    start_index = start_index+history_size
    for i in range(start_index, end_index):
        # Range indices for each timeframe. Should be history_size rows
        sample_indices = range(i - history_size, i)
        
        # Picks from the features dataframe and appends it in a list
        sample_features = np.array(features.iloc[sample_indices][:].values)
        sample_features = np.reshape(sample_features, (history_size, num_of_features))
        features_list.append(sample_features)
 
        # Range indice for prediction. Should be target_size rows
        target_index = i+target_size
        
        # Picks from the label dataframe and appends it in a list
        sample_label = label.loc[target_index]
        labels_list.append(sample_label)
    
    # Perform split
    features_list = np.array(features_list)
    labels_list = np.array(labels_list)
    num_of_train_size = (int)((1-test_size)*features_list.shape[0])
    
    return features_list[:num_of_train_size], features_list[num_of_train_size:], labels_list[:num_of_train_size], labels_list[num_of_train_size:]


# Rescale the selected column with MaxMinScaler
def min_max_scaler(df, col_name):
    scaler = MinMaxScaler()
    df[col_name] = scaler.fit_transform(df[col_name].values.reshape(-1, 1))
    return df[col_name], scaler


# It prepares the training and testing data for RNN by 
# 1. Label 'Close', Features: Others
# 2. Rescale the label with MinMaxScaler
# 3. Transforms the raw data to the sequential data that contains history of prices as one row of features
def transform_and_rescale(data, history_size):
    features = data.drop(columns=['Close'], axis=1)
    label, scaler_label = min_max_scaler(data, 'Close') # Select the label as 'Close' and rescale the label with MinMaxScaler
    xTrain, xTest, yTrain, yTest = univariate_data(features, label, start_index=1, end_index=None, history_size=history_size, target_size=1, test_size=0.2)
    return xTrain, xTest, yTrain, yTest, scaler_label

def fit_LSTM_reg(features, labels):
    BATCH_SIZE = 200
    BUFFER_SIZE = 10000
    train_univariate = tf.data.Dataset.from_tensor_slices((features, labels))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    
    lstm_model = tf.keras.models.Sequential([
        layers.LSTM(64, input_shape=features.shape[-2: ]), # The input shape is [Timestamp X features]
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    lstm_model.compile(optimizer='adam', loss='mae')
    EVALUATION_INTERVAL = 10
    EPOCHS = 15
    
    history = lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL)
    return lstm_model
'''