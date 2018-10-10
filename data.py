import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd
import sklearn.preprocessing as sk
import math
import numpy as np

TRAINING_DATA_SIZE = .8

def read():
    dataframe = pd.read_csv('data_stocks.csv')
    dataframe = dataframe.drop(['DATE'], 1)
    return dataframe

def get_dimens_of(dataframe: pd.DataFrame):
    return (dataframe.shape[0], dataframe.shape[1])

def get_subset_indices(data_length):
    train_start = 0
    train_end = math.floor(TRAINING_DATA_SIZE * data_length)

    test_start = train_end
    test_end = data_length

    return (train_start, train_end, test_start, test_end)

def get_subset(numpy_data: np.ndarray, start_index, end_index):
    return numpy_data[np.arange(start_index, end_index), :]

def get_training_and_test_subsets(dataframe: pd.DataFrame):
    (length, depth) = get_dimens_of(dataframe)
    (train_start, train_end, test_start, test_end) = get_subset_indices(length)
    numpy_data = dataframe.values
    return (get_subset(numpy_data, train_start, train_end), get_subset(numpy_data, test_start, test_end))

def get_inputs_and_results_of(set):
    # data:
    # S&P(t + 1); Stock1(t); Stock2(t); ...
    return (set[:, 1:], set[:, 0])


scaler = sk.MinMaxScaler()

def fit_scaler_on(data):
    return scaler.fit(data)

def scale(data):
    return scaler.transform(data)

def unscale(data):
    return scaler.inverse_transform(data)


def get_completely_preprocessed_inputs_and_results_of(dataframe: pd.DataFrame):
    (training_data, test_data) = get_training_and_test_subsets(dataframe)

    fit_scaler_on(training_data)
    training_data = scale(training_data)
    test_data = scale(test_data)

    (training_inputs, training_results) = get_inputs_and_results_of(training_data)
    (test_inputs, test_results) = get_inputs_and_results_of(test_data)

    return (training_inputs, training_results, test_inputs, test_results)
