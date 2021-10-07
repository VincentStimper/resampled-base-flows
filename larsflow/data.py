import os
from collections import Counter

import numpy as np
import pandas as pd
import torch


def load_miniboone(path):
    """
    Loads the UCI miniboone dataset
    :param path: String, path to the dataset
    :return: Tuple with training and test split as numpy array
    """
    # Load data
    data = np.load(path)

    # Train, validate, test split
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    # Normalization
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    # To tensor
    data_train = torch.tensor(data_train)
    data_validate = torch.tensor(data_validate)
    data_test = torch.tensor(data_test)

    return data_train, data_validate, data_test


def load_hepmass(path):
    """
    Load UCI hepmass dataset
    :param path: String, path to folder with training and test dataset
    :return: Tuple with training and test split as numpy array
    """
    # Load raw data
    data_train = pd.read_csv(filepath_or_buffer=os.path.join(path, '1000_train.csv'),
                             index_col=False)
    data_test = pd.read_csv(filepath_or_buffer=os.path.join(path, '1000_test.csv'),
                            index_col=False)

    # Gets rid of any background noise examples i.e. class label 0.
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    data_test = data_test.drop(data_test.columns[-1], axis=1)

    # Normalize data
    mu = data_train.mean()
    s = data_train.std()
    data_train = (data_train - mu) / s
    data_test = (data_test - mu) / s

    data_train, data_test = data_train.values, data_test.values

    # Remove any features that have too many re-occurring real values.
    i = 0
    features_to_remove = []
    for feature in data_train.T:
        # max_count = np.max(np.unique(feature, return_counts=True)[1])
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    ind = np.array([i for i in range(data_train.shape[1])
                    if i not in features_to_remove])
    data_train = data_train[:, ind]
    data_test = data_test[:, ind]

    # Get validation dataset
    N = data_train.shape[0]
    N_validate = int(N * 0.1)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    # To tensor
    data_train = torch.tensor(data_train)
    data_validate = torch.tensor(data_validate)
    data_test = torch.tensor(data_test)

    return data_train, data_validate, data_test


def load_power(path):
    # Load data
    data = np.load(path)

    rng = np.random.RandomState(42)

    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)

    # Add noise
    voltage_noise = 0.01 * rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data += noise

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    # Normalize data
    data_train, data_validate, data_test
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    # To tensor
    data_train = torch.tensor(data_train)
    data_validate = torch.tensor(data_validate)
    data_test = torch.tensor(data_test)

    return data_train, data_validate, data_test


def load_gas(path):
    # Load data
    with pd.read_pickle(path) as data:
        data.drop("Meth", axis=1, inplace=True)
        data.drop("Eth", axis=1, inplace=True)
        data.drop("Time", axis=1, inplace=True)

        # Remove columns with to high correlation
        def get_correlation_numbers(data):
            C = data.corr()
            A = C > 0.98
            B = A.sum(axis=1)
            return B

        B = get_correlation_numbers(data)
        while np.any(B > 1):
            col_to_remove = np.where(B > 1)[0][0]
            col_name = data.columns[col_to_remove]
            data.drop(col_name, axis=1, inplace=True)
            B = get_correlation_numbers(data)

        # Normalize data
        data = (data - data.mean()) / data.std()
        data_ = data.values

    # Train, validation, test split
    N_test = int(0.1 * data_.shape[0])
    data_test = data_[-N_test:]
    data_train = data_[0:-N_test]
    N_validate = int(0.1 * data_train.shape[0])
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    # To tensor
    data_train = torch.tensor(data_train)
    data_validate = torch.tensor(data_validate)
    data_test = torch.tensor(data_test)

    return data_train, data_validate, data_test


# Dictonary of UCI data loaders
uci_loader = {'miniboone': load_miniboone,
              'hepmass': load_hepmass,
              'power': load_power,
              'gas': load_gas}