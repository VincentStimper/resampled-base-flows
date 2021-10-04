import os

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
    data = np.loadtxt(path, skiprows=1)

    # Remove background events
    data = data[:36499]

    # Remove outliers
    indices = (data[:, 0] < -100)
    data = data[~indices]

    # Remove any features that have too many re-occuring real values.
    i = 0
    features_to_remove = []
    for feature in data.T:
        max_count = np.max(np.unique(feature, return_counts=True)[1])
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data = data[:, [i for i in range(data.shape[1]) if not i in features_to_remove]]

    # Train test split
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    data_train = data[0:-N_test]

    # Normalization
    mu = data_train.mean(axis=0)
    s = data_train.std(axis=0)
    data_train = (data_train - mu) / s
    data_test = (data_test - mu) / s

    return torch.tensor(data_train), torch.tensor(data_test)


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
        max_count = np.max(np.unique(feature, return_counts=True)[1])
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array(
        [i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array(
        [i for i in range(data_test.shape[1]) if i not in features_to_remove])]

    return torch.tensor(data_train), torch.tensor(data_test)


# Dictonary of UCI data loaders
uci_loader = {'miniboone': load_miniboone,
              'hepmass': load_hepmass}