# classify_rf.py
# 2/24/2018
# Dan Popp
#
# This file will classify the given IDS file using a random forest approach
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

import data_cleaning

DATA_ROOT_2018 = '/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/'
PICKLE_PATH = '/home/poppfd/College/ML_Cyber/ml-project/data/'


def load_2018_data(data_path):
    """
    Read in the entire 2018 dataset
    :param data_path: the path to the root data directory
    :return: a tuple for the full numpy arrays and labels
    """
    all_data = None
    all_labels = []

    for file in os.listdir(data_path):
        data, labels = get_data(os.path.join(DATA_ROOT_2018, file))

        if all_data is None:
            all_data = data
        else:
            all_data = np.concatenate((all_data, data))
        all_labels.append(labels)

    # Perform test/validation split
    data_train, data_test, labels_train, labels_test = train_test_split(all_data, all_labels, test_size=0.20)

    return data_train, data_test, labels_train, labels_test


def get_data(file):
    """
    Reads the csv file using pandas and returns the data and labels as numpy arrays
    :param file: The file to read from
    :return: a tuple of numpy arrays and the labels
    """

    filename = os.path.splitext(os.path.basename(file))[0] + '.pkl'
    pkl_path = os.path.join(PICKLE_PATH, filename)

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            data_np, labels_list = pickle.load(file)
    else:

        df = pd.read_csv(file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        data = df.drop('Label', axis=1)

        # Drop source port, source ip, and destination IP
        # TODO: Figure out how to handle target port number

        labels = df['Label']
        labels_list = labels.tolist()

        data_np = data.to_numpy(dtype=np.float32, na_value=0)

        data_np = data_cleaning.clean_np_data(data_np, labels_list)

        is_nan = np.any(np.isnan(data_np))
        is_finite = np.all(np.isfinite(data_np))
        print('Data is nan: %s' % str(is_nan))
        print('Data is finite: %s' % str(is_finite))

        # Normalize data
        data_np = normalize(data_np)

        with open(pkl_path, 'wb') as file:
            pickle.dump((data_np, labels_list), file)

    return data_np, labels_list


def normalize(array):
    """
    Will normalize each column of a numpy array between 0-1
    :param array: The data array
    :return: the normalized data
    """

    min = np.amin(array, axis=0)
    array -= min
    max = np.amax(array, axis=0)
    array /= (max + 1e-3)
    return array


def main():
    clf = RandomForestClassifier(max_depth=2, random_state=0, n_jobs=20)
    data_train, data_test, labels_train, labels_test = load_2018_data(DATA_ROOT_2018)
    clf.fit(data_train, labels_train)

    predictions = clf.predict(data_test)

    print(classification_report(labels_test, predictions))

    print('Done')


if __name__ == '__main__':
    main()
