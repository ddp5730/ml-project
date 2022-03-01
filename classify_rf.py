# classify_rf.py
# 2/24/2018
# Dan Popp
#
# This file will classify the given IDS file using a random forest approach
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

import data_cleaning

DATA_FILE = '/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv'


def get_data(file):
    """
    Reads the csv file using pandas and returns the data and labels as numpy arrays
    :param file: The file to read from
    :return: a tuple of numpy arrays and the labels
    """

    df = pd.read_csv(file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    data = df.drop('Label', axis=1)

    # Drop source port, source ip, and destination IP
    # TODO: Figure out how to handle target port number
    # TODO: Perform dataset cleaning

    labels = df['Label']

    data_np = data.to_numpy(dtype=np.float32, na_value=0)

    data_np = data_cleaning.clean_np_data(data_np)

    # Normalize data
    data_np = normalize(data_np)

    labels_list = labels.tolist()

    # Perform test/validation split
    data_train, data_test, labels_train, labels_test = train_test_split(data_np, labels_list, test_size=0.20)

    return data_train, data_test, labels_train, labels_test


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
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    data_train, data_test, labels_train, labels_test = get_data(DATA_FILE)
    clf.fit(data_train, labels_train)

    predictions = clf.predict(data_test)

    print('Done')


if __name__ == '__main__':
    main()
