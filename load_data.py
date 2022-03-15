import math
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN

import data_cleaning

PICKLE_PATH = '/home/poppfd/College/ML_Cyber/ml-project/data/'
DATA_ROOT_2018 = '/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/'


def load_2018_data(data_path):
    """
    Read in the entire 2018 dataset
    :param data_path: the path to the root data directory
    :return: a tuple for the full numpy arrays and labels
    """
    all_data = None
    all_labels = []
    all_dropped = 0

    for file in os.listdir(data_path):
        print('Loading file: %s ...' % file)
        data, labels, num_dropped = get_data(os.path.join(DATA_ROOT_2018, file))

        if all_data is None:
            all_data = data
        else:
            all_data = np.concatenate((all_data, data))
        all_labels += labels
        all_dropped += num_dropped

    print('Total Number of invalid values: %d' % all_dropped)
    print('Total Data values: %d' % len(all_labels))
    print('Invalid data: %.2f%%' % (all_dropped / float(all_data.size) * 100))

    # Perform test/validation split
    data_train, data_test, labels_train, labels_test = train_test_split(all_data, all_labels, test_size=0.20)

    # Resample Data
    class_samples = {}
    for label in labels_train:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    print('Initial Distribution of classes: ' + str(class_samples))

    # Goal is to have all classes represented as 20% of benign data
    smote_dict = {}
    target_num = round(class_samples['Benign'] * 0.15)
    print('Targeting %d samples for each minority class' % target_num)
    for label in class_samples.keys():
        if label == 'Benign' or class_samples[label] > target_num:
            smote_dict[label] = class_samples[label]
        else:
            smote_dict[label] = target_num

    smote = SMOTEENN(sampling_strategy=smote_dict, n_jobs=20)
    start = time.time()
    data_train, labels_train = smote.fit_resample(data_train, labels_train)
    print('SMOTE took %.2f minutes' % ((time.time() - start) / 60.0))
    print('Final Distribution of classes: ' + str(class_samples))
    print('Total Data values: %d' % len(all_labels))

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
            data_np, labels_list, num_dropped = pickle.load(file)
    else:

        df = pd.read_csv(file, dtype={'Timestamp': 'string'})
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # Remove invalid Dst Port values
        types = df.dtypes
        if types['Dst Port'].name != 'int64':
            df['C'] = np.where(df['Dst Port'].str.isdigit(), ['Retain'], ['Delete'])
            df = df[~df['C'].isin(['Delete'])]

            df = df.drop('C', axis=1)
        data = df.drop('Label', axis=1)

        if 'Flow ID' in df:
            data = data.drop('Flow ID', axis=1)
        if 'Src IP' in df:
            data = data.drop('Src IP', axis=1)
        if 'Src Port' in df:
            data = data.drop('Src Port', axis=1)
        if 'Dst IP' in df:
            data = data.drop('Dst IP', axis=1)
        # TODO: Figure out how to handle target port number

        labels = df['Label']
        labels_list = labels.tolist()

        data_np = data.to_numpy(dtype=np.float32, na_value=0)

        data_np, num_dropped = data_cleaning.clean_np_data(data_np, labels_list)

        is_nan = np.any(np.isnan(data_np))
        is_finite = np.all(np.isfinite(data_np))
        print('Data is nan: %s' % str(is_nan))
        print('Data is finite: %s' % str(is_finite))

        # Normalize data
        data_np = normalize(data_np)

        with open(pkl_path, 'wb') as file:
            pickle.dump((data_np, labels_list, num_dropped), file)

    return data_np, labels_list, num_dropped


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