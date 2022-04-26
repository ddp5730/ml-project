import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

import data_preprocessing
from data_preprocessing import resample_data
from utils.compare_dataset_features import unique_2018_attributes, get_attribute_map, attributes_2018, attributes_2017

# TODO: Make pickle_path a command line argument
# TODO: Make data paths command line arguments
# TODO: Make datasetname the argument compared against and command line argument
PICKLE_PATH = '/home/poppfd/College/ML_Cyber/ml-project/data/'

CIC_2017 = 'cic-2017'
CIC_2018 = 'cic-2018'

BENIGN_LABEL_2018 = 'Benign'
BENIGN_LABEL_2017 = 'BENIGN'


def load_data(dset, data_path):
    """
    Read in the entire 2018 dataset
    :param is_2018: True if loading 2018 data.  False for 2017 data
    :param data_path: the path to the root data directory
    :return: a tuple for the full numpy arrays and labels
    """
    all_data = None
    all_labels = []
    all_dropped = 0

    is_2018 = dset == CIC_2018

    pkl_path = os.path.join(PICKLE_PATH, 'all_data_%s.pkl' % ('2018' if is_2018 else '2017'))
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            data_train, data_test, labels_train, labels_test = pickle.load(file)
    else:
        for file in os.listdir(data_path):
            print('Loading file: %s ...' % file)
            data, labels, num_dropped = get_data(os.path.join(data_path, file), is_2018=is_2018)

            if all_data is None:
                all_data = data
            else:
                all_data = np.concatenate((all_data, data))
            all_labels += labels
            all_dropped += num_dropped

        print('Total Number of invalid values: %d' % all_dropped)
        print('Total Data values: %d' % len(all_labels))
        print('Invalid data: %.2f%%' % (all_dropped / float(all_data.size) * 100))

        label_mapping = []
        for label in all_labels:
            if label not in label_mapping:
                label_mapping.append(label)
        print('Dataset labels: %s' % str(label_mapping))

        # Perform test/validation split
        data_train, data_test, labels_train, labels_test = train_test_split(all_data, all_labels, test_size=0.20)

        # Resample Data
        data_train, labels_train, classes_to_drop = resample_data(data_train, labels_train, is_2018=is_2018)
        data_test, labels_test = data_preprocessing.drop_classes(data_test, labels_test, classes_to_drop)

        with open(pkl_path, 'wb') as file:
            pickle.dump((data_train, data_test, labels_train, labels_test), file)

    return data_train, data_test, labels_train, labels_test


def get_datasets(dset, data_path):
    data_train, data_test, labels_train, labels_test = load_data(dset, data_path)
    data_train = torch.tensor(data_train)
    data_test = torch.tensor(data_test)

    # Convert string list to list of integers
    label_mapping = {}
    value = 0
    for label in labels_test:
        if label not in label_mapping:
            label_mapping[label] = value
            value += 1

    labels_idx_train = []
    for i in range(len(labels_train)):
        label = labels_train[i]
        value = label_mapping[label]
        labels_idx_train.append(value)

    labels_idx_test = []
    for i in range(len(labels_test)):
        label = labels_test[i]
        value = label_mapping[label]
        labels_idx_test.append(value)

    labels_train = torch.tensor(labels_idx_train)
    labels_test = torch.tensor(labels_idx_test)
    classes = list(label_mapping.keys())

    dataset_train = TensorDataset(data_train, labels_train)
    dataset_test = TensorDataset(data_test, labels_test)

    dataset_train.classes = classes
    dataset_test.classes = classes

    return dataset_train, dataset_test


def get_data(file, is_2018=True):
    """
    Reads the csv file using pandas and returns the data and labels as numpy arrays
    :param is_2018:
    :param file: The file to read from
    :return: a tuple of numpy arrays and the labels
    """

    filename = os.path.splitext(os.path.basename(file))[0] + '.pkl'
    pkl_path = os.path.join(PICKLE_PATH, filename)

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as file:
            data_np, labels_list, num_dropped = pickle.load(file)
    else:

        if is_2018:
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

            # Drop data that isn't in 2017 dataset
            for attribute in unique_2018_attributes:
                if attribute in df:
                    data = data.drop(attribute, axis=1)
        else:
            df = pd.read_csv(file)
            df.columns = df.columns.str.strip()

            # Drop attributes unique to 2017 data
            for attribute in unique_2018_attributes:
                if attribute in df:
                    df = df.drop(attribute, axis=1)

            # Remap and reorder the columns to match 2018 data
            attribute_map = get_attribute_map()
            for i in range(len(attributes_2017)):
                df.columns = df.columns.str.replace(attributes_2017[i], attribute_map[(attributes_2017[i])])
            df = df[attributes_2018]

            data = df.drop('Label', axis=1)

        labels = df['Label']
        labels_list = labels.tolist()

        if not is_2018:
            # Convert labels to be consistent
            for i in range(len(labels)):
                if labels_list[i] == BENIGN_LABEL_2017:
                    labels_list[i] = BENIGN_LABEL_2018

        data_np = data.to_numpy(dtype=np.float32, na_value=0)

        data_np, num_dropped = data_preprocessing.clean_np_data(data_np, labels_list)

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
