# data_cleaning.py
# 3/01/21
# Dan Popp
#
# This file will contain functions to preprocess the dataset data.
import sys

import numpy as np
from tqdm import tqdm


def clean_np_data(data, labels):
    """
    Cleans the numpy data array.  The effect is to remove NaN and Inf values by using a nearest neighbor approach.
    Data deemed to be invalid will also be adjusted to the nearest valid value
    :param data: The data array
    :return: the processed data array
    """

    num_invalid = 0

    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    class_avg = {}
    for label in unique_labels:
        index = np.full(len(labels), False)
        for i in range(len(labels)):
            if labels[i] == label:
                index[i] = True
        class_data = data[index, :]
        class_avg[label] = np.average(np.ma.masked_invalid(class_data), axis=0)

    for flow_idx in tqdm(range(data.shape[0]), file=sys.stdout, desc='Cleaning data array...'):
        label = labels[flow_idx]
        for attribute_idx in range(data.shape[1]):
            data_val = data[flow_idx, attribute_idx]
            if np.isnan(data_val) or np.isinf(data_val):
                # Data is cleaned by replacing invalid values with the average value
                # within the given data class label
                data[flow_idx, attribute_idx] = class_avg[label][attribute_idx]
                num_invalid += 1
    print('Updated %d invalid values' % num_invalid)

    return data, num_invalid
