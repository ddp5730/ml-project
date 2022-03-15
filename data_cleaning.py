# data_cleaning.py
# 3/01/21
# Dan Popp
#
# This file will contain functions to preprocess the dataset data.
import sys

import numpy as np
from tqdm import tqdm


def get_nearest_value(data, flow_idx, attribute_idx):
    """
    Gets the average data value from the preceding and following flow data
    :param data: the original data
    :param flow_idx: the flow index of the invalid data
    :param attribute_idx: the attribute index of the invalid data
    :return: the average value of the neighboring flows
    """

    use_preceding = True
    use_following = True

    if flow_idx == 0:
        use_preceding = False
    if flow_idx == data.shape[0] - 1:
        use_following = False

    if use_preceding:
        preceding = data[flow_idx - 1, attribute_idx]
        # Preceding is always guaranteed to be valid
    if use_following:
        following = data[flow_idx + 1, attribute_idx]
        if np.isnan(following) or np.isinf(following):
            use_following = False

    if use_preceding and use_following:
        avg_value = 0.5 * (preceding + following)
    elif use_preceding:
        avg_value = preceding
    else:
        avg_value = following

    return avg_value


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
                # TODO: Finalize cleaning approach
                data[flow_idx, attribute_idx] = class_avg[label][attribute_idx]
                num_invalid += 1
    print('Updated %d invalid values' % num_invalid)

    return data
