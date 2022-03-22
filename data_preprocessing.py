# data_cleaning.py
# 3/01/21
# Dan Popp
#
# This file will contain functions to preprocess the dataset data.
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
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


def resample_data(data, labels):
    """
    Resample the data.  Undersamples the benign data so it's only 10x greater than largest minority using
    RandomUnderSampler.  Then any minority class less than 1% of the majority class is dropped.
    Finally SMOTE is used to oversample minority classes up to 20% of the majority class.
    :param data: The data
    :param labels: the labels
    :return: returns the updated data and labels along with a list of classes to drop from the testing data.
    """

    class_samples = {}
    orig_samples = len(labels)
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'orig_dist_2018')

    # Undersample Benign Data to 10x greater than next largest class
    benign_num = class_samples['Benign']
    largest_min_num = 0
    for class_name in class_samples.keys():
        if class_name != 'Benign' and class_samples[class_name] > largest_min_num:
            largest_min_num = class_samples[class_name]
    target_benign = 10 * largest_min_num
    print('Reducing Benign data from %d to %d samples' % (benign_num, target_benign))
    undersampler = RandomUnderSampler(sampling_strategy={'Benign': target_benign})
    data, labels = undersampler.fit_resample(data, labels)

    print('Finished Undersampling')
    class_samples = {}
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'after_undersampling_2018')

    # Drop extreme minority classes
    min_class_count = 0.01 * class_samples['Benign']
    print('Dropping classes with < %d samples' % min_class_count)

    classes_to_drop = []
    for class_name in class_samples.keys():
        if class_samples[class_name] < min_class_count:
            classes_to_drop.append(class_name)

    data, labels = drop_classes(data, labels, classes_to_drop)

    class_samples = {}
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'after_dropping_2018')

    # Goal is to have all classes represented as 20% of benign data
    smote_dict = {}
    target_num = round(class_samples['Benign'] * 0.20)
    print('Targeting %d samples for each minority class' % target_num)
    for label in class_samples.keys():
        if label == 'Benign' or class_samples[label] > target_num:
            smote_dict[label] = class_samples[label]
        else:
            smote_dict[label] = target_num

    smote = SMOTE(sampling_strategy=smote_dict, n_jobs=20)
    start = time.time()
    data, labels = smote.fit_resample(data, labels)
    print('SMOTE took %.2f minutes' % ((time.time() - start) / 60.0))
    class_samples = {}
    for label in labels:
        if label not in class_samples:
            class_samples[label] = 1
        else:
            class_samples[label] += 1
    save_class_hist(class_samples, 'after_smote_2018')
    print('Total Data values: %d' % len(orig_samples))

    return data, labels, classes_to_drop


def drop_classes(data, labels, classes_to_drop):
    drop_row = np.full(len(labels), False)
    for i in range(len(labels)):
        if labels[i] in classes_to_drop:
            drop_row[i] = True

    new_labels = []
    for i in range(len(labels)):
        if labels[i] not in classes_to_drop:
            new_labels.append(labels[i])
    labels = new_labels

    data = data[~drop_row, :]

    print('Done dropping.  Shape of data %s -- Size of labels %d' % (str(data.shape), len(labels)))
    return data, labels


def save_class_hist(samples_dict: dict, name: str):
    classes = samples_dict.keys()
    samples = []
    for class_name in classes:
        samples.append(samples_dict[class_name])

    plt.clf()
    plt.bar(classes, samples)
    plt.title('Class Distribution')
    plt.ylabel('Num Samples')
    plt.xlabel('Class Name')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.savefig(os.path.join('./out/', '%s.png' % name))
    plt.clf()
