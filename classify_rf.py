# classify_rf.py
# 2/24/2018
# Dan Popp
#
# This file will classify the given IDS file using a random forest approach
import argparse
import os
import time

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from load_data import load_data, CIC_2017, CIC_2018


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-depth', default=10, help='Max Depth Hyperparam for RF')
    parser.add_argument('--data-path', required=True, help='Path to root directory of dataset')
    parser.add_argument('--dset', required=True, choices=[CIC_2017, CIC_2018], help='Dataset to classify')
    parser.add_argument('--pkl-path', type=str, default=None,  help='Path to store pickle files.  Saves time by '
                                                                    'storing preprocessed data')

    args = parser.parse_args()

    clf = RandomForestClassifier(max_depth=args.max_depth, random_state=0, n_jobs=20, verbose=1)
    data_train, data_test, labels_train, labels_test = load_data(args.dset, args.data_path, pkl_path=args.pkl_path)

    print('\n\n-----------------------------------------------------------\n')
    print('Fitting RF Model')
    start = time.time()
    clf.fit(data_train, labels_train)
    print('Training took %.2f minutes' % ((time.time() - start) / 60.0))

    predictions = clf.predict(data_test)

    print(classification_report(labels_test, predictions))
    cf_matrix = confusion_matrix(labels_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix)
    disp.plot()
    plt.savefig(os.path.join('./out/', 'rf_cf.png'))

    print('Done')


if __name__ == '__main__':
    main()
