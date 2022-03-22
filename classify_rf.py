# classify_rf.py
# 2/24/2018
# Dan Popp
#
# This file will classify the given IDS file using a random forest approach
import os
import time

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from load_data import load_2018_data

DATA_ROOT_2018 = '/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/'


def main():
    clf = RandomForestClassifier(max_depth=10, random_state=0, n_jobs=20, verbose=1)
    data_train, data_test, labels_train, labels_test = load_2018_data(DATA_ROOT_2018)

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
