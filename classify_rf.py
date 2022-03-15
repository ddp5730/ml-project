# classify_rf.py
# 2/24/2018
# Dan Popp
#
# This file will classify the given IDS file using a random forest approach

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from load_data import load_2018_data

DATA_ROOT_2018 = '/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/'


def main():
    clf = RandomForestClassifier(max_depth=5, random_state=0, n_jobs=20, verbose=1)
    data_train, data_test, labels_train, labels_test = load_2018_data(DATA_ROOT_2018)

    print('\n\n-----------------------------------------------------------\n')
    print('Fitting RF Model')
    clf.fit(data_train, labels_train)

    predictions = clf.predict(data_test)

    print(classification_report(labels_test, predictions))

    print('Done')


if __name__ == '__main__':
    main()
