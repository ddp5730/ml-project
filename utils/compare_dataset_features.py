# compare_dataset_features.py
# 4/08/2018
# Dan Popp
#
# This file will handle information about comparing attributes between the two datasets.

import os

import pandas as pd

DATA_ROOT_2018 = '/home/poppfd/data/CIC-IDS2018/Processed_Traffic_Data_for_ML_Algorithms/'
DATA_ROOT_2017 = '/home/poppfd/data/CIC-IDS2017/MachineLearningCVE'

attributes_2017 = ['Destination Port',
                   'Flow Duration',
                   'Total Fwd Packets',
                   'Total Backward Packets',
                   'Total Length of Fwd Packets',
                   'Total Length of Bwd Packets',
                   'Fwd Packet Length Max',
                   'Fwd Packet Length Min',
                   'Fwd Packet Length Mean',
                   'Fwd Packet Length Std',
                   'Bwd Packet Length Max',
                   'Bwd Packet Length Min',
                   'Bwd Packet Length Mean',
                   'Bwd Packet Length Std',
                   'Flow Bytes/s',
                   'Flow Packets/s',
                   'Flow IAT Mean',
                   'Flow IAT Std',
                   'Flow IAT Max',
                   'Flow IAT Min',
                   'Fwd IAT Total',
                   'Fwd IAT Mean',
                   'Fwd IAT Std',
                   'Fwd IAT Max',
                   'Fwd IAT Min',
                   'Bwd IAT Total',
                   'Bwd IAT Mean',
                   'Bwd IAT Std',
                   'Bwd IAT Max',
                   'Bwd IAT Min',
                   'Fwd PSH Flags',
                   'Bwd PSH Flags',
                   'Fwd URG Flags',
                   'Bwd URG Flags',
                   'Fwd Header Length',
                   'Bwd Header Length',
                   'Fwd Packets/s',
                   'Bwd Packets/s',
                   'Min Packet Length',
                   'Max Packet Length',
                   'Packet Length Mean',
                   'Packet Length Std',
                   'Packet Length Variance',
                   'FIN Flag Count',
                   'SYN Flag Count',
                   'RST Flag Count',
                   'PSH Flag Count',
                   'ACK Flag Count',
                   'URG Flag Count',
                   'CWE Flag Count',
                   'ECE Flag Count',
                   'Down/Up Ratio',
                   'Average Packet Size',
                   'Avg Fwd Segment Size',
                   'Avg Bwd Segment Size',
                   'Fwd Avg Bytes/Bulk',
                   'Fwd Avg Packets/Bulk',
                   'Fwd Avg Bulk Rate',
                   'Bwd Avg Bytes/Bulk',
                   'Bwd Avg Packets/Bulk',
                   'Bwd Avg Bulk Rate',
                   'Subflow Fwd Packets',
                   'Subflow Fwd Bytes',
                   'Subflow Bwd Packets',
                   'Subflow Bwd Bytes',
                   'Init_Win_bytes_forward',
                   'Init_Win_bytes_backward',
                   'act_data_pkt_fwd',
                   'min_seg_size_forward',
                   'Active Mean',
                   'Active Std',
                   'Active Max',
                   'Active Min',
                   'Idle Mean',
                   'Idle Std',
                   'Idle Max',
                   'Idle Min',
                   'Label']

attributes_2018 = ['Dst Port',
                   'Flow Duration',
                   'Tot Fwd Pkts',
                   'Tot Bwd Pkts',
                   'TotLen Fwd Pkts',
                   'TotLen Bwd Pkts',
                   'Fwd Pkt Len Max',
                   'Fwd Pkt Len Min',
                   'Fwd Pkt Len Mean',
                   'Fwd Pkt Len Std',
                   'Bwd Pkt Len Max',
                   'Bwd Pkt Len Min',
                   'Bwd Pkt Len Mean',
                   'Bwd Pkt Len Std',
                   'Flow Byts/s',
                   'Flow Pkts/s',
                   'Flow IAT Mean',
                   'Flow IAT Std',
                   'Flow IAT Max',
                   'Flow IAT Min',
                   'Fwd IAT Tot',
                   'Fwd IAT Mean',
                   'Fwd IAT Std',
                   'Fwd IAT Max',
                   'Fwd IAT Min',
                   'Bwd IAT Tot',
                   'Bwd IAT Mean',
                   'Bwd IAT Std',
                   'Bwd IAT Max',
                   'Bwd IAT Min',
                   'Fwd PSH Flags',
                   'Bwd PSH Flags',
                   'Fwd URG Flags',
                   'Bwd URG Flags',
                   'Fwd Header Len',
                   'Bwd Header Len',
                   'Fwd Pkts/s',
                   'Bwd Pkts/s',
                   'Pkt Len Min',
                   'Pkt Len Max',
                   'Pkt Len Mean',
                   'Pkt Len Std',
                   'Pkt Len Var',
                   'FIN Flag Cnt',
                   'SYN Flag Cnt',
                   'RST Flag Cnt',
                   'PSH Flag Cnt',
                   'ACK Flag Cnt',
                   'URG Flag Cnt',
                   'CWE Flag Count',
                   'ECE Flag Cnt',
                   'Down/Up Ratio',
                   'Pkt Size Avg',
                   'Fwd Seg Size Avg',
                   'Bwd Seg Size Avg',
                   'Fwd Byts/b Avg',
                   'Fwd Pkts/b Avg',
                   'Fwd Blk Rate Avg',
                   'Bwd Byts/b Avg',
                   'Bwd Pkts/b Avg',
                   'Bwd Blk Rate Avg',
                   'Subflow Fwd Pkts',
                   'Subflow Fwd Byts',
                   'Subflow Bwd Pkts',
                   'Subflow Bwd Byts',
                   'Init Fwd Win Byts',
                   'Init Bwd Win Byts',
                   'Fwd Act Data Pkts',
                   'Fwd Seg Size Min',
                   'Active Mean',
                   'Active Std',
                   'Active Max',
                   'Active Min',
                   'Idle Mean',
                   'Idle Std',
                   'Idle Max',
                   'Idle Min',
                   'Label']

unique_2017_attributes = ['Fwd Header Length.1']
unique_2018_attributes = ['Protocol',
                          'Timestamp']


def main():
    """
    Print out some helpful info about the attributes of both datasets
    :return:
    """
    file_2018 = os.path.join(DATA_ROOT_2018, os.listdir(DATA_ROOT_2018)[0])
    file_2017 = os.path.join(DATA_ROOT_2017, os.listdir(DATA_ROOT_2017)[0])

    data_2018 = pd.read_csv(file_2018)
    data_2017 = pd.read_csv(file_2017)

    data_2017.columns = data_2017.columns.str.strip()

    print('2018 Columns: %d' % len(data_2018.columns))
    print(data_2018.columns)
    print('\n----------------------------------------------------------------\n')
    print('2017 Columns: %d' % len(data_2017.columns))
    print(data_2017.columns)

    attribute_map_2017_to_2018 = get_attribute_map(debug=True)

    print('\n----------------------------------------------------------------\n')
    print('Unique 2018 labels: %s' % (str(data_2018['Label'].unique())))
    print('Unique 2017 labels: %s' % (str(data_2017['Label'].unique())))

    print('Done')


def get_attribute_map(debug=False):
    """
    Return the mapping of attribute string names between the 2017 and 2018 dataset.
    :param debug: If set this function will print out debug info
    :return: A map containing the 2017 attribute string as the key and the 2018 attribute as the value
    """
    attribute_map_2017_to_2018 = {}
    for i in range(len(attributes_2017)):
        if attributes_2017[i] in unique_2017_attributes:
            # No valid mapping for this feature
            attribute_map_2017_to_2018[attributes_2017[i]] = ''
        else:
            # 2018 features with no mapping: timestamp, protocol
            # Timestamp is probably valid to just ignore actually since it won't work well across datasets.
            attribute_map_2017_to_2018[attributes_2017[i]] = attributes_2018[i]
        if debug:
            print('%s: %s' % (attributes_2017[i], attribute_map_2017_to_2018[attributes_2017[i]]))

    if debug:
        print(attribute_map_2017_to_2018)
    return attribute_map_2017_to_2018


if __name__ == '__main__':
    main()
