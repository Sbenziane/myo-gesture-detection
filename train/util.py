import csv
import numpy as np
import glob
import pprint


INPUT_SIZE = 2053


def read_csv(filepath):
    filelist = glob.glob(filepath)
    pprint.pprint(filelist)
    alldata = np.empty((0, INPUT_SIZE))
    for filename in filelist:
        print(filename)
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = [e for e in reader]
            data = np.array(data)
            # print(data.shape)
            # print(alldata.shape)
            alldata = np.append(alldata, data, axis=0)
            f.close()
    return alldata


def normalize(data):
    m = np.mean(data)
    s = np.std(data)
    return (data-m)/s


# if __name__ == "__main__":
#     filepath = '../create_dataset/*.csv'
#     data = read_csv('../create_dataset/dataset/*.csv')
#     print(data.shape)
