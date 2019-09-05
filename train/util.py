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
    return (data - m) / s


def check_same(p, l):
    c = 0
    for i, el in enumerate(p):
        if el == l[i]:
            c += 1
    # print(count)
    return c


def calc_acc(preds, labels, threshold):
    count_same = 0
    count_all = 0
    for i, p in enumerate(preds):
        count_all += len(p)
        l = labels[i]
        p_thr = list(map(lambda n: 1 if n > threshold else 0, p))
        count_same += check_same(p_thr, l)

    return count_same/count_all


if __name__ == "__main__":
    # filepath = '../create_dataset/*.csv'
    # data = read_csv('../create_dataset/dataset/*.csv')
    # print(data.shape)
    p = [[0.11, 0.834, 0.33, 0.999],
         [0.11, 0.834, 0.33, 0.999],
         [0.11, 0.834, 0.33, 0.999],
         [0.11, 0.834, 0.33, 0.999]]
    l = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 0, 1],
         [0, 0, 0, 0]]
    ans = calc_acc(p, l, 0.5)
    print(ans)
