import csv
import pathlib
import numpy as np


def write_csv(data, filepath):
    save_data = pathlib.Path(filepath)
    if save_data.exists():
        save_data.touch()

    with open(filepath, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
        writer.writerows(data)


def data_append(origin_data, new_data):
    # for i in new_data:
    #     origin_data.append(i)
    origin_data.append(new_data)
    return origin_data


def normalize(data):
    m = np.mean(data)
    s = np.std(data)
    return (data - m) / s

def sigmoid(x):
  return  1/(1+np.exp(-x))
