import csv
import pathlib


def write_csv(data, filepath):
    save_data = pathlib.Path(filepath)
    if save_data.exists():
        save_data.touch()

    with open(filepath, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
        writer.writerows(list(data))