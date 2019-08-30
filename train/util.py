import csv


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = [e for e in reader]
        f.close()
    return data
