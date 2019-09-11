from util import normalize, select_file
import numpy as np
import torch
import torch.utils.data
import linecache
import csv


class Dataset(torch.utils.data.Dataset):

    def __init__(self, filedata, filelen, labellen, transform=None):
        self.transform = transform
        self.filedata = filedata
        self.labellen = labellen
        self.filelen = filelen

    def __len__(self):
        return self.filelen

    def __getitem__(self, idx):

        filepath, line_num = select_file(idx, self.filedata)
        # print(idx, filepath, line_num)
        line = linecache.getline(filepath, line_num+1)
        csv_line = csv.reader([line])
        it = iter(csv_line)
        data = next(it)
        data = np.array(data)

        out_data = data[self.labellen:]
        out_label = data[:self.labellen]

        if self.transform:
            out_data = self.transform(out_data)
            # out_data = normalize(out_data)
            out_label = self.transform(out_label)
        # print(idx, line_num, out_label)
        return out_data, torch.FloatTensor(out_label)


class Transform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return data.astype(np.float)


# if __name__ == "__main__":
#     data = torch.tensor([1, 2, 3], dtype=torch.float)
#     print(normalize(data))
