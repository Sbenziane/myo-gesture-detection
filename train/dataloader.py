from util import normalize
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, labellen, transform=None):
        self.transform = transform
        self.data = data
        self.labellen = labellen

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        input_data = self.data[:, self.labellen:]
        label_data = self.data[:, :self.labellen]

        out_data = input_data[idx]
        out_label = label_data[idx]

        if self.transform:
            out_data = self.transform(out_data)
            out_data = normalize(out_data)
            out_label = self.transform(out_label)
        return out_data, out_label


class Transform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        return data.astype(np.float)


# if __name__ == "__main__":
#     data = torch.tensor([1, 2, 3], dtype=torch.float)
#     print(normalize(data))
