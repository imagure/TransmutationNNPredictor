import pandas as pd
import torch
from torch.utils.data import Dataset

core_map = {
    "C": 1,
    "Y": 2,
    "X": 3,
    "A": 4,
    "S": 5
}

class FeatureDataset(Dataset):

    def __init__(self, file_name):
        file_out = pd.read_csv(file_name)

        x = file_out.iloc[0:1024, 0:18].values
        y = file_out.iloc[0:1024, 18:22].values

        x_train = []
        y_train = []

        for row in range(len(x)):
            x_train.append([])
            for column in range(len(x[row])):
                x_train[row].append("")
                x_train[row][column]=core_map[x[row][column]]

        for row in range(len(y)):
            y_train.append([])
            for column in range(len(y[row])):
                y_train[row].append("")
                y_train[row][column]=core_map[y[row][column]]

        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.Y_train[idx]
