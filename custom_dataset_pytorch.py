import torch
from torch.utils.data import Dataset
import numpy as np

# read in pre porcessed data for pytorch Model

# dummy-Datensatz

data_train = [
    ["Das ist ein Test Satz", [1, 2, 0, 4, 5, 6, 7, 8, 9], 1],
    ["Dieses Model wird super", [0, 0, 1, 0, 0, 0, 0, 0, 0], 0],
    ["Ich brauche noch einen dritten Satz", [40, 10, 0, 41, 34, 89, 356, 8, 154], 1],
]

data_test = [
    ["Wow, noch mehr Sätze", [4, 5, 6], 1],
    ["Test-Satz nummer 2", [0, 15, 3], 0],
    ["Und Drittens", [1, 6, 1], 1],
]


class CustomEmbeddingDataset(Dataset):
    def __init__(self, embeddings_list):
        self.emb_list = embeddings_list

    def __len__(self):
        return len(self.emb_list)

    def __getitem__(self, idx):
        # assume form of input data: [(string, embedding_vector, label), ...]
        # !! -> current form of input data (based on corpus2embeddings -> [(embedding_vector, label)])
        emb = np.array(self.emb_list[idx][1])
        label = self.emb_list[idx][2]
        return emb, label


# little test main

if __name__ == "__main__":
    train_dataset = CustomEmbeddingDataset(data_train)
    print(train_dataset.__getitem__(1))
    print(train_dataset.__getitem__(2))
    print(train_dataset.__getitem__(0))
