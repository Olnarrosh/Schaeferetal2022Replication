"""
@Author Sandro Weick
"""
import torch
from torch.utils.data import Dataset
import numpy as np

# read in pre porcessed data for pytorch Model


class CustomEmbeddingDataset(Dataset):
    """
    Custom implementation of the pytorch Dataset Class
    The important features for pytorch are the sentence embeddings
    and the sentence label
    """
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

