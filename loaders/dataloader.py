from torch.utils.data import Dataset
import torch
import numpy as np


class GeometryRelationshipDataset(Dataset):
    """A custom dataset to handle pairs of geometric data."""

    def __init__(self, X1, X2, L1, L2, Y):
        super().__init__()
        self.X1 = X1
        self.X2 = X2
        self.L1 = L1
        self.L2 = L2
        self.Y = Y

    def __len__(self) -> int:
        """Returns: The number of samples in the dataset."""
        return len(self.X1)

    def __getitem__(self, idx: int):
        """
        Retrieves the pair of geometries and its label at the specified index.

        Args:
        idx (int): The index of the item to retrieve.

        Returns:
        tuple: Two tensors for the geometries and one tensor for the label.
        """
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        l1 = self.L1[idx]
        l2 = self.L2[idx]
        y = self.Y[idx]
        return x1, x2, l1, l2, y


class RegionLoader(torch.utils.data.Dataset):
    def __init__(self, region_features, attn_masks, labels):
        # load region features
        self.region_features = region_features
        self.attn_masks = attn_masks
        self.labels = labels

    def __len__(self):
        return len(self.region_features)

    def __getitem__(self, idx):
        return {
            'features': self.region_features[idx],
            'attn_mask': self.attn_masks[idx],
            'y': self.labels[idx]
        }


class KNNDataset(Dataset):
    """A custom dataset to handle pairs of geometric data."""

    def __init__(self, X1: torch.Tensor, X2: torch.Tensor,
                 dis_matrix: np.array,
                 L2: torch.Tensor = None):
        super().__init__()
        self.X1 = X1
        self.X2 = X2
        self.dis_matrix = dis_matrix
        self.L2 = L2

    def __len__(self) -> int:
        """Returns: The number of samples in the dataset."""
        return len(self.X1)

    def __getitem__(self, idx: int):
        """
        Retrieves the pair of geometries and its label at the specified index.

        Args:
        idx (int): The index of the item to retrieve.

        Returns:
        tuple: Two tensors for the geometries and one tensor for the label.
        """
        geom_1 = self.X1[idx]
        geom_2 = self.X2[idx]
        if self.L2 is not None:
            len2 = self.L2[idx]
        else:
            len2 = torch.tensor([])
        return geom_1, geom_2, torch.tensor([]), len2, idx