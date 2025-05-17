import torch
import torch.nn as nn
from models.poly2vec import Poly2Vec

class GeometryEncoder(nn.Module):
    """
    GeometryEncoder: A wrapper of the whole framework for the spatial reasoning task.
    """

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

        self.encoder_type = args.encoder_type
        if self.encoder_type == "poly2vec":
            self.encoder = Poly2Vec(args=args, device=self.device)
        else:
            raise NotImplementedError(f"Encoder {self.encoder_type} not supported")

    def forward(self, x, lengths, dataset_type):
        x_emb = self.encoder.encode(x, lengths, dataset_type=dataset_type)

        return x_emb