import torch
import torch.nn as nn
import torch.nn.functional as F

class TaskEncoder(nn.Module):
    """
    GeometryEncoder: A wrapper of the whole framework for the spatial reasoning task.
    """

    def __init__(self, args, geometry_encoder, device):
        super().__init__()
        self.args = args
        self.geometry_encoder = geometry_encoder
        self.task = args.task
        #self.fc = nn.Linear(1, args.num_classes)
        self.fc = nn.Sequential(
            nn.Linear(args.d_out * 2, args.d_out),
            nn.ReLU(),
            nn.Linear(args.d_out, args.num_classes)
        )

    def forward(self, x1, len1, dataset_type1, x2, len2, dataset_type2):
        x1_emb = self.geometry_encoder(x1, len1, dataset_type1)
        x2_emb = self.geometry_encoder(x2, len2, dataset_type2)
        if self.task == "relation-prediction":
            logits = torch.bmm(x1_emb.view(x1.shape[0], 1, -1), x2_emb.view(x2.shape[0], -1, 1)).squeeze()
            logits = torch.sigmoid(logits)
        elif self.task == "distance-prediction" or self.task == "knn":
            logits = torch.sqrt(((x1_emb - x2_emb) ** 2).sum(-1))
        elif self.task == "multi-relation" or self.task == "direction-prediction":
            #logits = torch.bmm(x1_emb.view(x1.shape[0], 1, -1), x2_emb.view(x2.shape[0], -1, 1))
            logits = torch.cat([x1_emb, x2_emb], dim=1)
            logits = self.fc(logits).squeeze()
            #logits = F.softmax(logits, dim=1)
            #logits = torch.argmax(logits, dim=1)
            #logits = logits.float()

        return logits
