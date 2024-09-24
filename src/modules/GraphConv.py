import numpy as np
import torch
import torch.nn as nn
from modules.Aggregator import Aggregator
import torch.nn.functional as F


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, embed_dim, n_hops, device, dropout_rate):
        super(GraphConv, self).__init__()

        self.embed_dim = embed_dim
        self.aggregator_layers = nn.ModuleList()
        self.device = device
        self.dropout_rate = dropout_rate

        for i in range(n_hops):
            self.aggregator_layers.append(Aggregator().to(self.device))

        self.dropout = nn.Dropout(p=dropout_rate)

    def edge_sampling(self, edge_index, edge_type, rate):
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * (1 - rate)), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def forward(self, ego_embed, edge_index, edge_type, relation_embed, dropout):
        if dropout:
            edge_index, edge_type = self.edge_sampling(edge_index, edge_type, self.dropout_rate)

        ego_res_embed = ego_embed

        for i in range(len(self.aggregator_layers)):
            ego_embed = self.aggregator_layers[i](ego_embed, edge_index, edge_type, relation_embed)
            if dropout:
                ego_embed = self.dropout(ego_embed)
            ego_embed = F.normalize(ego_embed)
            ego_res_embed = torch.add(ego_res_embed, ego_embed)

        return ego_res_embed
