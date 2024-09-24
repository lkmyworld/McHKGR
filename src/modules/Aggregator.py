import torch.nn as nn
from modules.Hyperbolic import *
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum


class Aggregator(nn.Module):
    """
    Relational-aware Convolution Network
    """

    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, ego_embed, edge_index, edge_type, relation_embed):
        head, tail = edge_index
        relation_type = edge_type
        n_entities = ego_embed.shape[0]

        head_emb = ego_embed[head]
        tail_emb = ego_embed[tail]
        relation_emb = relation_embed[relation_type]

        # hyperbolic
        hyper_head_emb = expmap0(head_emb)
        hyper_tail_emb = expmap(tail_emb, hyper_head_emb)
        hyper_relation_emb = expmap(relation_emb, hyper_head_emb)
        res = project(mobius_add(hyper_tail_emb, hyper_relation_emb))
        res = logmap(res, hyper_head_emb)
        entity_agg = scatter_mean(src=res, index=head, dim_size=n_entities, dim=0)

        return entity_agg

    def calculate_sim_hrt(self, entity_emb_head, entity_emb_tail, relation_emb):
        tail_relation_emb = entity_emb_tail * relation_emb
        tail_relation_emb = tail_relation_emb.norm(dim=1, p=2, keepdim=True)
        head_relation_emb = entity_emb_head * relation_emb
        head_relation_emb = head_relation_emb.norm(dim=1, p=2, keepdim=True)
        att_weights = torch.matmul(head_relation_emb.unsqueeze(dim=1), tail_relation_emb.unsqueeze(dim=2)).squeeze(
            dim=-1)
        att_weights = att_weights ** 2
        return att_weights
