import torch
import torch.nn as nn
from modules.GraphConv import GraphConv
import torch.nn.functional as F
from utils.util import L2_loss_mean


class McHKGR(nn.Module):
    def __init__(self, args, data, device):
        super(McHKGR, self).__init__()
        self.device = device
        self.n_users = data.n_users
        self.n_items = data.n_items
        self.n_entities = data.n_entities
        self.n_nodes = data.n_nodes

        self.ckg_n_relations = data.ckg_n_relations
        self.ukg_n_relations = data.ukg_n_relations

        self.embedding_dim = args.embedding_dim
        self.context_hops = args.context_hops

        self.dropout = args.node_dropout
        self.dropout_rate = args.node_dropout_rate

        self.l2_lambda = args.l2_lambda
        self.contrastive_lambda = args.contrastive_lambda
        self.tau = args.tau

        self.ckg_edge_index, self.ckg_edge_type = self.get_edges(data.ckg_graph)
        self.ukg_edge_index, self.ukg_edge_type = self.get_edges(data.ukg_graph)

        self.image_embedding = nn.Embedding.from_pretrained(data.image_features, freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(data.text_features, freeze=True)
        self.other_embedding_image = nn.Embedding(self.n_nodes - self.n_items, self.embedding_dim)
        self.other_embedding_text = nn.Embedding(self.n_nodes - self.n_items, self.embedding_dim)
        self.ckg_relation_embedding_image = nn.Embedding(self.ckg_n_relations, self.embedding_dim)
        self.ckg_relation_embedding_text = nn.Embedding(self.ckg_n_relations, self.embedding_dim)
        self.ukg_relation_embedding_image = nn.Embedding(self.ukg_n_relations, self.embedding_dim)
        self.ukg_relation_embedding_text = nn.Embedding(self.ukg_n_relations, self.embedding_dim)

        nn.init.xavier_uniform_(self.other_embedding_image.weight)
        nn.init.xavier_uniform_(self.other_embedding_text.weight)
        nn.init.xavier_uniform_(self.ckg_relation_embedding_image.weight)
        nn.init.xavier_uniform_(self.ckg_relation_embedding_text.weight)
        nn.init.xavier_uniform_(self.ukg_relation_embedding_image.weight)
        nn.init.xavier_uniform_(self.ukg_relation_embedding_text.weight)

        self.item_embedding_image = nn.Embedding(self.n_items, self.embedding_dim)
        self.item_embedding_text = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_embedding_image.weight)
        nn.init.xavier_uniform_(self.item_embedding_text.weight)

        self.image_linear = nn.Linear(data.image_features.shape[1], self.embedding_dim * 4)
        self.image_linear_2 = nn.Linear(4 * self.embedding_dim, int(self.embedding_dim))
        self.text_linear = nn.Linear(data.text_features.shape[1], self.embedding_dim * 4)
        self.text_linear_2 = nn.Linear(4 * self.embedding_dim, int(self.embedding_dim))

        self.gcn = GraphConv(embed_dim=self.embedding_dim,
                             n_hops=self.context_hops,
                             device=self.device,
                             dropout_rate=self.dropout_rate)

        self.criterion = torch.nn.BCELoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, *input, mode):
        if mode == 'topk':
            return self.calc_topk_score(*input)
        if mode == 'ctr':
            return self.calc_ctr_score(*input)
        if mode == 'train':
            return self.calc_train(*input)

    def get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))
        index = graph_tensor[:, :-1]
        type = graph_tensor[:, -1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def calc_cf_embeddings(self, user_ids, item_ids):
        image_features = self.image_linear_2(F.leaky_relu(self.image_linear(self.image_embedding.weight)))
        text_features = self.text_linear_2(F.leaky_relu(self.text_linear(self.text_embedding.weight)))

        ego_embed_image = torch.cat((image_features, self.other_embedding_image.weight), dim=0).to(self.device)
        ego_embed_text = torch.cat((text_features, self.other_embedding_text.weight), dim=0).to(self.device)

        all_embed_image = self.gcn(ego_embed_image, self.ckg_edge_index, self.ckg_edge_type,
                                   self.ckg_relation_embedding_image.weight,
                                   dropout=self.dropout)
        all_embed_text = self.gcn(ego_embed_text, self.ckg_edge_index, self.ckg_edge_type,
                                  self.ckg_relation_embedding_text.weight,
                                  dropout=self.dropout)

        user_ego_embed_image = all_embed_image[self.n_entities:]
        final_user_embed_image = self.gcn(user_ego_embed_image, self.ukg_edge_index, self.ukg_edge_type,
                                          self.ukg_relation_embedding_image.weight,
                                          dropout=self.dropout)

        user_ego_embed_text = all_embed_text[self.n_entities:]
        final_user_embed_text = self.gcn(user_ego_embed_text, self.ukg_edge_index, self.ukg_edge_type,
                                         self.ukg_relation_embedding_text.weight,
                                         dropout=self.dropout)

        user_f_image = final_user_embed_image[(user_ids - self.n_entities)]
        user_f_text = final_user_embed_text[(user_ids - self.n_entities)]

        user_embed = torch.cat((user_f_image, user_f_text), dim=1).to(self.device)
        item_embed = torch.cat((all_embed_image[item_ids], all_embed_text[item_ids]), dim=1).to(self.device)

        loss_contrast = self.calculate_loss_1(user_f_image, user_f_text) + \
                        self.calculate_loss_2(all_embed_image[item_ids], all_embed_text[item_ids])

        return user_embed, item_embed, loss_contrast

    def calc_train(self, user_ids, item_ids, labels):
        user_embed, item_embed, loss_contrast = self.calc_cf_embeddings(user_ids, item_ids)

        logits = torch.sigmoid((user_embed * item_embed).sum(dim=-1)).squeeze()
        cf_loss = self.criterion(logits, labels)

        l2_loss = L2_loss_mean(user_embed) + L2_loss_mean(item_embed)
        loss = cf_loss + self.l2_lambda * l2_loss + self.contrastive_lambda * loss_contrast
        return loss

    def calculate_loss_1(self, A_embedding, B_embedding):
        tau = self.tau
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def calculate_loss_2(self, A_embedding, B_embedding):
        tau = self.tau
        f = lambda x: torch.exp(x / tau)
        refl_sim = f(self.sim(A_embedding, A_embedding))
        between_sim = f(self.sim(A_embedding, B_embedding))

        loss_1 = -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        refl_sim_1 = f(self.sim(B_embedding, B_embedding))
        between_sim_1 = f(self.sim(B_embedding, A_embedding))
        loss_2 = -torch.log(
            between_sim_1.diag()
            / (refl_sim_1.sum(1) + between_sim_1.sum(1) - refl_sim_1.diag()))
        ret = (loss_1 + loss_2) * 0.5
        ret = ret.mean()
        return ret

    def calc_ctr_score(self, user_ids, item_ids):
        user_embed, item_embed, _ = self.calc_cf_embeddings(user_ids, item_ids)

        cf_score = torch.sigmoid((user_embed * item_embed).sum(dim=1)).squeeze()

        return cf_score

    def calc_topk_score(self, user_ids, item_ids):
        user_embed, item_embed, _ = self.calc_cf_embeddings(user_ids, item_ids)

        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))
        cf_score = torch.sigmoid(cf_score)

        return cf_score
