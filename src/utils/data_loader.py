import collections

import numpy as np
import pandas as pd
import os
from collections import defaultdict
import networkx as nx
import scipy.sparse as sp
import pickle


class Dataloader(object):
    def __init__(self, args, logging):
        self.args = args
        self.dataset = args.dataset
        self.dataset_dir = str(os.path.join(args.data_dir, args.dataset))
        self.train_valid_test = os.path.join(self.dataset_dir, "train_valid_test")
        self.rating_file = os.path.join(self.dataset_dir, "ratings_final.txt")

        self.ckg_file = os.path.join(self.dataset_dir, "kg_final.txt")
        self.ukg_file = os.path.join(self.dataset_dir, "ukg_final.txt")
        self.pkl_file = os.path.join(self.dataset_dir, 'image_text_pair.pkl')

        if not os.path.exists(self.train_valid_test + '/train_data_with_neg.pkl'):
            rating_np = np.loadtxt(self.rating_file, dtype=np.int64)
            self.train_data_with_neg, self.valid_data_with_neg, self.test_data_with_neg = self.dataset_split(rating_np)
        else:
            self.train_data_with_neg = self.load_data_with_neg(
                os.path.join(self.train_valid_test + '/train_data_with_neg.pkl'))
            self.valid_data_with_neg = self.load_data_with_neg(
                os.path.join(self.train_valid_test + '/valid_data_with_neg.pkl'))
            self.test_data_with_neg = self.load_data_with_neg(
                os.path.join(self.train_valid_test + '/test_data_with_neg.pkl'))

        self.train_file = os.path.join(self.train_valid_test + '/train.txt')
        self.valid_file = os.path.join(self.train_valid_test + '/valid.txt')
        self.test_file = os.path.join(self.train_valid_test + '/test.txt')

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_valid_data, self.valid_user_dict = self.load_cf(self.valid_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)

        self.n_users = max(max(max(self.cf_train_data[0]), max(self.cf_test_data[0])), max(self.cf_valid_data[0])) + 1
        self.n_items = max(max(max(self.cf_train_data[1]), max(self.cf_test_data[1])), max(self.cf_valid_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        self.test_batch_size = args.test_batch_size

        ckg_data = self.load_ckg()
        self.ckg_graph, self.ckg_relation_dict = self.construct_ckg(ckg_data, logging)
        ukg_data = self.load_ukg()
        self.ukg_graph, self.ukg_relation_dict = self.construct_ukg(ukg_data, logging)

        self.adj_mat_list = self.build_sparse_relational_graph(self.ckg_relation_dict)

        self.load_multi_modal(logging)

        self.print_info(logging)

    def dataset_split(self, rating_np):
        print('splitting dataset ...')

        valid_ratio = 0.2
        test_ratio = 0.2

        n_ratings = rating_np.shape[0]
        valid_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * valid_ratio), replace=False)
        left = set(range(n_ratings)) - set(valid_indices)
        test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
        train_indices = list(left - set(test_indices))

        user_item_dict = collections.defaultdict(list)
        for i in train_indices:
            user = rating_np[i][0]
            item = rating_np[i][1]
            rating = rating_np[i][2]
            if rating == 1:
                user_item_dict[user].append(item)

        train_indices = [i for i in train_indices if rating_np[i][0] in user_item_dict]
        valid_indices = [i for i in valid_indices if rating_np[i][0] in user_item_dict]
        test_indices = [i for i in test_indices if rating_np[i][0] in user_item_dict]

        train_data = rating_np[train_indices]
        valid_data = rating_np[valid_indices]
        test_data = rating_np[test_indices]

        train_user_dict = {}
        valid_user_dict = {}
        test_user_dict = {}

        with open(self.train_valid_test + '/train_data_with_neg.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open(self.train_valid_test + '/valid_data_with_neg.pkl', 'wb') as f:
            pickle.dump(valid_data, f)
        with open(self.train_valid_test + '/test_data_with_neg.pkl', 'wb') as f:
            pickle.dump(test_data, f)

        for triple in train_data:
            if int(triple[2]) == 0:
                continue
            if triple[0] not in train_user_dict:
                train_user_dict[triple[0]] = list()
            train_user_dict[triple[0]].append(triple[1])
        with open(self.train_valid_test + '/train.txt', 'w') as f:
            for user, items in train_user_dict.items():
                f.write(str(user) + " ")
                for item in items:
                    f.write(str(item) + " ")
                f.write("\n")

        for triple in valid_data:
            if int(triple[2]) == 0:
                continue
            if triple[0] not in valid_user_dict:
                valid_user_dict[triple[0]] = list()
            valid_user_dict[triple[0]].append(triple[1])
        with open(self.train_valid_test + '/valid.txt', 'w') as f:
            for user, items in valid_user_dict.items():
                f.write(str(user) + " ")
                for item in items:
                    f.write(str(item) + " ")
                f.write("\n")

        for triple in test_data:
            if int(triple[2]) == 0:
                continue
            if triple[0] not in test_user_dict:
                test_user_dict[triple[0]] = list()
            test_user_dict[triple[0]].append(triple[1])
        with open(self.train_valid_test + '/test.txt', 'w') as f:
            for user, items in test_user_dict.items():
                f.write(str(user) + " ")
                for item in items:
                    f.write(str(item) + " ")
                f.write("\n")

        return train_data, valid_data, test_data

    def load_data_with_neg(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def load_cf(self, filename):
        user = []
        item = []
        user_dict = defaultdict(list)

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, pos_ids = inter[0], inter[1:]
                pos_ids = list(set(pos_ids))

                for item_id in pos_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = pos_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict

    def load_ckg(self):
        kg_data = pd.read_csv(self.ckg_file, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def load_ukg(self):
        kg_data = pd.read_csv(self.ukg_file, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_ckg(self, kg_data, logging):
        # add inverse relation
        ckg_n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += ckg_n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # add bi-interactions relation
        kg_data['r'] += 2
        self.ckg_n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_nodes = self.n_users + self.n_entities

        # update index add user behind the last entity
        self.cf_train_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32),
            self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32),
            self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = defaultdict(list, {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                                  self.train_user_dict.items()})
        self.test_user_dict = defaultdict(list, {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                                 self.test_user_dict.items()})
        self.valid_user_dict = defaultdict(list, {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                                  self.valid_user_dict.items()})

        self.train_data_with_neg[:, 0] += self.n_entities
        self.valid_data_with_neg[:, 0] += self.n_entities
        self.test_data_with_neg[:, 0] += self.n_entities

        # add bi-interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]
        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)

        self.n_ckg_train = len(self.kg_train_data)
        self.ckg_n_relations = max(self.kg_train_data['r']) + 1

        ckg_graph = nx.MultiDiGraph()
        logging.info("begin load ckg triples ...")
        rd = defaultdict(list)
        for row in self.kg_train_data.iterrows():
            head, relation, tail = row[1]
            ckg_graph.add_edge(head, tail, key=relation)
            rd[relation].append([head, tail])
        return ckg_graph, rd

    def construct_ukg(self, kg_data, logging):
        self.ukg_train_data = kg_data
        self.ukg_n_train = len(self.ukg_train_data)
        self.ukg_n_relations = max(kg_data['r']) + 1

        ukg_graph = nx.MultiDiGraph()
        logging.info("begin load ukg triples ...")
        rd = defaultdict(list)
        for row in self.ukg_train_data.iterrows():
            head, relation, tail = row[1]
            ukg_graph.add_edge(head, tail, key=relation)
            rd[relation].append([head, tail])
        return ukg_graph, rd

    def build_sparse_relational_graph(self, relation_dict):
        adj_mat_list = []
        for r_id in relation_dict.keys():
            np_mat = np.array(relation_dict[r_id])
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(self.n_nodes, self.n_nodes))
            adj_mat_list.append(adj)
        return adj_mat_list

    def load_multi_modal(self, logging):
        logging.info('begin load image_text_pair ...')
        with open(self.pkl_file, 'rb') as f:
            image_text_pair = pickle.load(f)
        image_features = image_text_pair[0]
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = image_text_pair[1]
        text_features /= text_features.norm(dim=-1, keepdim=True)

        self.image_features = image_features
        self.text_features = text_features

    def print_info(self, logging):
        logging.info('n_users:           %d' % self.n_users)
        logging.info('n_items:           %d' % self.n_items)
        logging.info('n_entities:        %d' % self.n_entities)
        logging.info('n_nodes:           %d' % self.n_nodes)
        logging.info('ckg_n_relations:       %d' % self.ckg_n_relations)
        logging.info('n_train:        %d' % self.n_cf_train)
        logging.info('n_test:         %d' % self.n_cf_test)
        logging.info('n_ckg_train:        %d' % self.n_ckg_train)
