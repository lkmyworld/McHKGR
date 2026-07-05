import collections
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import scipy.sparse as sp
import pickle


class Dataloader(object):
    def __init__(self, args, logging):
        self.args = args
        self.dataset = args.dataset
        self.dataset_dir = str(os.path.join(args.data_dir, args.dataset))

        self.train_file = os.path.join(self.dataset_dir + '/train.txt')
        self.valid_file = os.path.join(self.dataset_dir + '/valid.txt')
        self.test_file = os.path.join(self.dataset_dir + '/test.txt')

        self.ckg_file = os.path.join(self.dataset_dir, "kg_final.txt")
        self.ukg_file = os.path.join(self.dataset_dir, "ukg_final.txt")
        self.pkl_file = os.path.join(self.dataset_dir, 'image_text_pair.pkl')

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_valid_data, self.valid_user_dict = self.load_cf(self.valid_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)

        self.n_users = max(max(max(self.cf_train_data[0]), max(self.cf_test_data[0])), max(self.cf_valid_data[0])) + 1
        self.n_items = max(max(max(self.cf_train_data[1]), max(self.cf_test_data[1])), max(self.cf_valid_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])
        self.test_batch_size = args.test_batch_size

        ckg_data = self.load_ckg()
        ukg_data = self.load_ukg()

        self.ckg_graph, self.ckg_relation_dict = self.construct_kg(ckg_data, logging)
        self.ukg_graph, self.ukg_relation_dict = self.construct_kg(ukg_data, logging)

        self.adj_mat_list = self.build_sparse_relational_graph(self.ckg_relation_dict)

        self.load_multi_modal(logging)

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
