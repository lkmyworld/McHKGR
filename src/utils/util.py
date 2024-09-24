import torch
import numpy as np
import random
from utils.log_helper import *


def set_log(args):
    log_config(args=args, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info('=' * 30)
    logging.info(' ' * 9 + 'Dataset {}'.format(args.dataset) + ' ' * 9)
    logging.info('=' * 30)
    args_dict = args.__dict__
    parameters = 'Hyper-parameters: \n'
    for idx, (key, value) in enumerate(args_dict.items()):
        if idx == len(args_dict) - 1:
            parameters += '\t\t{}: {}'.format(key, str(value))
        else:
            parameters += '\t\t{}: {}\n'.format(key, str(value))
    logging.info(parameters)
    logging.info('=' * 32)


def set_seed(seed_num):
    seed = seed_num
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def shuffle_train(logging, train_cf_pairs, args, data):
    index = np.arange(len(train_cf_pairs))
    np.random.shuffle(index)
    train_cf_pairs = train_cf_pairs[index]
    all_feed_data = get_feed_data(train_cf_pairs, args.num_neg_sample, data.train_user_dict,
                                  data.test_user_dict, data.n_items)
    return all_feed_data, train_cf_pairs


def get_feed_data(train_cf_pairs, num_neg_sample, train_user_dict, test_user_dict, n_items):
    feed_dict = {}
    entity_pairs = train_cf_pairs
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    train_dict = train_user_dict.copy()
    test_dict = test_user_dict.copy()
    feed_dict['neg_items'] = torch.LongTensor(
        negative_sampling(entity_pairs, num_neg_sample, train_dict, test_dict, n_items))
    return feed_dict


def negative_sampling(user_item, num_neg_sample, train_dict, test_dict, n_items):
    neg_items = list()
    for user, _ in user_item.cpu().numpy():
        user = int(user)
        each_negs = list()
        neg_item = np.random.randint(low=0, high=n_items, size=num_neg_sample)
        if len(set(neg_item) & set(train_dict[user])) == 0 and len(set(neg_item) & set(test_dict[user])) == 0:
            each_negs += list(neg_item)
        else:
            neg_item = list(set(neg_item) - set(train_dict[user]) - set(test_dict[user]))
            each_negs += neg_item
            while len(each_negs) < num_neg_sample:
                n1 = np.random.randint(low=0, high=n_items, size=1)[0]
                if n1 not in train_dict[user] and n1 not in test_dict[user] and n1 not in each_negs:
                    each_negs += [n1]
        neg_items.append(each_negs)
    return neg_items


def L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def get_total_parameters(model):
    return str(sum(p.numel() for p in model.parameters() if p.requires_grad))


def get_inputs(data, start, end, device):
    user_index = data[start:end, 0]
    item_index = data[start:end, 1]
    labels = data[start:end, 2]

    return torch.LongTensor(user_index).to(device), \
        torch.LongTensor(item_index).to(device), \
        torch.FloatTensor(labels).to(device)
