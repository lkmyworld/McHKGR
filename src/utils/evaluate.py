import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score


def evaluate_valid(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    valid_user_dict = dataloader.valid_user_dict

    model.eval()

    user_ids = list(valid_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    for batch_user_ids in user_ids_batches:
        batch_user_ids = batch_user_ids.to(device)

        with torch.no_grad():
            batch_scores = model(batch_user_ids, item_ids, mode='topk')  # (n_batch_users, n_items)

        batch_scores = batch_scores.cpu()
        batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, valid_user_dict, batch_user_ids.cpu().numpy(),
                                          item_ids.cpu().numpy(), Ks)

        cf_scores.append(batch_scores.numpy())
        for k in Ks:
            for m in metric_names:
                metrics_dict[k][m].append(batch_metrics[k][m])

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def evaluate_test(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()

    user_ids = list(test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]

    n_items = dataloader.n_items
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}

    for batch_user_ids in user_ids_batches:
        batch_user_ids = batch_user_ids.to(device)

        with torch.no_grad():
            batch_scores = model(batch_user_ids, item_ids, mode='topk')  # (n_batch_users, n_items)

        batch_scores = batch_scores.cpu()
        batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(),
                                          item_ids.cpu().numpy(), Ks)

        cf_scores.append(batch_scores.numpy())
        for k in Ks:
            for m in metric_names:
                metrics_dict[k][m].append(batch_metrics[k][m])

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def calc_metrics_at_k(cf_scores, train_user_dict, test_user_dict, user_ids, item_ids, Ks):
    """
    cf_scores: (n_users, n_items)
    """
    test_pos_item_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, u in enumerate(user_ids):
        train_pos_item_list = train_user_dict[u]
        test_pos_item_list = test_user_dict[u]
        cf_scores[idx][train_pos_item_list] = -np.inf
        test_pos_item_binary[idx][test_pos_item_list] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)  # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(user_ids)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    metrics_dict = {}
    for k in Ks:
        metrics_dict[k] = {}
        metrics_dict[k]['precision'] = precision_at_k_batch(binary_hit, k)
        metrics_dict[k]['recall'] = recall_at_k_batch(binary_hit, k)
        metrics_dict[k]['ndcg'] = ndcg_at_k_batch(binary_hit, k)
    return metrics_dict


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
    ndcg = (dcg / idcg)
    return ndcg


def ctr_eval(model, data, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    f1_list = []
    while start + batch_size <= data.shape[0]:
        users = data[start:start + batch_size, 0]
        items = data[start:start + batch_size, 1]
        with torch.no_grad():
            scores = model(users, items, mode="ctr")
        scores = scores.cpu().numpy()
        labels = data[start:start + batch_size, 2]

        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        acc = np.mean(np.equal(scores, labels))

        auc_list.append(auc)
        acc_list.append(acc)
        f1_list.append(f1)

        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(acc_list)), float(np.mean(f1_list))
