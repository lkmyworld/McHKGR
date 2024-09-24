import math
from modules.McHKGR import McHKGR
from utils.util import *
from utils.evaluate import *
from time import time
from tqdm import tqdm


def train(args, data, device):
    """load model"""
    logging.info("begin load model ...")
    model = McHKGR(args, data, device)
    logging.info("model parameters: " + get_total_parameters(model))
    model.to(device)
    logging.info(model)

    """set parameters"""
    Ks = eval(args.Ks)

    """prepare optimizer"""
    logging.info("begin prepare optimizer ...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """training"""
    train_data = data.train_data_with_neg
    valid_data = data.valid_data_with_neg
    test_data = data.test_data_with_neg
    batch_num = math.ceil(len(train_data) / args.batch_size)

    logging.info('batch_num: %d' % batch_num)
    logging.info("start training ...")

    for epoch in range(1, args.epoch + 1):
        torch.cuda.empty_cache()
        if epoch % 1 == 0:
            index = np.arange(len(train_data))
            np.random.shuffle(index)
            train_data = train_data[index]

        model.train()
        time_start = time()

        loss, s = 0, 0
        for i in tqdm(range(batch_num), desc="Train processing..."):
            user_index, item_index, labels = get_inputs(train_data, i * args.batch_size, (i + 1) * args.batch_size,
                                                        device)
            batch_loss = model(user_index, item_index, labels, mode="train")

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            s += args.batch_size

        time_train = time() - time_start

        if epoch % 1 == 0 or epoch == 1:
            model.eval()

            time_start = time()
            valid_auc, valid_acc, valid_f1 = ctr_eval(model, valid_data, args.batch_size)
            test_auc, test_acc, test_f1 = ctr_eval(model, test_data, args.batch_size)
            _, valid_metrics_dict = evaluate_valid(model, data, Ks, device)
            _, test_metrics_dict = evaluate_test(model, data, Ks, device)
            time_eval = time() - time_start

            logging.info('Epoch: %d  loss: %.4f  train_time: %.2fs  eval_time: %.2fs' \
                         % (epoch, loss / batch_num, time_train, time_eval))
            logging.info('valid auc: %.4f  acc: %.4f  f1: %.4f    test auc: %.4f  acc: %.4f  f1: %.4f' % \
                         (valid_auc, valid_acc, valid_f1, test_auc, test_acc, test_f1))
            logging.info('valid P: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}] | '
                         'R: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}] | '
                         'N: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'
                         .format(valid_metrics_dict[1]['precision'], valid_metrics_dict[2]['precision'],
                                 valid_metrics_dict[5]['precision'], valid_metrics_dict[10]['precision'],
                                 valid_metrics_dict[20]['precision'], valid_metrics_dict[50]['precision'],
                                 valid_metrics_dict[100]['precision'],
                                 valid_metrics_dict[1]['recall'], valid_metrics_dict[2]['recall'],
                                 valid_metrics_dict[5]['recall'], valid_metrics_dict[10]['recall'],
                                 valid_metrics_dict[20]['recall'], valid_metrics_dict[50]['recall'],
                                 valid_metrics_dict[100]['recall'],
                                 valid_metrics_dict[1]['ndcg'], valid_metrics_dict[2]['ndcg'],
                                 valid_metrics_dict[5]['ndcg'], valid_metrics_dict[10]['ndcg'],
                                 valid_metrics_dict[20]['ndcg'], valid_metrics_dict[50]['ndcg'],
                                 valid_metrics_dict[100]['ndcg']
                                 ))
            logging.info('test  P: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}] | '
                         'R: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}] | '
                         'N: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'
                         .format(test_metrics_dict[1]['precision'], test_metrics_dict[2]['precision'],
                                 test_metrics_dict[5]['precision'], test_metrics_dict[10]['precision'],
                                 test_metrics_dict[20]['precision'], test_metrics_dict[50]['precision'],
                                 test_metrics_dict[100]['precision'],
                                 test_metrics_dict[1]['recall'], test_metrics_dict[2]['recall'],
                                 test_metrics_dict[5]['recall'], test_metrics_dict[10]['recall'],
                                 test_metrics_dict[20]['recall'], test_metrics_dict[50]['recall'],
                                 test_metrics_dict[100]['recall'],
                                 test_metrics_dict[1]['ndcg'], test_metrics_dict[2]['ndcg'],
                                 test_metrics_dict[5]['ndcg'], test_metrics_dict[10]['ndcg'],
                                 test_metrics_dict[20]['ndcg'], test_metrics_dict[50]['ndcg'],
                                 test_metrics_dict[100]['ndcg']
                                 ))
            logging.info("-")
