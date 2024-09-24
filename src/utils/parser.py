import argparse


def parse_args():
    # steam
    parser = argparse.ArgumentParser(description="McHKGR")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dataset', type=str, default='steam')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument("--l2_lambda", type=float, default=1e-4)

    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument('--context_hops', type=int, default=2)
    parser.add_argument("--contrastive_lambda", type=float, default=1e-1)
    parser.add_argument("--tau", type=float, default=1e-1)

    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--test_batch_size', type=int, default=10000)
    parser.add_argument('--Ks', default='[1, 2, 5, 10, 20, 50, 100]')

    parser.add_argument("--node_dropout", type=bool, default=False, help="consider dropout or not")
    parser.add_argument("--node_dropout_rate", type=float, default=0, help="ratio of dropout")

    # movielens
    # parser = argparse.ArgumentParser(description="McHKGR")
    # parser.add_argument('--seed', type=int, default=2024)
    # parser.add_argument('--dataset', type=str, default='movielens')
    # parser.add_argument('--batch_size', type=int, default=1024)
    # parser.add_argument('--embedding_dim', type=int, default=64)
    # parser.add_argument("--l2_lambda", type=float, default=1e-4)
    #
    # parser.add_argument('--epoch', type=int, default=1000)
    # parser.add_argument("--cuda", type=bool, default=True)
    # parser.add_argument('--lr', type=float, default=1e-03)
    # parser.add_argument("--gpu_id", type=int, default=1)
    # parser.add_argument('--context_hops', type=int, default=2)
    # parser.add_argument("--contrastive_lambda", type=float, default=1e-1)
    # parser.add_argument("--tau", type=float, default=5e-2)
    #
    # parser.add_argument('--data_dir', type=str, default='../data/')
    # parser.add_argument('--test_batch_size', type=int, default=10000)
    # parser.add_argument('--Ks', default='[1, 2, 5, 10, 20, 50, 100]')
    #
    # parser.add_argument("--node_dropout", type=bool, default=False, help="consider dropout or not")
    # parser.add_argument("--node_dropout_rate", type=float, default=0, help="ratio of dropout")

    return parser.parse_args()
