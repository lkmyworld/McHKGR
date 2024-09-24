from utils.parser import parse_args
from utils.util import *
from utils.data_loader import Dataloader
from train import train

if __name__ == '__main__':
    """read args"""
    args = parse_args()
    """set seed"""
    set_seed(args.seed)

    """set log"""
    set_log(args)

    """load data"""
    data = Dataloader(args, logging)

    """start training"""
    device = torch.device("cuda:" + str(args.gpu_id)) if args.cuda else torch.device("cpu")
    train(args, data, device)
