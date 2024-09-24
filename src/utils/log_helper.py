import logging
import datetime
import os


def create_log_name(dataset):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    return 'log-{}-{}.log'.format(dataset, formatted_time)


def log_config(args=None, level=logging.DEBUG, console_level=logging.DEBUG, console=True):
    log_dir = '../logs/{}/dim{}/'.format(args.dataset, args.embedding_dim)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    """clean handlers"""
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []

    log_name = create_log_name(args.dataset)
    logpath = os.path.join(log_dir, log_name)
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='[%Y/%m/%d %H:%M:%S]')

    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
