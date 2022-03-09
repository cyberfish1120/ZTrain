import argparse


def arg_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', default=1e-5)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--epochs', default=10)

    args = parser.parse_args()

    return args
