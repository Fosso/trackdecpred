import argparse
from models.knn.knn_basic import run_knn
# from models.dt.dt_basic import run_dt


def parse_args():
    """
    :return: arguments, configuration dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-knn", "--knn", default=False, nargs='*', const=True)
    parser.add_argument("-dt", "--dt", default=False, nargs='*', const=True)
    parser.add_argument("-ex", "--ex", type=str, default=["1"], nargs=1)
    parser.add_argument("-o", "--optimal", type=str, default=False, nargs='?')
    parser.add_argument()
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.knn:
        model = run_knn(10, args.ex)
    # else:
    #    model = run_dt()
    else:
        pass
