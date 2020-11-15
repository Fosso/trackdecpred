import argparse
# from models.decisiontree.dt import run_dt
from models.knn.knn import run_knn
# from models.dt.dt_basic import run_dt


def parse_args():
    """
    :return: arguments, configuration dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", default=False, nargs="?", const=True, help="just a test")
    parser.add_argument("-knn", "--knn", default=False, nargs='?', const=True)
    parser.add_argument("-dt", "--dt", default=False, nargs='?', const=True)
    parser.add_argument("-exp", "--exp", type=str, default=[1], nargs='?')
    parser.add_argument("-o", "--optimal", type=str, default=False, nargs='?')
    args = parser.parse_args()
    # print(args)

    return args


if __name__ == '__main__':

    args = parse_args()
    if args.knn:
        model = run_knn(6, 3)
    # elif args.dt:
        #   model = run_dt(2)
    else:
        pass
