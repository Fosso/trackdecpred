import argparse
from models.decisiontree.dt import run_dt_on_dataset
from models.knn.knn import run_knn
from models.suppoertvectormachines.svm import run_svm_on_dataset


def parse_args():
    """
    :return: arguments, configuration dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-knn", "--knn", default=False, nargs='?', const=True)
    parser.add_argument("-dt", "--dt", default=False, nargs='?', const=True)
    parser.add_argument("-svm", "--svm", default=False, nargs='?', const=True)
    parser.add_argument("-exp", "--experiment", type=int, default=False, nargs='?')
    parser.add_argument("-o", "--optimal", default=False, nargs='?', const=True)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    # knn
    if args.knn and args.experiment == 1:
        model = run_knn(103, args.experiment, args.optimal)
    elif args.knn and args.experiment == 2:
        model = run_knn(70, args.experiment, args.optimal)
    elif args.knn and args.experiment == 3:
        model = run_knn(80, args.experiment, args.optimal)
    elif args.knn and args.experiment == 4:
        model = run_knn(11, args.experiment, args.optimal)
    elif args.knn and args.experiment == 5:
        model = run_knn(11, args.experiment, args.optimal)

    # dt
    elif args.dt and args.experiment == 3:
        model = run_dt_on_dataset(args.experiment, 10, args.optimal)
    elif args.dt and args.experiment == 5:
        model = run_dt_on_dataset(args.experiment, 10, args.optimal)

    # svm
    elif args.svm and args.experiment == 3:
        model = run_svm_on_dataset(args.experiment, 'g', 0)
    elif args.svm and args.experiment == 5:
        model = run_svm_on_dataset(args.experiment, 'p', 3)
    else:
        pass
