import inspect

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import time

start_time = time.time()


def svm(df, kernel, deg, average, **kwargs):
    print("---start of svm---")
    global clf
    y = np.array(df["decade"])
    X = np.array(df.drop(["decade"], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    if kernel == "l":
        print("reached kernel == l")
        kernel = "linear"
        clf = SVC(kernel='linear')
    elif kernel == "p":
        kernel = "polynomial"
        if deg == 2:
            clf = SVC(kernel='poly', degree=2)
        elif deg == 3:
            clf = SVC(kernel='poly', degree=3)
        elif deg == 4:
            clf = SVC(kernel='poly', degree=4)
        elif deg == 5:
            clf = SVC(kernel='poly', degree=5)
        else:
            print("Degree must be between 2 and 5")

    elif kernel == "s":
        kernel = "Sigmoid"
        clf = SVC(kernel='sigmoid')

    elif kernel == "g":
        kernel = "Gaussian"
        clf = SVC(kernel='rbf')
    else:
        print("Kernel-function must be chosen")


    # Training and prediction
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Cross validate and return accuracy mean
    cv_scores = cross_val_score(clf, X_train, y_train)
    cv_scores_mean = np.mean(cv_scores)

    # print("Cross validated accuracy in SVM: ", cv_scores_mean)
    # print("This is svm with {} kernel {}: \n".format(kernel, (", with degree: ", deg)))

    f1 = f1_score(y_test, y_pred, average=average)
    print("f1::::: ", f1)



    # create confusion matrix
    plot_confusion_matrix(clf, X_test, y_test)

    print(cv_scores_mean)
    # print confusion matrix to png in resutls
    # plt.savefig("../../results/svm/confusion_matrix_exp3.png")
    # plt.savefig("../../results/svm/confusion_matrix_exp5_p4.png")

    print("---end of svm---")

    return cv_scores_mean, f1


# Update file paths dependant on if your running from main og locally.
def run_svm_on_dataset(exp, kernel, deg):
    cv_scores_mean, f1_micro = 0, 0
    if exp == 3:
        df_3 = pd.read_csv("data/cleanneddata_exp3.csv")
        # df_3 = pd.read_csv("../../data/cleanneddata_exp3.csv")
        average = "weighted"
        cv_scores_mean, f1 = svm(df_3, kernel, deg, average)

    elif exp == 5:
        df_5 = pd.read_csv("data/cleanneddata_exp5.csv")
        # df_5 = pd.read_csv("../../data/cleanneddata_exp5.csv")
        average = "micro"
        cv_scores_mean, f1 = svm(df_5, kernel, deg, average)

    else:
        print("DT is only implemented for experiment 3 and 5")

    return cv_scores_mean, f1


# With optimal methods for relevant experiments with time stamp
# run_svm_on_dataset(3, "g", 0)
# print("---SVM for experiment 3 had a {} seconds execution time---".format(time.time() - start_time))

# run_svm_on_dataset(5, "p", 3)
# print("---SVM for experiment 5 had a {} seconds execution time---".format(time.time() - start_time))


# For dataset 5

# run_svm_on_dataset(5, "l", 0)
"""
run_svm_on_dataset(5, "s", 0)
run_svm_on_dataset(5, "g", 0)
run_svm_on_dataset(5, "p", 2)
run_svm_on_dataset(5, "p", 3)
run_svm_on_dataset(5, "p", 4)

"""

# For dataset 3

#run_svm_on_dataset(3, "l", 0)
"""
run_svm_on_dataset(3, "s", 0)
run_svm_on_dataset(3, "g", 0)
run_svm_on_dataset(3, "p", 2)
run_svm_on_dataset(3, "p", 3)
run_svm_on_dataset(3, "p", 4)
"""
