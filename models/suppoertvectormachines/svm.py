import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt


def svm(df, kernel, deg, **kwargs):
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
        else:
            print("Degree must be between 2 and 4")

    elif kernel == "s":
        kernel = "Sigmoid"
        clf = SVC(kernel='sigmoid')

    elif kernel == "g":
        kernel = "Gaussian"
        clf = SVC(kernel='rbf')
    else:
        print("Kernel-function must be chosen")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cv_scores = cross_val_score(clf, X_train, y_train)

    # Cross validate accuracy
    cv_scores_mean = np.mean(cv_scores)
    print("Cross validated accuracy in SVM: ", cv_scores_mean)

    print("This is svm with {} kernel {}: \n".format(kernel, (", with degree: ", deg)))

    f1_micro = f1_score(y_test, y_pred, average='micro')

    # print("F1_MICRO I SVM: ", f1_micro, "CV_SCORES_MEAN I SVM: ", cv_scores_mean)

    print(classification_report(y_test, y_pred))
    return cv_scores_mean, f1_micro


# update filepaths
def run_svm_on_dataset(exp, kernel, deg):
    cv_scores_mean, f1_micro = 0, 0
    if exp == 3:
        print("reached exp = 3")
        # svm_3 = pd.read_csv("data/cleanneddata_exp3")
        df_3 = pd.read_csv("../../data/cleanneddata_exp3.csv")
        cv_scores_mean, f1_micro = svm(df_3, kernel, deg)

    elif exp == 5:
        # df_5 = pd.read_csv("data/cleanneddata_exp3")
        df_5 = pd.read_csv("../../data/cleanneddata_exp5.csv")
        cv_scores_mean, f1_micro = svm(df_5, kernel, deg)

    else:
        print("DT is only implemented for experiment 3 and 5")

    return cv_scores_mean, f1_micro


#For dataset 5

#run_svm_on_dataset(5, "l", 0)
"""
run_svm_on_dataset(5, "s", 0)
run_svm_on_dataset(5, "g", 0)
run_svm_on_dataset(5, "p", 2)
run_svm_on_dataset(5, "p", 3)
run_svm_on_dataset(5, "p", 4)

"""

# For dataset 3
run_svm_on_dataset(3, "l", 0)
# run_svm_on_dataset(3, "s", 0)
# run_svm_on_dataset(3, "g", 0)"""
# run_svm_on_dataset(3, "p", 2)
# run_svm_on_dataset(3, "p", 3)
# run_svm_on_dataset(3, "p", 4)
