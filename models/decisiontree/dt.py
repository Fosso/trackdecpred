import matplotlib
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from IPython.display import display


def dt(df):
    y = np.array(df["decade"])
    X = np.array(df.drop(["decade"], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    cv_scores = cross_val_score(clf, X_train, y_train)

    # Cross validate accuracy
    cv_scores_mean = np.mean(cv_scores)
    print("Cross validated accuracy: ", cv_scores_mean)


    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()

    print(classification_report(y_test, y_pred))
    f1_micro = f1_score(y_test, y_pred, average='micro')

    # print("F1_MICRO I DT: ", f1_micro, "CV_SCORES_MEAN I DT: ", cv_scores_mean)
    return cv_scores_mean, f1_micro


# update filepaths
def run_dt_on_dataset(exp):
    print("---start of dt---")
    cv_scores_mean, f1_micro = 0, 0
    if exp == 3:
        # df_3 = pd.read_csv("data/cleanneddata_exp3")
        df_3 = pd.read_csv("../../data/cleanneddata_exp3.csv")
        cv_scores_mean, f1_micro = dt(df_3)


    elif exp == 5:
        # df_5 = pd.read_csv("data/cleanneddata_exp3")
        df_5 = pd.read_csv("../../data/cleanneddata_exp5.csv")
        cv_scores_mean, f1_micro = dt(df_5)
    else:
        print("DT is only implemented for experiment 3 and 5")


    return cv_scores_mean, f1_micro

# run_dt_on_dataset(5)

# exp_1: 9%
# exp_2: 31%
# exp_3: 39%
# exp_4: 96%
# exp_5: 91%
