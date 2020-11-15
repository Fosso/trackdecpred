import matplotlib
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

    accuracy_dt = clf.score(X_test, y_test)
    print("Accuracy: ", accuracy_dt)

    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()

    print(classification_report(y_test, y_pred))


# TODO: return cv og f1


# update filepaths
def run_dt_on_dataset(exp):
    if exp == 3:
        # df_3 = pd.read_csv("data/cleanneddata_exp3")
        df_3 = pd.read_csv("../../data/cleanneddata_exp3.csv")
        dt(df_3)


    elif exp == 5:
        # df_5 = pd.read_csv("data/cleanneddata_exp3")
        df_5 = pd.read_csv("../../data/cleanneddata_exp5.csv")
        dt(df_5)
    else:
        print("DT is only implemented for experiment 3 and 5")

run_dt_on_dataset(3)

# exp_1: 9%
# exp_2: 31%
# exp_3: 39%
# exp_4: 96%
# exp_5: 91%
