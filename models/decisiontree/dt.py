import pandas as pd
from pandas import DataFrame
import seaborn as sns  # for data visualization
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import time

start_time = time.time()


def dt(df, average, md, optimal):
    y = np.array(df["decade"])
    X = np.array(df.drop(["decade"], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # if optimal arg is true, run iterative experiment on max depth
    if optimal:
        x_axis2, y_axis2 = run_dt_with_find_depth(df)

        # create dataframe using max depth value and accuracy
        df_md = pd.DataFrame({"max_depth_value": x_axis2, "accuracy": y_axis2})

        # draw line plot
        sns.set_style("darkgrid")
        sns.lineplot(x="max_depth_value", y="accuracy", dashes=False, marker="o", data=df_md)

        # to show graph
        plt.show()
        # plt.savefig("results/dt/optimal_md_exp5.png")
        # should only be used to find optimal n, therfore doesnt return anything, 0,0 to stop it from crash.
        return 0, 0
    else:

        # set max_depth
        clf = tree.DecisionTreeClassifier(max_depth=md)
        clf = clf.fit(X_train, y_train)

        # plot tree
        tree.plot_tree(clf)
        # plt.show()
        # To save example of tree with max depth 2
        # plt.savefig("results/dt/tree_md_2_example.png")

        y_pred = clf.predict(X_test)

        cv_scores = cross_val_score(clf, X_train, y_train)

        # cross validate accuracy
        cv_scores_mean = np.mean(cv_scores)
        print("Cross validated accuracy: ", cv_scores_mean)

        plot_confusion_matrix(clf, X_test, y_test)

        # plt.savefig("../../results/dt/confusion_matrix_exp3.png")
        # plt.savefig("../../results/dt/confusion_matrix_exp5.png")

        # print(classification_report(y_test, y_pred))
        f1 = f1_score(y_test, y_pred, average=average)
        print("F1-Score: ", f1)
        print("---end of dt---")

        return cv_scores_mean, f1


# Update file paths dependant on if your running from main or locally.
def run_dt_on_dataset(exp, md, optimal):
    cv_scores_mean, f1 = 0, 0
    if exp == 3:
        df_3 = pd.read_csv("data/cleanneddata_exp3.csv")
        # df_3 = pd.read_csv("../../data/cleanneddata_exp3.csv")
        average = "weighted"
        cv_scores_mean, f1 = dt(df_3, average, md, optimal)

    elif exp == 5:
        df_5 = pd.read_csv("data/cleanneddata_exp5.csv")
        # df_5 = pd.read_csv("../../data/cleanneddata_exp5.csv")
        average = "micro"
        cv_scores_mean, f1 = dt(df_5, average, md, optimal)
    elif optimal and exp == 3:
        df_optimal = pd.read_csv("data/cleanneddata_exp3.csv")
        average = "micro"
        run_dt_with_find_depth(df_optimal)
    elif optimal and exp == 5:
        df_optimal = pd.read_csv("data/cleanneddata_exp5.csv")
        average = "micro"
        run_dt_with_find_depth(df_optimal)

    else:
        print("DT is only implemented for experiment 3 and 5")

    return cv_scores_mean, f1


def run_dt_with_find_depth(df):
    # vars for storing accuracy for different values for max depth
    x_axis = []
    y_axis = []

    y = np.array(df["decade"])

    X = np.array(df.drop(["decade"], axis=1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # loop for iterative experiment for max depth, values from 1-20, can be changed
    for d in range(1, 20, 1):
        # append values for max depth
        x_axis.append(d)

        # build classifier and set max_depth
        clf = tree.DecisionTreeClassifier(max_depth=d)
        clf = clf.fit(X_train, y_train)

        # cross validate accurcary for current iteratiation
        cv_scores = cross_val_score(clf, X_train, y_train)
        cv_scores_mean = np.mean(cv_scores)

        # append values for accuracy
        y_axis.append(cv_scores_mean)

    return x_axis, y_axis

# With optimal k for relevant experiments with time stamp
# run_dt_on_dataset(3)
# print("---kNN for experiment 3 had a {} seconds execution time---".format(time.time() - start_time))

# run_dt_on_dataset(5)
# print("---kNN for experiment 5 had a {} seconds execution time---".format(time.time() - start_time))

# Results
# exp_1: 9%
# exp_2: 31%
# exp_3: 39%
# exp_4: 96%
# exp_5: 91%
