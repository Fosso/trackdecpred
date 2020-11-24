import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from models.knn.knn_year import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import plot_confusion_matrix
import time

start_time = time.time()


# update filepaths
def run_knn(param_k, exp, optimal):
    print("---start of knn---")
    print("k : ", param_k, "exp: ", exp)
    df = pd.read_csv("data/cleanneddata_exp2.csv")
    # df = pd.read_csv("../../data/cleanneddata_exp3.csv")
    # df = []
    # optimal k: 103
    if exp == 1:
        knn_exp1(param_k)
    else:

        # optimal k: 70
        if exp == 2:
            df = pd.read_csv("data/cleanneddata_exp2.csv")
            # df = pd.read_csv("../../data/cleanneddata_exp2.csv")
        # optimal k: 80
        elif exp == 3:
            df = pd.read_csv("data/cleanneddata_exp3.csv")
            # df = pd.read_csv("../../data/cleanneddata_exp3.csv")
        # optimal k: 11
        elif exp == 4:
            df = pd.read_csv("data/cleanneddata_exp4.csv")
            # df = pd.read_csv("../../data/cleanneddata_exp4.csv")

        # optimal k: 11
        elif exp == 5:
            df = pd.read_csv("data/cleanneddata_exp5.csv")
            # df = pd.read_csv("../../data/cleanneddata_exp5.csv")

        y = np.array(df["decade"])

        X = np.array(df.drop(["decade"], axis=1))

        # Divide the set in 20% for testing 80% for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # build classifier with k = 12
        # k for vanlig datasett 1930-1970 = 56

        # If main.py is executed with -o flag
        if optimal:
            x_axis2, y_axis2 = run_knn_with_find_k(X_train, X_test, y_train, y_test)

            # create dataframe using k-value and accuracy
            df_k = pd.DataFrame({"k-value": x_axis2, "accuracy": y_axis2})

            # Draw line plot
            sns.set_style("darkgrid")
            sns.lineplot(x="k-value", y="accuracy", dashes=False, marker="o", data=df_k)

            # to show graph
            plt.show()
            # plt.savefig("results/knn/optimal_k_exp5.png")

        # This is knn with predefined k
        else:

            k = param_k
            clf = neighbors.KNeighborsClassifier(k, algorithm='auto', weights='uniform', p=2)
            # print("INFO::", clf.get_params())

            cv_scores = cross_val_score(clf, X_train, y_train)
            clf.fit(X_train, y_train)

            # Cross validate accuracy
            cv_scores_mean = np.mean(cv_scores)
            print("Cross validated accuracy: ", cv_scores_mean)

            y_pred = clf.predict(X_test)

            # f1 weighted (macro) for exp 3
            if exp == 3:
                f1 = f1_score(y_test, y_pred, average='weighted')


            # f1 micro for exp 5 (and all the other experiments)
            else:
                f1 = f1_score(y_test, y_pred, average='micro')

            # Prints confusion matrix with presentation view.
            plot_confusion_matrix(clf, X_test, y_test)

            # plt.savefig("../../results/knn/confusion_matrix_exp3.png")
            # plt.savefig("../../results/knn/confusion_matrix_exp5.png")

            print("---end of knn---")

            return cv_scores_mean, f1

            # return classification_report(y_test, y_pred, output_dict=True)


def run_knn_with_find_k(X_train, X_test, y_train, y_test):
    x_axis = []
    y_axis = []

    for k in range(1, 20, 1):
        # values for k
        x_axis.append(k)

        # build classifiers
        clf = neighbors.KNeighborsClassifier(k)

        # adopt the data x and y training sets
        clf.fit(X_train, y_train)

        # to test the accuracy
        accuracy = clf.score(X_test, y_test)

        # values for accuracy
        y_axis.append(accuracy)

    return x_axis, y_axis

# With optimal k for relevant experiments with time stamp
# run_knn(80, 3)
# print("---kNN for experiment 3 had a {} seconds execution time---".format(time.time() - start_time))

# run_knn(11, 5)
# print("---kNN for experiment 5 had a {} seconds execution time---".format(time.time() - start_time))
