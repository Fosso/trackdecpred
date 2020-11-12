import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd


def knn_exp1(param_k):
    k = param_k
    # attributa or featrues withouth the "solution"/ class/ decade.

    df = pd.read_csv("data/cleanneddata_exp1.csv")
    X = np.array(df.drop(["year"], axis=1))

    # solution is stored to an array y
    y = np.array(df["year"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(k)

    # adopt the dataen x and y trainingsets
    clf.fit(X_train, y_train)

    # predict and loop over to create an accuracy_list
    prediction = clf.predict(X_test)
    correct = 0
    interval = 5
    accuracy_list = []

    for i in range(len(X_test)):
        if y_test[i] + interval >= prediction[i] >= y_test[i] - interval:
            correct += 1
            accuracy2 = (correct / float(len(X_test)) * 100.0)
            # print("Accuracy inni for: ", accuracy2)
            accuracy_list.append(accuracy2)

    print("accuracy: ", accuracy2)
    if param_k == 10:
        x_axis2, y_axis2 = run_knn_with_find_k(X_train, X_test, y_train, y_test)

        df_k = pd.DataFrame({"k-value": x_axis2, "accuracy": y_axis2})

        sns.set_style("darkgrid")
        sns.lineplot(x="k-value", y="accuracy", dashes=False, marker="o", data=df_k)

        # plt.savefig("results/knn/optimal_k_exp1.png")
        plt.show()

def run_knn_with_find_k(X_train, X_test, y_train, y_test):
    x_axis = []
    y_axis = []

    for k in range (90, 110, 1):
        # k values
        x_axis.append(k)

        # build classifiere
        clf = neighbors.KNeighborsClassifier(k)

        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)
        correct = 0
        interval = 5
        accuracy_list = []
        for i in range(len(X_test)):
            if y_test[i] + interval >= prediction[i] >= y_test[i] - interval:
                correct += 1

        accuracy2 = (correct / float(len(X_test)) * 100.0)
        y_axis.append(accuracy2)

        # y_axis.append(accuracy_list)

    return x_axis, y_axis
