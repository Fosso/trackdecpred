import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
import sklearn
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd
import time


def knn_exp1(df, param_k):

    k = param_k
    # attributa or featrues withouth the "solution"/ class/ decade.
    X = np.array(df.drop(["year"], axis=1))

    # solution is stores to an array y
    y = np.array(df["year"])

    # Devide the set in 20% for testing 80% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(k)

    # adopt the dataen x and y trainingsets
    clf.fit(X_train, y_train)

    # Checking performance on the test set
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(clf.score(X_test, y_test)))
    # Calculate new accuracy percentage

    prediction = clf.predict(X_test)
    correct = 0
    interval = 20
    for i in range(len(X_test)):
        if y_test[i] + interval >= prediction[i] >= y_test[i] - interval:
            correct += 1

    # Checking performance on the test set
    print('Accuracy of K-NN classifier on test set with a range of +-20 years: {:.2f}'
          .format(correct / float(len(X_test)) * 100.0))

