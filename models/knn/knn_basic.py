import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import time
from models.knn.optimal_k import find_k
# from .optimal_k import find_k


# update filepaths
def run_knn(param_k, exp, **kwargs):
    start_time = time.time()
    df = pd.read_csv("../../data/normalizeddata.csv")
    if exp == 1:
        df = pd.read_csv("../../data/normalizeddata.csv")
    elif exp == 2:
        df = pd.read_csv("../../data/cleaneddata_exp2")
    elif exp == 3:
        df = pd.read_csv("../../data/cleaneddata_exp3")
    elif exp == 4:
        df = pd.read_csv("../../data/cleaneddata_exp5")
    elif exp == 5:
        df = pd.read_csv("../../data/cleaneddata_exp5")

    y = np.array(df["decade"])

    X = np.array(df.drop(["decade"], axis=1))

    # Divide the set in 20% for testing 80% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # build classifier with k = 12
    # k for vanlig datasett 1930-1970 = 56
    if 'optimal' in kwargs:
        k = find_k(extra_path + "../../data/normalizeddata.csv"
    else:
        k = param_k

    clf = neighbors.KNeighborsClassifier(k)

    cv_scores = cross_val_score(clf, X_train, y_train)
    clf.fit(X_train, y_train)

    # to test the accuracy
    cv_scores_mean = np.mean(cv_scores)
    print(cv_scores, "\n""mean =", "{:.2f}".format(cv_scores_mean))

    # to test the accuracy
    accuracy = clf.score(X_test, y_test)

    print("dette er treffsikkerheten:", accuracy)

    print("Program took", time.time() - start_time, "s to run")


if __name__ == '__main__':
    run_knn(5, 1)
