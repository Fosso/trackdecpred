import matplotlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


# update filepaths
def run_dt(exp):
    #df = pd.read_csv("data/cleanneddata_exp2")
    df = pd.read_csv("../../data/cleanneddata_exp1.csv")

    if exp == 1:
        dataset = pd.read_csv("../../data/cleanneddata_exp1.csv")

        y = np.array(dataset["year"])
        X = np.array(dataset.drop(["year"], axis=1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    else:
        print("Reached start of else")
        if exp == 2:
            #df = pd.read_csv("data/cleanneddata_exp2")
            df = pd.read_csv("../../data/cleanneddata_exp2.csv")
        elif exp == 3:
            #df = pd.read_csv("data/cleanneddata_exp3")
            df = pd.read_csv("../../data/cleanneddata_exp3.csv")

        elif exp == 4:
            #df = pd.read_csv("data/cleanneddata_exp4")
            df = pd.read_csv("../../data/cleanneddata_exp4.csv")
        elif exp == 5:
            #df = pd.read_csv("data/cleanneddata_exp5")
            df = pd.read_csv("../../data/cleanneddata_exp5.csv")

        y = np.array(df["decade"])
        X = np.array(df.drop(["decade"], axis=1))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

run_dt(5)

# exp_1: 9%
# exp_2: 31%
# exp_3: 39%
# exp_4: 96%
# exp_5: 91%
