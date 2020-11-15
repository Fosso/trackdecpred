import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
from models.knn.knn_year import *
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import plot_confusion_matrix

# update filepaths
def run_knn(param_k, exp, **kwargs):
    # df = pd.read_csv("data/cleanneddata_exp2.csv")
    df = pd.read_csv("../../data/cleanneddata_exp3.csv")
    # df = []
    print("exp: ", exp)
    # optimal k: 103
    if exp == 1:
        print("Reached start of exp 1")
        knn_exp1(param_k)

    else:
        print("Reached start of else")
        # optimal k: 70
        if exp == 2:
            df = pd.read_csv("data/cleanneddata_exp2.csv")
            print("første if, altså exp2")
            # df = pd.read_csv("../../data/cleanneddata_exp2.csv")
        # optimal k: 70
        elif exp == 3:
            # df = pd.read_csv("data/cleanneddata_exp3.csv")
            df = pd.read_csv("../../data/cleanneddata_exp3.csv")
        # optimal k: 11
        elif exp == 4:
            df = pd.read_csv("data/cleanneddata_exp4.csv")
            #df = pd.read_csv("../../data/cleanneddata_exp4.csv")

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

        # This should be if optimal... Just a placeholder
        if param_k == 10:
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
            print("Reached: predined k else, and k is: ", k)
            clf = neighbors.KNeighborsClassifier(k)

            cv_scores = cross_val_score(clf, X_train, y_train)
            clf.fit(X_train, y_train)

            # The accuracy
            accuracy = clf.score(X_test, y_test)
            print("Accuracy: ", accuracy)

            # Cross validate accuracy
            cv_scores_mean = np.mean(cv_scores)
            print("Cross validated accuracy: ", cv_scores_mean)

            y_pred = clf.predict(X_test)
            print("Y_PRED:: ", y_pred)

            # Prints confusion matrix with presentation view.
            plot_confusion_matrix(clf, X_test, y_test)
            plt.show()

            f1_micro = f1_score(y_test, y_pred, average='micro')
            print()
            return cv_scores_mean, f1_micro
            #return classification_report(y_test, y_pred, output_dict=True)


            # print("Program took", time.time() - start_time, "s to run")




def run_knn_with_find_k(X_train, X_test, y_train, y_test):
    x_axis = []
    y_axis = []

    for k in range(1, 20, 1):
        # values for k
        x_axis.append(k)

        # build classifiere
        clf = neighbors.KNeighborsClassifier(k)

        # adopt the dataen x and y trainingsets
        clf.fit(X_train, y_train)

        # to test the accuracy
        accuracy = clf.score(X_test, y_test)

        plot_confusion_matrix(clf, X_test, y_test)
        plt.show()

        # values for accuracy
        y_axis.append(accuracy)

    return x_axis, y_axis
