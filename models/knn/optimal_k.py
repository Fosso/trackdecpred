import time

import matplotlib.pyplot as plt  # for data visualization
import numpy as np
import pandas as pd
import seaborn as sns  # for data visualization
from sklearn import neighbors
from sklearn.model_selection import train_test_split

start_time = time.time()


def find_k(filepath):
    x_axis = []
    y_axis = []
    filepath = "../../data/normalizeddata.csv"
    df = pd.read_csv("../../data/normalizeddata.csv")

    y = np.array(df["decade"])

    X = np.array(df.drop(["decade"], axis=1))

    # Divide the set in 20% for testing 80% for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for k in range(25, 35, 2):
        # values for k
        x_axis.append(k)

        # build classifiere
        clf = neighbors.KNeighborsClassifier(k)

        # adopt the dataen x and y trainingsets
        clf.fit(X_train, y_train)
        # to test the accuracy
        accuracy = clf.score(X_test, y_test)

        # values for accuracy
        y_axis.append(accuracy)

    print(x_axis, y_axis)
    return x_axis, y_axis


x_axis2, y_axis2 = find_k()
print("Program took", time.time() - start_time, "s to run")
# create dataframe using k-value and accuracy
df_k = pd.DataFrame({"k-value": x_axis2, "accuracy": y_axis2})

# Draw line plot
sns.set_style("darkgrid")
sns.lineplot(x="k-value", y="accuracy", dashes=False, marker="o", data=df_k)

# to show graph
plt.show()

if __name__ == '__main__':
    find_k()