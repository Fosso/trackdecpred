from collections import Counter
import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd
import time



start_time = time.time()


df = pd.read_csv("../../data/normalizeddata.csv")

# bytter ut alle spørsmåltegn med -99999 og setter inn datasettet med en gang
# Denne verdien er for å behandle dette som en outlier.
df.replace("?", -99999, inplace=True)

# Try without all attributes
# X =df.drop(["liveness"], axis=1, inplace=True)
# X =df.drop(["speechiness"], axis=1, inplace=True)

X = df.drop(["tempo"], axis=1, inplace=True)
X = df.drop(["loudness"], axis=1, inplace=True)
# X =df.drop(["liveness"], axis=1, inplace=True)
# 0.27838267317991877


# attributa or featrues withouth the "solution"/ class/ decade.
X = np.array(df.drop(["decade"], axis=1))

# solution is stores to an array y
y = np.array(df["decade"])

# Devide the set in 20% for testing 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# k = np.log(170000).astype(int)
"""
predict = [X_test, y_test]
data = [X_train, y_train]
"""


def run_knn():
    x_axis = []
    y_axis = []
    for k in range(45, 50, 2):
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

        # print(df.head())
        # print(accuracy)

    print(x_axis, y_axis)
    return x_axis, y_axis


x_axis2, y_axis2 = run_knn()
print("Program took", time.time() - start_time, "s to run")
# create dataframe using two list days and temperature
df_k = pd.DataFrame({"k-value": x_axis2, "accuracy": y_axis2})

# Draw line plot
sns.set_style("darkgrid")
sns.lineplot(x="k-value", y="accuracy", dashes=False, marker="o", data=df_k)
plt.show()  # to show graph

