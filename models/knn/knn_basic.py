import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import time

start_time = time.time()
df = pd.read_csv("../../data/normalizeddata.csv")

# bytter ut alle spørsmåltegn med -99999 og setter inn datasettet med en gang
# Denne verdien er for å behandle dette som en outlier.
df.replace("?", -99999, inplace=True)


y = np.array(df["decade"])

X = np.array(df.drop(["decade"], axis=1))
#"valence", "speechiness", "tempo", "loudness", "liveness", "danceability", "instrumentalness",

# Divide the set in 20% for testing 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build classifier with k = 12
#k for vanlig datasett 1930-1970 = 56
k = 25
clf = neighbors.KNeighborsClassifier(k)
# KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto',
# leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)[source]

cv_scores = cross_val_score(clf, X_train, y_train)
clf.fit(X_train, y_train)

# to test the accuracy
cv_scores_mean = np.mean(cv_scores)
print(cv_scores , "\n""mean =" ,"{:.2f}".format(cv_scores_mean))

# to test the accuracy
accuracy = clf.score(X_test, y_test)

print("dette er treffsikkerheten:", accuracy)

print("Program took", time.time() - start_time, "s to run")


def run_knn():
    x_axis = []
    y_axis = []
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



#0.34 i accuracy!
