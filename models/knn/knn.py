import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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
# X = df.drop(["tempo"], axis=1, inplace=True)
# X = df.drop(["loudness"], axis=1, inplace=True)
# X =df.drop(["liveness"], axis=1, inplace=True)
# 0.27838267317991877

# attributa or featrues withouth the "solution"/ class/ decade.
X = np.array(df.drop(["decade"], axis=1))
# energy_acousticness = df.drop(["liveness", "speechiness", "tempo", "loudness", "liveness", "decade",
#                               "instrumentalness", "danceability", "valence"], axis=1)
# print(energy_acousticness.describe())

# print(energy_acousticness.head())
# solution is stores to an array y
y = np.array(df["decade"])

# Devide the set in 20% for testing 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# build classifiere with k = 12
k = 12
clf = neighbors.KNeighborsClassifier(k)
# KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto',
# leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)[source]

# adopt the dataen x and y trainingsets
clf.fit(X_train, y_train)
# to test the accuracy
accuracy = clf.score(X_test, y_test)

predictions = clf.predict(X_test)
print(predictions)
# accuracy_score = clf

# cross_validate
cv_scores = cross_val_score(clf, X_train, y_train, cv=10)

cv_scores_mean = np.mean(cv_scores)
print(cv_scores , "\n\n""mean =" ,"{:.2f}".format(cv_scores_mean))


# print(df.head(10))
print(accuracy)

print("Program took", time.time() - start_time, "s to run")

#0.34 i accuracy!