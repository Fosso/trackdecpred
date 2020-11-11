import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
import sklearn
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd
import time

start_time = time.time()
df = pd.read_csv("../../data/normalizeddata_year.csv")

# bytter ut alle spørsmåltegn med -99999 og setter inn datasettet med en gang
# Denne verdien er for å behandle dette som en outlier.
df.replace("?", -99999, inplace=True)

# Try without all attributes
# X = df.drop(["liveness"], axis=1, inplace=True)
# X = df.drop(["speechiness"], axis=1, inplace=True)
# X = df.drop(["tempo"], axis=1, inplace=True)
# X = df.drop(["loudness"], axis=1, inplace=True)
# X = df.drop(["liveness"], axis=1, inplace=True)
# 0.27838267317991877

# attributa or featrues withouth the "solution"/ class/ decade.
X = np.array(df.drop(["year"], axis=1))

# solution is stores to an array y
y = np.array(df["year"])

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
#Checking performance on the training set
"""print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
     """
#Checking performance on the test set
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
# Calculate new accuracy percentage

prediction = clf.predict(X_test)

correct = 0
for i in range(len(X_test)):
    if y_test[i] + 20 >= prediction[i] >= y_test[i] - 20:
        correct += 1

#Checking performance on the test set
print('Accuracy of K-NN classifier on test set with a range of +-20 years: {:.2f}'
     .format(correct / float(len(X_test)) * 100.0))

print("Program took", time.time() - start_time, "s to run")

