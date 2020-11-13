import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


df = pd.read_csv("../../data/cleanneddata_exp3.csv")

y = np.array(df["decade"])
X = np.array(df.drop(["decade"], axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#svclassifier = SVC(kernel='poly', degree=1)

#svclassifier = SVC(kernel='rbf')
svclassifier = SVC(kernel='linear')
#svclassifier = SVC(kernel='sigmoid')

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
