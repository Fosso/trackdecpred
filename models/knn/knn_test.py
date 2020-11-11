import seaborn as sns  # for data visualization
import matplotlib.pyplot as plt  # for data visualization
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_val_score

import pandas as pd
import time


start_time = time.time()
df = pd.read_csv("../../data/normalizeddata.csv")

#X = df.drop(["tempo"], axis=1, inplace=True)
#X = df.drop(["loudness"], axis=1, inplace=True)


X = np.array(df.drop(["decade", "tempo", "loudness"], axis=1))

y = np.array(df["decade"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k = 12
clf = neighbors.KNeighborsClassifier(k)

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)


#===========================================================================================
#Not in dataset in should be testet
around_the_world_daft_punk_1997 = np.array([[0.0036, 0.956, 0.795, 0.889, 0.0906, 0.15, 0.841]])

#YEAR: 1997
#DECADE: 1990
#loudness=-5.311, tatt bort pga ikke normalisert
#tempo=121.294, tatt bort pga ikke normalisert
#Full= ([[0.0036,0.956,0.795,0.889,0.0906,-5.311,0.15,121.294,0.841]])
#SPOTIFY URI LINK: potify:track:1pKYYY0dkg23sQQXi0Q5zN
#LINK: https://songdata.io/track/1pKYYY0dkg23sQQXi0Q5zN/Around-the-World-by-Daft-Punk
#LINK FOR FEATURES: https://www.slideshare.net/MarkKoh9/audio-analysis-with-spotifys-web-api

#Estimate the example_measure
prediction = clf.predict(around_the_world_daft_punk_1997)
around_the_world_daft_punk_1997 = around_the_world_daft_punk_1997.reshape(1, -1)
print("The correct year is: 1997 and predicted decade is:", prediction)

#===========================================================================================
kind_hearted_women_blues_1936 = np.array([[0.928, 0.577, 0.1510, 0.0018, 0.16, 0.0453, 0.258]])

#Year: 1936
#Decade: 1930
#loudness: -17.41
#tempo: 87.919
#full: ([[0.928, 0.577, 0.1510, 0.0018, 0.16, -17.41, 0.0453, 87.919, 0.258 ]])
#LINK FOR FEATURES: https://medium.com/@samlupton/spotipy-get-features-from-your-favourite-songs-in-python-6d71f0172df0

prediction = clf.predict(kind_hearted_women_blues_1936)
kind_hearted_women_blues_1936 = kind_hearted_women_blues_1936.reshape(1, -1)
print("The correct year is: 1936 and predicted decade is:", prediction)

#===========================================================================================
the_struggle_2019 = np.array([[0.364, 0.639, 0.4, 0, 0.108, 0.404, 0.51]])

#Year: 2019
#Decade: 2010
#loudness: -11.732
#tempo: 91.427
#valence was found on LINK rest on LINK FOR FEATURES
#full: ([[0.364, 0.639, 0.4, 0, 0.108, -17.41, 0.404, 87.919, ?]])
#LINK: https://songdata.io/track/4jwYSwakRTFvukWnBCmZDk/The-Struggle-Freestyle-by-Big-Zuu
#LINK FOR FEATURES:https://www.theinformationlab.co.uk/2019/08/08/getting-audio-features-from-the-spotify-api/

prediction = clf.predict(the_struggle_2019)
the_struggle_2019 = the_struggle_2019.reshape(1, -1)
print("The correct year is: 2019 and predicted decade is:", prediction)

#===========================================================================================
#Check two at the same time
#example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
#prediction = clf.predict(example_measures)
#example_measures = example_measures.reshape(2, -1) #her må man ha to samples
#print(prediction)

print("Program took", time.time() - start_time, "s to run")


"""
def kNN(data, predict, k):
    if len(data) >= k:
        warnings.warn("K is less then the total attributes")
    d = []
    for group in data:
        for features in group:
            euclid_d = np.linalg.norm(np.array(features) - np.array(predict))
            d.append([euclid_d, group]
    votes = []
    for i in sorted(d[:k]):
        votes = [i[1]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result
result = kNN([X_train, y_train], [X_test, y_test], 93)

print(result)
"""