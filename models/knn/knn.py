import seaborn as sns # for data visualization
import matplotlib.pyplot as plt # for data visualization
import numpy as np
import sklearn
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split, cross_validate
import pandas as pd
import time
start_time = time.time()

df = pd.read_csv("../../data/normalizeddata.csv")

#bytter ut alle spørsmåltegn med -99999 og setter inn datasettet med en gang
#Denne verdien er for å behandle dette som en outlier.
df.replace("?", -99999, inplace=True)

#Try without all attributes
#X =df.drop(["liveness"], axis=1, inplace=True)
#X =df.drop(["speechiness"], axis=1, inplace=True)

X =df.drop(["tempo"], axis=1, inplace=True)
X =df.drop(["loudness"], axis=1, inplace=True)
#X =df.drop(["liveness"], axis=1, inplace=True)
#0.27838267317991877


#attributa or featrues withouth the "solution"/ class/ decade.
X = np.array(df.drop(["decade"], axis=1))

#solution is stores to an array y
y = np.array(df["decade"])

#Devide the set in 20% for testing 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#k = np.log(170000).astype(int)

def run_knn():
    x_axis = []
    y_axis = []
    for k in range(10, 5000, 100):
        #values for k
        x_axis.append(k)

        # build classifiere
        clf = neighbors.KNeighborsClassifier(k)
        # adopt the dataen x and y trainingsets
        clf.fit(X_train, y_train)
        # to test the accuracy
        accuracy = clf.score(X_test, y_test)

        #values for accuracy
        y_axis.append(accuracy)

        # print(df.head())
        # print(accuracy)
        #
    print(x_axis, y_axis)
    return x_axis, y_axis

x_axis2, y_axis2 = run_knn()
# create dataframe using two list days and temperature
df_k = pd.DataFrame({"k-value": x_axis2, "accuracy": y_axis2})

# Draw line plot
sns.set_style("darkgrid")
sns.lineplot(x="k-value", y="accuracy", dashes=False, marker="o", data=df_k)
plt.show()  # to show graph

print("Program took", time.time() - start_time, "s to run")

"""
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
print(prediction) #returns 1990 | correct

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
print(prediction) #returns 1960 | wrong

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
print(prediction) #returns 2010 | correct

#===========================================================================================
#Check two at the same time
#example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
#prediction = clf.predict(example_measures)
#example_measures = example_measures.reshape(2, -1) #her må man ha to samples
#print(prediction)

"""

