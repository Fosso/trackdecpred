import pandas as pd
from sklearn import preprocessing

#Leser datasettet fra Kaggle
df = pd.read_csv('../../data/cleandata.csv')


#Normlization of colomn tempo
df_tempo = pd.DataFrame(df["tempo"])

A = df_tempo.values #returnerer et array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(A)

df_tempo = pd.DataFrame(x_scaled).round(4)
df["tempo"] = df_tempo

#Normlization of colomn loudness
df_loudness = pd.DataFrame(df["loudness"])

A = df_loudness.values #returnerer et array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(A)

df_loudness = pd.DataFrame(x_scaled).round(4)
df["loudness"] = df_loudness

#Reduce number of decimals
df["energy"] = df["energy"].round(4)
df["acousticness"] = df["acousticness"].round(4)
df["danceability"] = df["danceability"].round(4)
df["instrumentalness"] = df["instrumentalness"].round(4)
#df["key"] = df["key"].round(4)
df["liveness"] = df["liveness"].round(4)
df["loudness"] = df["loudness"].round(4)
df["speechiness"] = df["speechiness"].round(4)
df["tempo"] = df["tempo"].round(4)
df["valence"] = df["valence"].round(4)

#create new datasets
df.to_csv(r"../../data/normalizeddata.csv", index=False)

