#remove arrtist, duration, ecplicit, id, name, popularity, release_date, mode, key
#convert from year to decade
#convert to 4 decimals

import pandas as pd
import numpy as np #reads csv (Kaggle)

df = pd.read_csv('../../data/procdata.csv')

#remove irrelevant columns

df.drop(["artists"], axis=1, inplace=True)
df.drop(["duration_ms"], axis=1, inplace=True)
df.drop(["explicit"], axis=1, inplace=True)
df.drop(["id"], axis=1, inplace=True)
df.drop(["name"], axis=1, inplace=True)
df.drop(["popularity"], axis=1, inplace=True)
df.drop(["release_date"], axis=1, inplace=True)
df.drop(["mode"], axis=1, inplace=True)

#this might be added later on
df.drop(["key"], axis=1, inplace=True)
#change data from year to decade

df["year"] = df["year"]/10
df["year"] = df["year"].apply(np.floor)*10
df["year"] = df["year"].astype(int)
#change name from year to decade

df.rename(columns={'year': 'decade'}, inplace=True)

df = df[df.decade != 1930]
df = df[df.decade != 1940]
df = df[df.decade != 1950]
df = df[df.decade != 1960]
df = df[df.decade != 1970]
df = df[df.decade != 1980]
df = df[df.decade != 1990]
df = df[df.decade != 2000]
df = df[df.decade != 2010]

# df_dec = df.groupby(['decade'])

# print(df_dec.size())#reduce decimals
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
#create new datasets with removed unessasary columns etc.

df.to_csv(r"../../data/cleandata.csv", index=False)#Leser renset dataset for å kunne printe det.
#clean_df = pd.read_csv('../../data/cleandata.csv')
#print(clean_df["energy"].head(40))