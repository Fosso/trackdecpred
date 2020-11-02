import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Leser datasettet fra Kaggle
df = pd.read_csv('../../data/procdata.csv')

#Fjernet kolonner som er irrelevante
df.drop(["artists"], axis=1, inplace=True)
df.drop(["duration_ms"], axis=1, inplace=True)
df.drop(["explicit"], axis=1, inplace=True)
df.drop(["id"], axis=1, inplace=True)
df.drop(["name"], axis=1, inplace=True)
df.drop(["popularity"], axis=1, inplace=True)
df.drop(["release_date"], axis=1, inplace=True)

#endrer årstall til tiår-gruppe
df["year"] = df["year"]/10
df["year"] = df["year"].apply(np.floor)*10
df["year"] = df["year"].astype(int)

#endrer navn fra year til decade
df.rename(columns={'year': 'decade'}, inplace=True)

#Oppretter ny fil etter fjernet kolonner.
df.to_csv(r"../../data/cleandata.csv", index=False)

#Leser renset dataset for å kunne printe det.
#clean_df = pd.read_csv('../../data/cleandata.csv')
#print(clean_df.head())