import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../../data/cleandata.csv')

df.drop(["artists"], axis=1, inplace=True)
df.drop(["duration_ms"], axis=1, inplace=True)
df.drop(["explicit"], axis=1, inplace=True)
df.drop(["id"], axis=1, inplace=True)
df.drop(["name"], axis=1, inplace=True)
df.drop(["popularity"], axis=1, inplace=True)
df.drop(["release_date"], axis=1, inplace=True)

print(df.head())
