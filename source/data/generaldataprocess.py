import pandas as pd
import numpy as np
from sklearn import preprocessing


def readfile(file):
    df = pd.read_csv(file)
    return df


def proc_and_norm_gen(df):
    # remove irrelevant columns: arrtist, duration, ecplicit, id, name, popularity, release_date, mode, key
    df.drop(["artists"], axis=1, inplace=True)
    df.drop(["duration_ms"], axis=1, inplace=True)
    df.drop(["explicit"], axis=1, inplace=True)
    df.drop(["id"], axis=1, inplace=True)
    df.drop(["name"], axis=1, inplace=True)
    df.drop(["popularity"], axis=1, inplace=True)
    df.drop(["release_date"], axis=1, inplace=True)
    df.drop(["mode"], axis=1, inplace=True)
    df.drop(["key"], axis=1, inplace=True)

    # reduce decimals for remaining columns
    df["energy"] = df["energy"].round(4)
    df["acousticness"] = df["acousticness"].round(4)
    df["danceability"] = df["danceability"].round(4)
    df["instrumentalness"] = df["instrumentalness"].round(4)
    df["liveness"] = df["liveness"].round(4)
    df["loudness"] = df["loudness"].round(4)
    df["speechiness"] = df["speechiness"].round(4)
    df["tempo"] = df["tempo"].round(4)
    df["valence"] = df["valence"].round(4)

    # Normlization of colomn tempo
    df_tempo = pd.DataFrame(df["tempo"])
    A = df_tempo.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(A)
    df_tempo = pd.DataFrame(x_scaled).round(4)
    df["tempo"] = df_tempo

    # Normlization of colomn loudness
    df_loudness = pd.DataFrame(df["loudness"])
    A = df_loudness.values  # returnerer et array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(A)
    df_loudness = pd.DataFrame(x_scaled).round(4)
    df["loudness"] = df_loudness

    return df


def proc_ex1(df):
    df_ex1 = df
    # converting from year to decade
    df_ex1["year"] = df_ex1["year"] / 10
    df_ex1["year"] = df_ex1["year"].apply(np.floor) * 10
    df_ex1["year"] = df_ex1["year"].astype(int)

    # change name from year to decade
    df_ex1.rename(columns={'year': 'decade'}, inplace=True)

    return df_ex1


def proc_ex2(df):
    df_ex2 = df

    df_ex2 = df_ex2[df_ex2.decade != 1930]
    df_ex2 = df_ex2[df_ex2.decade != 1940]
    df_ex2 = df_ex2[df_ex2.decade != 1950]
    df_ex2 = df_ex2[df_ex2.decade != 1960]
    df_ex2 = df_ex2[df_ex2.decade != 1970]
    df_ex2 = df_ex2[df_ex2.decade != 1980]
    df_ex2 = df_ex2[df_ex2.decade != 1990]
    df_ex2 = df_ex2[df_ex2.decade != 2000]
    df_ex2 = df_ex2[df_ex2.decade != 2010]

    return df_ex2


def createfile(dataframe, filename):
    # Creating cleaned CSV-data files
    dataframe.to_csv(r"../../data/file{0}.csv".format(filename), index=False)


if __name__ == '__main__':
    df = readfile('../../data/procdata.csv')
    df = proc_and_norm_gen(df)
    df_ex1 = proc_ex1(df)
    df_ex2 = proc_ex1(df)

    #creates clean and normalize data
    createfile(df_ex1,"datanormalization_ex1.csv")
    createfile(df_ex2, "datanormalization_ex2.csv")




