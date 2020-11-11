import pandas as pd
import numpy as np
from sklearn import preprocessing


# read csv file with pandas
def readfile(file):
    df = pd.read_csv(file)
    return df


# Normalizing and cleaning data
def proc_and_norm_gen(df):
    # remove irrelevant columns: artist, duration, explicit, id, name, popularity, release_date, mode, key
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

    #sort by year
    df.sort_values(by=["year"])

    return df

# Processing data for experiment two (year)
def proc_exp1(df):
    df_exp1 = df
    # not changing anything

    return df_exp1


# Function for proc exp_2 and exp_3, 4 and 5
def year_to_decade(df):
    df_decade = df
    # converting from year to decade
    df_decade["year"] = df_decade["year"] / 10
    df_decade["year"] = df_decade["year"].apply(np.floor) * 10
    df_decade["year"] = df_decade["year"].astype(int)

    # change name from year to decade and sort
    df_decade.rename(columns={"year": "decade"}, inplace=True)
    df_decade.sort_values(by=["decade"])

    return df_decade


# Processing data for expperiment two (decade)
def proc_exp2(df):
    df_exp2 = year_to_decade(df)
    return df_exp2


# Processing data for experiment three (1930 - 1980, decade)
def proc_exp3(df):
    df_exp3 = year_to_decade(df)
    # Removing uninteresting rows
    df_exp3 = df_exp3[df_exp3.decade != 1920]

    df_exp3 = df_exp3[df_exp3.decade != 1990]
    df_exp3 = df_exp3[df_exp3.decade != 2000]
    df_exp3 = df_exp3[df_exp3.decade != 2010]
    df_exp3 = df_exp3[df_exp3.decade != 2020]

    return df_exp3


# Processing data for experiment four (1920 vs. 2020, decade)
def proc_exp4(df):
    df_exp4 = year_to_decade(df)
    # Removing uninteresting rows
    #remove_list = [1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    df_exp4 = df_exp4[df_exp4.decade != 1930]
    df_exp4 = df_exp4[df_exp4.decade != 1940]
    df_exp4 = df_exp4[df_exp4.decade != 1950]
    df_exp4 = df_exp4[df_exp4.decade != 1960]
    df_exp4 = df_exp4[df_exp4.decade != 1970]
    df_exp4 = df_exp4[df_exp4.decade != 1980]
    df_exp4 = df_exp4[df_exp4.decade != 1990]
    df_exp4 = df_exp4[df_exp4.decade != 2000]
    df_exp4 = df_exp4[df_exp4.decade != 2010]

    return df_exp4

# Processing data for experiment four (1930 vs. 1980, decade)
def proc_exp5(df_5):
    df_exp5 = year_to_decade(df_5)
    # Removing uninteresting rows
    df_exp5 = df_exp5[df_exp5.decade != 1920]
    df_exp5 = df_exp5[df_exp5.decade != 1940]
    df_exp5 = df_exp5[df_exp5.decade != 1950]
    df_exp5 = df_exp5[df_exp5.decade != 1960]
    df_exp5 = df_exp5[df_exp5.decade != 1970]
    df_exp5 = df_exp5[df_exp5.decade != 1990]
    df_exp5 = df_exp5[df_exp5.decade != 2000]
    df_exp5 = df_exp5[df_exp5.decade != 2010]
    df_exp5 = df_exp5[df_exp5.decade != 2020]

    return df_exp5


# Creating cleaned CSV-data files
def createfile(dataframe, filename):
    dataframe.to_csv(r"../../data/file{0}.csv".format(filename), index=False)


# on run, this gets expectued
if __name__ == '__main__':
    # Creates csv file for correct problems

    df_read = readfile('../../data/procdata.csv')
    df_proc = proc_and_norm_gen(df_read)
    df_exp1_cleaned = proc_exp1(df_proc)
    createfile(df_exp1_cleaned, "cleaneddata_exp1.csv")

    df_read_2 = readfile('../../data/procdata.csv')
    df_proc_2 = proc_and_norm_gen(df_read_2)
    df_exp2_cleaned = proc_exp2(df_proc_2)
    createfile(df_exp2_cleaned, "cleaneddata_exp2.csv")

    df_read_3 = readfile('../../data/procdata.csv')
    df_proc_3 = proc_and_norm_gen(df_read_3)
    df_exp3_cleaned = proc_exp3(df_proc_3)
    createfile(df_exp3_cleaned, "cleaneddata_exp3.csv")

    df_read_4 = readfile('../../data/procdata.csv')
    df_proc_4 = proc_and_norm_gen(df_read_4)
    df_exp4_cleaned = proc_exp4(df_proc_4)
    createfile(df_exp4_cleaned, "cleaneddata_exp4.csv")

    df_read_5 = readfile('../../data/procdata.csv')
    df_proc_5 = proc_and_norm_gen(df_read_5)
    df_exp5_cleaned = proc_exp5(df_proc_5)
    createfile(df_exp5_cleaned, "cleaneddata_exp5.csv")


