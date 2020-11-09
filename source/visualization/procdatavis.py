import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('../../data/cleandata.csv')


def descriptive():
    print(df.describe(include='all'))



def plot_histogram():
    df.hist(figsize=(20, 20))
    plt.show()


def lineplot():
    plt.figure(figsize=(16, 10))
    sns.set(style="darkgrid")
    # Add tempo, loudness, key,  once normalized
    columns = ["acousticness", "danceability", "energy", "instrumentalness", "liveness",
               "mode", "speechiness", "valence"]
    for col in columns:
        x = df.groupby("decade")[col].mean()
        ax = sns.lineplot(x=x.index, y=x, label=col)
    ax.set_title('Audio features over decades')
    ax.set_ylabel('Measure')
    ax.set_xlabel('Decade')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.show()


def radarplot():
    radar = df.loc[:, "acousticness": "valence"]
    labels = list(radar.columns)
    values = radar.mean().values
    df_radar = pd.DataFrame(dict(r=values, theta=labels))
    fig = px.line_polar(df_radar, r="r", theta="theta", line_close=True)
    fig.update_traces(fill="toself")
    fig.show()
    return df_radar


if __name__ == '__main__':
    descriptive()
    plot_histogram()
    # radarplot()
    lineplot()
