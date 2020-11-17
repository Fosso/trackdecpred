import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('../../data/cleanneddata_exp2.csv')


def descriptive():
    # remove hashtags to see count for given decade
    # dec_count = df.groupby("decade")
    # print(dec_count.count())
    print(df.describe(include='all'))


def plot_histogram():
    df.hist(figsize=(20, 20))
    plt.show()


def lineplot():
    plt.figure(figsize=(16, 10))
    sns.set(style="darkgrid")
    columns = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness",
               "speechiness", "tempo", "valence"]
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
    fig = go.Figure()
    radar = df.loc[:, "acousticness": "valence"]
    dec = df.groupby('decade')
    keys = dec.groups.keys()

    for key in keys:
        fig.add_trace(go.Scatterpolar(
            r=radar.loc[key].values,
            theta=radar.columns,
            fill='toself',
            opacity=0.99,
            name=key
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )),
        showlegend=True
    )

    fig.show()


def radarplot2():
    fig = go.Figure()
    radar = df.loc[:, "acousticness": "valence"]
    columns = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness",
               "speechiness", "tempo", "valence"]
    dec = df.groupby('decade')
    keys = dec.groups.keys()
    print("Sånn ser df ut: ", df.head(5))
    test1920 = df.loc["decade", 1920]

    fig.add_trace(go.Scatterpolar(
        # r=[0.9, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.4],
        r=test1920,
        theta=columns,
        fill='toself',
        opacity=0.99,
        name="1920"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )),
        showlegend=True
    )

    fig.show()


if __name__ == '__main__':
    descriptive()
    plot_histogram()
    # radarplot()
    # radarplot2()
    lineplot()
