import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
import plotly.graph_objects as go


df = pd.read_csv('../../data/normalizeddata.csv')


def descriptive():
    print(df.describe(include='all'))
    df.head(10)


def plot_histogram():
    df.hist(figsize=(20, 20))
    plt.show()


def lineplot():
    plt.figure(figsize=(16, 10))
    sns.set(style="darkgrid")
    columns = ["acousticness", "danceability", "energy", "instrumentalness", "liveness",
               "loudness", "speechiness", "tempo", "valence"]
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


if __name__ == '__main__':
    descriptive()
    radarplot()
    plot_histogram()
    lineplot()