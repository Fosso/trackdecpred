import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import ticker
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv('../../data/cleanneddata_exp2.csv')


# df = pd.read_csv('../../data/cleanneddata_exp5.csv')


def descriptive():
    # remove hashtags to see count for given decade
    # dec_count = df.groupby("decade")
    # print(dec_count.count())
    print(df.describe(include='all'))


def plot_histogram():
    df.drop(["decade"], axis=1, inplace=True)
    plt.xlabel('Measure value', fontsize=20)
    plt.ylabel('Number of tracks', fontsize=20)
    plt.title("Relationship for audio features between values and tracks")
    df.hist(figsize=(20, 20))
    # plt.show()

    plt.savefig("../../results/exploration/histogram_plot_decades.png")


def lineplot():
    plt.figure(figsize=(16, 10))
    sns.set(style="darkgrid")
    columns = ["acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness",
               "speechiness", "tempo", "valence"]
    for col in columns:
        x = df.groupby("decade")[col].mean()
        ax = sns.lineplot(x=x.index, y=x, label=col)
    ax.set_title('Audio features measure values over time')
    ax.set_ylabel('Measure value')
    ax.set_xlabel('Decade')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.savefig("../../results/exploration/line_plot_decades.png")


def scatter_plot():
    # df = pd.read_csv('../../data/cleanneddata_exp3.csv')
    df = pd.read_csv('../../data/cleanneddata_exp5.csv')

    df.drop(df.columns.difference(["acousticness", "energy", "decade"]), 1, inplace=False)
    markers = {"1980": "X"}
    sns.scatterplot(data=df, x="acousticness", y="energy", hue="decade", palette="deep", markers=markers)
    # sns.lmplot('acousticness', 'energy', df, hue='decade', fit_reg=False)
    fig: object = plt.gcf()
    fig.set_size_inches(15, 10)
    # plt.savefig("../../results/exploration/scatter_plot_exp5_acousticness_energy.png")
    plt.show()


if __name__ == '__main__':
    descriptive()
    plot_histogram()
    lineplot()
    scatter_plot()
