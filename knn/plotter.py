import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

OUTCOME_PATH = '../outcomes/'
metrics_names = ['acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']
col_names_to_groupby = ["dataset","n-k","voting","distance"]
col_names_to_groupby_folds = ["dataset","n-folds"]


def plot_metrics_for_all_datasets_average_folds(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    for ds_name in datasets:
        ds_data = df[df['dataset'] == ds_name]
        _plot_metrics(ds_data, ds_name)


def _plot_metrics(df, ds_name):
    df = df.groupby(col_names_to_groupby, as_index=False).mean()
    for metric_name in metrics_names:
        _plot_metric_k_voting_dist(df, ds_name, metric_name)


def _plot_metric_k_voting_dist(df, ds_name, metric):
    fig, ax = plt.subplots()
    ax = sns.lineplot(x="n-k", y=metric, hue = "voting", style = "distance", data=df, markers=True, dashes=True)
    for i in range(len(ax.lines)):
        if i % 2 == 0:
            ax.lines[i].set_linestyle("-.")
        else:
            ax.lines[i].set_linestyle("--")
    ax.set_title("Miara {} w zaleznosci od k dla zbioru {}.".format(metric, ds_name))
    path = os.path.join(OUTCOME_PATH, "plots")
    path_pdf = os.path.join(path, "{}-{}.pdf".format(ds_name, metric))
    path_png = os.path.join(path, "{}-{}.png".format(ds_name, metric))
    fig.savefig(path_pdf, bbox_inches='tight')
    fig.savefig(path_png, bbox_inches='tight')


def plot_metrics_for_all_datasets_by_folds(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    for ds_name in datasets:
        ds_data = df[df['dataset'] == ds_name]
        _plot_metrics_by_folds(ds_data, ds_name)


def _plot_metrics_by_folds(df, ds_name):
    df = df.groupby(col_names_to_groupby_folds, as_index=False).mean()
    df = df[['n-folds', 'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']]
    df = df.melt('n-folds', var_name='metric', value_name='value')

    fig, ax = plt.subplots()
    ax = sns.lineplot(x="n-folds", y="value", hue="metric", style="metric", data=df, markers=True)
    for i in range(len(ax.lines)):
        ax.lines[i].set_linestyle("--")
    ax.set_title("Jakosc klasyfikacji w zaleznosci od liczby foldow dla zbioru {}.".format(ds_name))
    path = os.path.join(OUTCOME_PATH, "plots")
    path_pdf = os.path.join(path, "folds-{}.pdf".format(ds_name))
    path_png = os.path.join(path, "folds-{}.png".format(ds_name))
    fig.savefig(path_pdf, bbox_inches='tight')
    fig.savefig(path_png, bbox_inches='tight')


def parse_distance_to_names(df):
    for i in range(len(df)):
        if df.loc[:, 'distance'].iloc[i] == 1:
            df.loc[:, 'distance'].iloc[i] = 'manhattan'
        elif df.loc[:, 'distance'].iloc[i] == 2:
            df.loc[:, 'distance'].iloc[i] = 'euclidean'


if __name__ == '__main__':
    file_path = "../outcomes/2019-05-13-17-46/all.csv"
    df = pd.read_csv(file_path)
    # plot_metrics_for_all_datasets_average_folds(df)
    plot_metrics_for_all_datasets_by_folds(df)
