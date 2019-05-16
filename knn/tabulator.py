import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from plotter import col_names_to_groupby, OUTCOME_PATH, col_names_to_groupby_folds


def make_metrics_for_all_datasets_average_folds(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    for ds_name in datasets:
        ds_data = df[df['dataset'] == ds_name]
        _make_metrics_average_folds_one_dataset(ds_data, ds_name)


def _make_metrics_average_folds_one_dataset(df, ds_name):
    df = df.groupby(col_names_to_groupby, as_index=False).mean()
    col_names = ["n-k", "voting", "distance", 'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']
    df = df[col_names]
    df = df.round({"acc_mean": 3, "prec_mean": 3, "rec_mean": 3, "f1_mean": 3})
    path = "table-{}-metrics.csv".format(ds_name)
    path = os.path.join(OUTCOME_PATH, path)
    df.to_csv(path, index=False)

def make_metrics_for_all_datasets_over_folds(df: pd.DataFrame):
    datasets = df["dataset"].unique()
    for ds_name in datasets:
        ds_data = df[df['dataset'] == ds_name]
        _make_metrics_over_folds_one_dataset(ds_data, ds_name)


def _make_metrics_over_folds_one_dataset(df, ds_name):
    df = df.groupby(col_names_to_groupby_folds, as_index=False).mean()
    col_names = ["n-folds", 'acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']
    df = df[col_names]
    df = df.round({"acc_mean": 3, "prec_mean": 3, "rec_mean": 3, "f1_mean": 3})
    path = "table-{}-folds.csv".format(ds_name)
    path = os.path.join(OUTCOME_PATH, path)
    df.to_csv(path, index=False)


def get_max_f1_for_every_k(df):
    datasets = df["dataset"].unique()
    ks = df["n-k"].unique()
    df = df.groupby(col_names_to_groupby, as_index=False).mean()
    for ds_name in datasets:
        df_ds = df[df["dataset"]==ds_name]
        iterator = 0
        multiplier = 6  # 6 rows for every k

        print("For {}".format(ds_name))
        for k in ks:
            df_k=df_ds[df_ds["n-k"]==k]
            df_k=df_k[["n-k", "f1_mean"]]
            index = df_k.idxmax(0)["f1_mean"]
            #print("{}".format(str(index+iterator*multiplier)))
            print()
            row = df.iloc[index,:]
            print(row)
            iterator += 1


if __name__ == '__main__':
    file_path = "../outcomes/2019-05-14-16-00/all.csv"
    df = pd.read_csv(file_path)
    make_metrics_for_all_datasets_average_folds(df)
    make_metrics_for_all_datasets_over_folds(df)
    get_max_f1_for_every_k(df)
