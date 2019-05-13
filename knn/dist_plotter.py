import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


OUTPUT_PATH = '../outcomes/dists'


def pima_diab():
    # load
    header = [str(i) for i in range(1, 10)]
    diabetes = pd.read_csv("../pima_diabetes/diabetes.data", names=header)

    # plot
    plt.clf()
    g = sns.pairplot(diabetes, hue="9", diag_kind='hist', vars=header[:-1])
    fig = g.fig
    path = os.path.join(OUTPUT_PATH, "diabetes.pdf")
    fig.savefig(path, bbox_inches='tight')

    # corr
    diabetes = diabetes[header[:-1]]
    correlations = diabetes.corr()
    path = os.path.join(OUTPUT_PATH, "diabetes_corr.csv")
    correlations.to_csv(path)


def glass():
    # load
    col_names = [str(i) for i in range(1, 11)]
    glass = pd.read_csv('../glass_identification/glass.data', header=None,
                        usecols=range(1,11), names=col_names)

    # plot
    plt.clf()
    g = sns.pairplot(glass, diag_kind='hist', vars=col_names[:-1])
    fig = g.fig
    path = os.path.join(OUTPUT_PATH, "glass.pdf")
    fig.savefig(path, bbox_inches='tight')

    # corr
    glass = glass[col_names[:-1]]
    correlations = glass.corr()
    path = os.path.join(OUTPUT_PATH, "glass_corr.csv")
    correlations.to_csv(path)


def wine():
    # load
    col_names = [str(i) for i in range(1, 15)]
    wine = pd.read_csv('../wine/wine.data', header=None, names=col_names)

    # plot
    plt.clf()
    g = sns.pairplot(wine, diag_kind='hist', vars=col_names[1:])
    fig = g.fig
    path = os.path.join(OUTPUT_PATH, "wine.pdf")
    fig.savefig(path, bbox_inches='tight')

    # corr
    wine = wine[col_names[1:]]
    correlations = wine.corr()
    path = os.path.join(OUTPUT_PATH, "wine_corr.csv")
    correlations.to_csv(path)


if __name__ == "__main__":
    pima_diab()
    # glass()
    # wine()
