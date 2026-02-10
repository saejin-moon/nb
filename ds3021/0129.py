import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    return KNeighborsClassifier, pd, plt, sns


@app.function
def minmax(x):
    u = (x-min(x))/(max(x)-min(x))
    return u


@app.cell
def _(pd):
    diabetes_df = pd.read_csv("./data/diabetes-dataset.csv")
    diabetes_df.head()
    return (diabetes_df,)


@app.cell
def _(diabetes_df, plt, sns):
    plot = sns.scatterplot(x=diabetes_df["Glucose"], y=diabetes_df["DiabetesPedigreeFunction"], hue=diabetes_df["Outcome"], style=diabetes_df["Outcome"])
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
    plt.show()
    return


@app.cell
def _(KNeighborsClassifier, diabetes_df):
    y = diabetes_df['Outcome']
    ctrl_list = ["Glucose", "DiabetesPedigreeFunction"]
    x = diabetes_df.loc[:, ctrl_list]
    u = x.apply(minmax)
    k = 2
    model = KNeighborsClassifier(n_neighbors = k)
    model = model.fit(u,y)
    y_hat = model.predict(u)
    return model, u, y, y_hat


@app.cell
def _(diabetes_df, plt, sns, y_hat):
    pred_plot = sns.scatterplot(x=diabetes_df["Glucose"], y=diabetes_df["DiabetesPedigreeFunction"], hue=diabetes_df["Outcome"], style=y_hat)
    sns.move_legend(pred_plot, "upper left", bbox_to_anchor=(1, 1))
    plt.show()
    return


@app.cell
def _(pd, y, y_hat):
    pd.crosstab(y,y_hat)
    return


@app.cell
def _(model, u, y):
    model.score(u,y)
    return


if __name__ == "__main__":
    app.run()
