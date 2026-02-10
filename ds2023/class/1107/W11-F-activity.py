import marimo

__generated_with = "0.19.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W11 F Activity: Data Ink and Axis Manipulation

    DS 2023 | Communicating with Data

    In this notebook, we create plots that demonstrate the principles of data ink and axis manipulation.

    The main idea is to learn how to play with these design dimensions using Matplotlib and Seaborn.

    We will work with a data set of baseball statistics from 1871 to 2012.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Set Up
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import Libraries
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use("dark_background")
    return pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read in the Data
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("mlbpitching.csv")
    df = df.set_index('year')
    df = df.drop(columns='id')
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df["year"] = df.index
    df.columns
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Challenges
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Challenge 1: Axis Manipulation

    Create a line plot of `total_pitchers` using Seaborn's `relplot`.

    Set the aspect to $1.5$.

    You will notice a dip between the years 1940 and 1960.
    """)
    return


@app.cell
def _(df, sns):
    sns.relplot(data=df, x="year", y="total_pitchers", aspect=1.5, kind="line")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try to get a better sense visually of where the dip begins and ends by adding vertical lines to the plot.

    You can do this by calling `g.ax.grid(axis='x')`.

    Note that you can make these lines lighter by setting the alpha to something like $.3$.

    The variable `g` is the object returned by `sns.relplot()`.
    """)
    return


@app.cell
def _(df, sns):
    _g = sns.relplot(data=df, x='year', y='total_pitchers', aspect=1.5, kind='line')
    _g.ax.grid(axis='x', alpha=0.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a data frame `X` that selects only rows between the beginning and end years of the dip.
    """)
    return


@app.cell
def _(df):
    X = df[df["year"].between(1938, 1950)]
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then use the same plot function and parameters to visualize `X`.

    This time, show all the years on the x-axis by adding the following line:

    `g.ax.set_xticks(X.index.to_list())`
    """)
    return


@app.cell
def _(X, sns):
    _g = sns.relplot(data=X, x='year', y='total_pitchers', aspect=1.5, kind='line')
    _g.ax.set_xticks(X.index.to_list())
    _g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now show the same data frame as a bar plot using Seaborn's `catplot`.

    Set the aspect to $1.5$.

    Do not make any modifications to the axes object.

    What difference do you see in the default behavior between the bar and line plots?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Answer**: The y axis **limits** are different.
    """)
    return


@app.cell
def _(X, sns):
    sns.catplot(data=X, x="year", y="total_pitchers", aspect=1.5, kind="bar")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's redo the line plot so that it has the same y-axis as the bar plot.

    You can do this with `g.ax.set_ylim(bottom=0)`

    Keep all the axes modifications you made to your earlier line plot.

    Also add horizontal grid lines.
    """)
    return


@app.cell
def _(X, sns):
    _g = sns.relplot(data=X, x='year', y='total_pitchers', aspect=1.5, kind='line')
    _g.ax.set_xticks(X.index.to_list())
    _g.ax.grid(axis='y', alpha=0.3)
    _g.ax.grid(axis='x', alpha=0.3)
    _g.ax.set_ylim(bottom=0)
    _g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Challenge 2: Data/Ink Ratio
    """)
    return


@app.cell
def _(sns):
    def clean(plt, ax):
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set(xlabel=None, ylabel=None)
        ax.tick_params(
            axis='both',
            which='both',
            left=False,
            bottom=False,
            labelbottom=False,
            labelleft=False
        )
        plt
    return (clean,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the spirit of increasing the data-to-ink ratio of the plot, remove the following:

    - x and y axis labels
    - all spines
    - tick marks

    Also, set the title to "Total Pitches by Year".
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a function that creates similarly "clean" line plots.

    Apply it to another feature, e.g. `wild_pitches`.
    """)
    return


@app.cell
def _(X, clean, sns):
    _g = sns.relplot(data=X, x='year', y='total_pitchers', kind='line', aspect=1.5)
    _g.ax.set_title('Total Pitchers by Year')
    clean(_g, _g.ax)
    _g = sns.relplot(data=X, x='year', y='wild_pitches', kind='line', aspect=1.5)
    _g.ax.set_title('Wild Pitches by Year')
    clean(_g, _g.ax)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Challenge 3: Sparklines

    It would be nice to see all the trends at once without getting bogged down in the numbers.

    One way to compare a bunch of trends at once it with sparklines.

    Here is a short [article on the topic](https://www.edwardtufte.com/notebook/sparkline-theory-and-practice-edward-tufte/) by Tufte. Below is an example from the essay:

    <img src="https://s3.amazonaws.com/edwardtufte.com/sparkline_twitter.png" style="width:90%;">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use Pandas' `.plot()` with `subplots=True` on `df` to create this effect.

    Set `figsize=(5, len(df.columns))` and turn of the legend.

    This will create a list of line plots.

    To achieve the sparkline effect, do the following:

    - Remove as much non-data-ink as possible, including labels, spines, and ticks.

    - Keep the y axis label and make it bold and large.

    Play around with other parameters to get the effect you want.

    Your result should look something like this:

    <img src="short-sparklines-example.png" />
    """)
    return


@app.cell
def _(df, num_cols, plt):
    plot_result = df.plot(
        figsize=(5, num_cols * 0.7),
        subplots=True, legend=False,
        layout=(num_cols, 1), sharex=True, lw=2
    )

    if isinstance(plot_result, tuple):
        fig, axes_array = plot_result
    else:
        axes_array = plot_result
        fig = axes_array.flat[0].figure

    axes = axes_array.flatten()

    for i, ax in enumerate(axes):
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(
            axis='both', which='both',
            left=False, bottom=False,
            labelleft=False, labelbottom=False
        )
        ax.set_xlabel(None)
        col_name = df.columns[i]
        ax.set_ylabel(
            col_name.replace("_", " "),
            rotation=0, fontsize=8,
            fontweight='bold', ha='left',
            va='center', labelpad=50,
        )

    fig.subplots_adjust(left=0.2)
    fig.tight_layout()
    fig.subplots_adjust(left=0.001, hspace=0.3, right=0.998, top=0.98, bottom=0.02)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Participation

    Submit a PDF of this notebook as your participation.
    """)
    return


if __name__ == "__main__":
    app.run()
