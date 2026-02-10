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
    # W W13 M Activity: Challenge

    DS 2023 | Communicating with Data

    **Instructions**

    Using the provided data set `pitching-sample.csv`, create a simple dashboard using IPyWidgets, Matplotlib, and Seaborn that matches the following screenshot:

    ![](example.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Solution
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns

    import ipywidgets as widgets
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("./pitching-sample.csv").set_index(['ab_id','pitch_num'])
    return


@app.cell
def _():
    # CODE
    return


if __name__ == "__main__":
    app.run()
