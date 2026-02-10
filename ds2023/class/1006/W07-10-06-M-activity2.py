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
    # Alluvial Diagrams Challenge KEY

    DS 2023 | Communicating with Data

    Create alluvial diagrams of various data sets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Intructions

    Using the code given below, create alluvial diagrams of the following data sets:

    - Penguins
    - Titanic
    - Taxis

    Submit this notebook as your participation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set Up
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Import Libraries
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    from seaborn import load_dataset
    return load_dataset, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Read in Data
    """)
    return


@app.cell
def _(load_dataset):
    PENGUINS = load_dataset('penguins')
    TITANIC = load_dataset('titanic')
    TAXIS = load_dataset('taxis')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define Plotting Function
    """)
    return


@app.cell
def _(px):
    def alluvialplot(df, dim_cols:list, color_col:str, title=None):
    
        # The color column must be a category
        if df[color_col].dtype != 'category':
            df[color_col] = df[color_col].astype('category')
    
        fig = px.parallel_categories(
            df, 
            dimensions=dim_cols,
            color=df[color_col].cat.codes,
            height=1000,
            title=title
        )
        fig.update_traces(line={'shape': 'hspline'})
        fig.update_layout(coloraxis_showscale=False)
        fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualize Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Penguins

    Show species, island, and sex, coloring by island.
    """)
    return


@app.cell
def _():
    # CODE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Titanic

    Show sex and survived colored by sex.
    """)
    return


@app.cell
def _():
    # CODE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Show sex, class, and survived, colored by sex.
    """)
    return


@app.cell
def _():
    # CODE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Show class, sex, and survived, colored by sex.
    """)
    return


@app.cell
def _():
    # CODE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Taxis

    Show pickup_borough and dropoff_borough, colored by pickup_borough.
    """)
    return


@app.cell
def _():
    # CODE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Show pickup_borough, passengers, and dropoff_borough, colored by pickup_borough.
    """)
    return


@app.cell
def _():
    # CODE
    return


if __name__ == "__main__":
    app.run()
