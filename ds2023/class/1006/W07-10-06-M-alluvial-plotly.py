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
    # Alluvial Diagrams with Plotly

    DS 2023 | Communicating with Data
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.templates.default = "plotly_dark"
    return pd, px


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plotly's Parallel Categories
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Tips
    """)
    return


@app.cell
def _(px):
    TIPS = px.data.tips()
    return (TIPS,)


@app.cell
def _(TIPS):
    TIPS.head()
    return


@app.cell
def _(TIPS, px):
    px.parallel_categories(TIPS)
    return


@app.cell
def _(TIPS, px):
    px.parallel_categories(TIPS, 
            dimensions=['sex', 'smoker', 'day'],
            color="size", 
            color_continuous_scale=px.colors.qualitative.G10,
            labels={'sex':'Payer sex', 'smoker':'Smokers at the table', 'day':'Day of week'})
    return


@app.cell
def _(TIPS, px):
    fig = px.parallel_categories(TIPS, 
            dimensions=['sex', 'smoker', 'day'],
            color="size", color_continuous_scale=px.colors.qualitative.G10,
            labels={'sex':'Payer sex', 'smoker':'Smokers at the table', 'day':'Day of week'})
    fig.update_traces(line={'shape': 'hspline'})
    fig.update_layout(coloraxis_showscale=False)
    fig.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Wine Reviews (Again :-)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let's get the data.
    """)
    return


@app.cell
def _(pd):
    data_file = "./winereviews-DOC_MOD.csv"
    DOC = pd.read_csv(data_file)
    DOC = DOC.dropna()
    return (DOC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And let's visualize only the most frequent values.
    """)
    return


@app.cell
def _(DOC):
    COUNTRY = DOC.value_counts('doc_country')
    top_countries = COUNTRY[COUNTRY > 1000].index

    TASTER = DOC.value_counts('doc_taster')
    top_tasters = TASTER[TASTER > 1000].index

    VARIETY = DOC.value_counts('doc_variety')
    top_varieties = VARIETY[VARIETY > 1000].index

    DOC2 = DOC[(DOC.doc_country.isin(top_countries)) & (DOC.doc_taster.isin(top_tasters)) & (DOC.doc_variety.isin(top_varieties))].copy()
    return (DOC2,)


@app.cell
def _(DOC2):
    DOC2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, let's convert our columns to categories.

    This will be useful in assigning color to our plots.
    """)
    return


@app.cell
def _(DOC2):
    DOC2.doc_country = DOC2.doc_country.astype('category')
    DOC2.doc_taster = DOC2.doc_taster.astype('category')
    DOC2.doc_variety = DOC2.doc_variety.astype('category')
    DOC2.doc_points = DOC2.doc_points.astype('category')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Country and Taster
    """)
    return


@app.cell
def _(DOC2, px):
    fig_1 = px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_taster'], color=DOC2.doc_country.cat.codes, height=1000)
    fig_1.update_traces(line={'shape': 'hspline'})
    fig_1.update_layout(coloraxis_showscale=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Color by Taster.
    """)
    return


@app.cell
def _(DOC2, px):
    fig_2 = px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_taster'], color=DOC2.doc_taster.cat.codes, color_continuous_scale=px.colors.qualitative.Set1, height=1000)
    fig_2.update_traces(line={'shape': 'hspline'})
    fig_2.update_layout(coloraxis_showscale=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Country and Variety
    """)
    return


@app.cell
def _(DOC2, px):
    fig_3 = px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_variety'], color=DOC2.doc_country.cat.codes, color_continuous_scale=px.colors.qualitative.Plotly, height=1000)
    fig_3.update_traces(line={'shape': 'hspline'})
    fig_3.update_layout(coloraxis_showscale=False)
    return (fig_3,)


@app.cell
def _(DOC2, fig_3, px):
    px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_variety'], color=DOC2.doc_variety.cat.codes, color_continuous_scale=px.colors.qualitative.Plotly, height=1000)
    fig_3.update_traces(line={'shape': 'hspline'})
    fig_3.update_layout(coloraxis_showscale=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Country and Points
    """)
    return


@app.cell
def _(DOC2, fig_3, px):
    px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_points'], color=DOC2.doc_country.cat.codes, color_continuous_scale=px.colors.qualitative.Set2, height=1000)
    fig_3.update_traces(line={'shape': 'hspline'})
    fig_3.update_layout(coloraxis_showscale=False)
    return


@app.cell
def _(DOC2, fig_3, px):
    px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_points'], color=DOC2.doc_points.cat.codes, color_continuous_scale=px.colors.qualitative.Set2, height=1000)
    fig_3.update_traces(line={'shape': 'hspline'})
    fig_3.update_layout(coloraxis_showscale=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Country, Taster, and Points
    """)
    return


@app.cell
def _(DOC2, fig_3, px):
    px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_taster', 'doc_points'], color=DOC2.doc_taster.cat.codes, height=1000)
    fig_3.update_traces(line={'shape': 'hspline'})
    fig_3.update_layout(coloraxis_showscale=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding Labels
    """)
    return


@app.cell
def _(DOC2, px):
    fig_4 = px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_taster'], labels={'doc_country': 'Country', 'doc_taster': 'Reviewer'}, color=DOC2.doc_country.cat.codes, height=1000, title='')
    fig_4.update_traces(line={'shape': 'hspline'})
    fig_4.update_layout(coloraxis_showscale=False)
    fig_4.show()
    return


@app.cell
def _(DOC2, px):
    fig_5 = px.parallel_categories(DOC2, dimensions=['doc_country', 'doc_taster', 'doc_points'], labels={'doc_country': 'Country', 'doc_taster': 'Reviewer', 'doc_points': 'Points'}, color=DOC2.doc_taster.cat.codes, height=1000, title='Reviewer Inputs and Outputs')  # .sort_values('doc_points', ascending=False)
    fig_5.update_traces(line={'shape': 'hspline'})
    fig_5.update_layout(coloraxis_showscale=False)
    fig_5.show()  # color=DOC2.doc_country.cat.codes,
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Handy-Dandy Function
    """)
    return


@app.cell
def _(px):
    def alluvialplot(df, dim_cols_dict:dict, color_col:str, title=None):

        # The color column must be a category
        if df[color_col].dtype != 'category':
            df[color_col] = df[color_col].astype('category')

        fig = px.parallel_categories(df, 
            dimensions=dim_cols_dict.keys(),
            labels=dim_cols_dict,
            color=df[color_col].cat.codes,
            height=1000,
            title=title
        )
        fig.update_traces(line={'shape': 'hspline'})
        fig.update_layout(coloraxis_showscale=False)
        fig.show()
    return (alluvialplot,)


@app.cell
def _(DOC2, alluvialplot):
    my_cols = {
        'doc_country': 'Country',
        'doc_taster': "Taster",
    }
    alluvialplot(DOC2, my_cols, 'doc_taster', "Country to Taster")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparison to Heatmaps
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The Good**

    Compare this heatmap to the previous alluvial diagram.

    The alluvial shows more clearly the dominance of certain tasters.
    """)
    return


@app.cell
def _(DOC2, px):
    px.imshow(DOC2.value_counts(['doc_country', 'doc_taster']).unstack(fill_value=0), color_continuous_scale=px.colors.colorbrewer.YlGnBu, height=1000)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <!-- <img src="country-taster-heatmap-nocluster.png" style="height:1000px;border:1px solid grey;"/> -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **The Bad**

    The alluvial here, however, is clearly more noisy than the heatmap.

    <img src="points-topics-sankey.png" />
    <img src="points-topics-heatmap.png" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway**:

    - Use Heatmaps when the data are more evenly distributed (high entropy)
    - Use Sankey when the data are more unevenly distributed (low entropy)
    """)
    return


if __name__ == "__main__":
    app.run()
