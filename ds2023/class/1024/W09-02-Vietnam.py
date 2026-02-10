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
    # W09 Vietnam Memorial
    """)
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    return pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get Casuality Data

    https://repository.duke.edu/catalog/29367025-709c-4c31-93fb-d022a2a09a3b
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("CACCF.csv", low_memory=False)

    # Remove columns with nulls
    df = df.T.dropna().T

    # Handle dates
    df['died_date'] = pd.to_datetime(df['DIED'], format='%m/%d/%y')
    df['died_year'] = df.DIED.str.split("/").str[-1].astype(int)
    df['died_month'] = df.DIED.str.split("/").str[0].astype(int)
    df['died_day'] = df.DIED.str.split("/").str[1].astype(int)

    # Include only active war years
    df = df.query("died_year < 76 and died_year > 54").copy()
    return (df,)


@app.cell
def _(df):
    G = df.groupby(['died_year','died_month'])
    return (G,)


@app.cell
def _(G):
    DIED = G.DIED.count().unstack(fill_value=0)
    return (DIED,)


@app.cell
def _(DIED):
    DIED.head()
    return


@app.cell
def _(DIED, plt):
    DIED.plot.bar(rot=90, figsize=(10, 5), stacked=True, cmap=plt.get_cmap('tab20'))
    plt.show()
    return


@app.cell
def _(DIED):
    DIED.iloc[:(75-56),:].style.background_gradient(axis=None, cmap='Reds')
    return


@app.cell
def _(DIED, plt):
    DIED.sum().plot.bar()
    plt.title("Deaths by Month")
    plt.show()
    return


@app.cell
def _(DIED):
    DIED2 = DIED.stack().to_frame('n')
    DIED2['date'] = DIED2.apply(lambda x: f"{str(x.name[1]).zfill(2)}/{x.name[0]}", axis=1)
    DIED2 = DIED2.reset_index().set_index('date')
    return (DIED2,)


@app.cell
def _(DIED2):
    DIED2
    return


@app.cell
def _(DIED2, plt):
    DIED2.n.plot(legend=False, style=".-", figsize=(10,5))
    plt.title("US Casualities in Viet Nam 1965 – 1975")
    plt.show()
    return


@app.cell
def _(DIED2):
    max_idx = DIED2.n.idxmax()
    max_idx_row_number = DIED2.index.get_loc(max_idx)
    return max_idx, max_idx_row_number


@app.cell
def _(DIED2, max_idx, max_idx_row_number, plt):
    DIED2.n.cumsum().plot.line(x = 'date', y = 'n', style=".-", figsize=(10,5))
    plt.title("US Casualities in Viet Nam 1965 – 1975")
    plt.axvline(max_idx_row_number, ls='--', c='red', lw=2, label=f'Peak: {max_idx}')
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Names List
    """)
    return


@app.cell
def _():
    month_names = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    return (month_names,)


@app.cell
def _(df):
    NAMES = (
        df[df.died_year == 68]
            .groupby(['died_month','died_day'])
            .apply(lambda x: "\n".join(x.NAME), include_groups=False)
            .to_frame('names')
            .reset_index()
    )
    return (NAMES,)


@app.cell
def _(NAMES):
    NAMES.head()
    return


@app.cell
def _(NAMES):
    NAMES.names = NAMES.names.str.replace(" ", "&nbsp;")
    return


@app.cell
def _(NAMES, month_names):
    ROWS = NAMES.apply(lambda x: f"<td><b>{month_names[x.died_month - 1]} {x.died_day}</b><br /><hr />" + x.names.replace("\n", "<br />") + "</td>", axis=1)
    return (ROWS,)


@app.cell
def _(ROWS):
    ROWS.head()
    return


@app.cell
def _(ROWS):
    TABLE = "\n".join(ROWS.values.tolist())
    return (TABLE,)


@app.cell
def _(TABLE):
    HTML = f"""
    <html>
        <head>
            <title>Deaths by Month and Day</title>
            <style type="text/css">
    body {{
        color: white;
        background-color: black;
    }}
    td {{
        vertical-align: top;
        font-size: 8px;
        width: 30px;
    }}
            </style>
        </head>
        <body>
            <h1>1968</h1>
            <table>
                <tr>
                {TABLE}
                </tr>
            </table>
        </body>
    </html>
    """
    return (HTML,)


@app.cell
def _(HTML):
    with open('./vn-table.html', 'w') as outfile:
        outfile.write(HTML)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![image.png](attachment:image.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
