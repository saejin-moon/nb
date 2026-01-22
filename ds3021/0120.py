import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.array([1,3,5])
    y = np.log(x)

    plt.plot(x,y)
    plt.show()
    return


@app.cell
def _():
    import pandas as pd

    df = pd.read_csv('./data/airbnb_NYC.csv',
                     encoding='latin1')

    df.head()
    return df, pd


@app.cell
def _(df, pd):
    price = df['Price'].str.replace(',', '')
    df['Price'] = pd.to_numeric(price, errors = "coerce")
    return


@app.cell
def _(df):
    df.loc[df["Price"] < 350, :]
    return


@app.cell
def _(df):
    df.loc[:, ["Price", "Zipcode"]]
    return


@app.cell
def _(df):
    df.to_csv("./data/mod/airbnb.csv", index=False)
    df.to_parquet("./data/mod/airbnb.parquet", index=False, engine="fastparquet")
    return


@app.cell
def _(pd):
    gifts = pd.read_csv("./data/ForeignGifts_edu.csv")
    gifts.head()
    return (gifts,)


@app.cell
def _(gifts):
    gifts["Gift Type"] = gifts["Gift Type"].str.lower()
    gifts.head()
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
