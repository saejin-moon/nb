import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    df = pd.read_csv("./data/2022 election cycle fundraising.csv")
    df.head()
    return df, pd


@app.cell
def _(df):
    df["Raised"] = df["Raised"].astype("string").str.replace(r'[\$,]', '', regex=True).astype(int)
    df["Spent"] = df["Spent"].astype("string").str.replace(r'[\$,]', '', regex=True).astype(int)
    df["Cash on Hand"] = df["Cash on Hand"].astype("string").str.replace(r'[\$,]', '', regex=True).astype(int)
    df["Debts"] = df["Debts"].astype("string").str.replace(r'[\$,]', '', regex=True).astype(int)
    df.head()
    return


@app.cell
def _(df):
    import seaborn as sns
    sns.kdeplot(df["Raised"])
    return (sns,)


@app.cell
def _(pd):
    gifts = pd.read_csv("./data/ForeignGifts_edu.csv")
    gifts.head()
    return (gifts,)


@app.cell
def _(gifts, sns):
    import matplotlib.pyplot as plt
    sns.kdeplot(x=gifts["Foreign Gift Amount"], hue=gifts["Gift Type"],common_norm=False)
    plt.xlim(0, 2000000)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
