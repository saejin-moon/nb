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
    # Establishing the Data
    ## Acquire and read the data source(s)
    """)
    return


@app.cell
def _():
    import pandas as pd
    df = pd.read_csv("../data/raw.csv")
    # some quick processing since i have no earthly idea why the age is considered a float even though it was recorded as an integer.
    df["age"] = df["age"].round(0).astype(int)
    df = df[df["tech_company"] == "Yes"]
    df.to_csv('../data/data.csv', index=False)
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Describe How to Get the Data
    Visit [this URL](https://data.mendeley.com/datasets/mmnzx4w8cg/1) and download the CSV file (data.csv) on the site.
    I could not find the direct URL for the CSV file.

    ## Describe Who Produced the Data and How
    Open Sourcing Mental Illness (a non-profit) conducted the surveys that lasted five years (2017-2021) that were compiled for this dataset to better understand the attitudes and opportunities surrounding mental health in the workplace of the tech industry.

    ## Describe the Data Features with a `COLS` Table
    """)
    return


@app.cell
def _(df):
    cols = df.dtypes.reset_index()
    cols.columns = ["Feature Name", "Pandas Data Type"]
    cols["Description"] = [
        "Is your employer primarily a tech company/organization?",
        "Does your employer provide mental health benefits as part of healthcare coverage?",
        "Does your employer offer resources to learn more about mental health disorders and options for seeking help?",
        "Have you ever discussed your mental health with your employer?",
        "Have you ever discussed your mental health with coworkers?",
        "Do you have medical coverage (private insurance or state-provided) that includes treatment of mental health disorders?",
        "Do you currently have a mental health disorder?",
        "How willing would you be to share with friends and family that you have a mental illness?",
        "What is your age?",
        "What is your gender?",
        "What country do you live in?"
    ]
    cols = cols[["Feature Name", "Description", "Pandas Data Type"]]
    cols
    return


if __name__ == "__main__":
    app.run()
