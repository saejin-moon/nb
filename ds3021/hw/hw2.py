import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return mo, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Assignment 1: Wrangling and EDA
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q1.**
    """)
    return


@app.cell
def _(pd):
    df_airbnb = pd.read_csv('../data/airbnb_NYC.csv', encoding="latin1")

    # Q1.1: Remove "$" and ","
    df_airbnb['price_clean'] = pd.to_numeric(
        df_airbnb['Price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
    )
    print("Q1.1 Missing Prices:", df_airbnb['price_clean'].isna().sum())

    # Q1.2: Data appears to be disproportionately missing when bodily force was used.
    df_police = pd.read_csv('../data/mn_police_use_of_force.csv')
    print("Q1.2 Prop Missing:", df_police['subject_injury'].isna().mean())
    print(pd.crosstab(df_police['subject_injury'].fillna('NaN'), df_police['force_type']))
    df_police['subject_injury_clean'] = df_police['subject_injury'].fillna('Missing')

    # Q1.3
    df_metabric = pd.read_csv('../data/metabric.csv')
    df_metabric['survival_dummy'] = df_metabric['Overall Survival Status'].map({'Living': 1, 'Deceased': 0})

    # Q1.4: If missing values are correlated with negative experiences, this artificially inflates ratings.
    missing_reviews = df_airbnb['Review Scores Rating'].isna().sum()
    median_rating = df_airbnb['Review Scores Rating'].median()
    df_airbnb['review_imputed'] = df_airbnb['Review Scores Rating'].fillna(median_rating)
    print("Q1.4 Missing Reviews:", missing_reviews)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q2.**
    """)
    return


@app.cell
def _(pd, plt, sns):
    # Q2.1
    df_shark = pd.read_excel("../data/GSAF5.xls")

    # Q2.2
    df_shark = df_shark.dropna(axis=1, how='all')

    # Q2.3: An observation is an event that involves a shark and a human/ship. Some of the rows in the dataset contain invalid reports/scavenging which aren't attacks.

    # Q2.4: Attacks are increasing over time.
    df_shark['Year_Clean'] = pd.to_numeric(df_shark['Year'], errors='coerce')
    df_recent = df_shark[df_shark['Year_Clean'] >= 1940].copy()
    plt.figure()
    sns.histplot(data=df_recent, x='Year_Clean', bins=range(1940, 2025))
    plt.title("Attacks per Year")
    plt.show()

    # Q2.5
    df_recent['Age_Clean'] = pd.to_numeric(df_recent['Age'], errors='coerce')
    plt.figure()
    sns.histplot(data=df_recent, x='Age_Clean', bins=20)
    plt.title("Age Distribution")
    plt.show()

    # Q2.6
    def clean_type(val):
        val = str(val).lower()
        if "unprovoked" in val: return "Unprovoked"
        if "provoked" in val: return "Provoked"
        return "Unknown"
    df_recent['Type_Clean'] = df_recent['Type'].apply(clean_type)
    print("Q2.6 Unprovoked Prop:", (df_recent['Type_Clean'] == 'Unprovoked').mean())

    # Q2.7
    def clean_fatal(val):
        val = str(val).upper()
        if 'Y' in val: return 'Y'
        if 'N' in val: return 'N'
        return "Unknown"
    df_recent['Fatal_Clean'] = df_recent['Fatal Y/N'].apply(clean_fatal)
    print(pd.crosstab(df_recent['Type_Clean'], df_recent['Fatal_Clean'], normalize='index'))

    # Q2.8: Attacks are more likely to be fatal when unprovoked, possibly due to unprovoked attacks being attributed to a shark's predatory behavior.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q3.**
    1.  The paper is about tidy data, standardizing data cleaning, and making data analysis more efficient.

    2.  The standard is meant to ensure compatibility across tools and reduce time spent on data wrangling.

    3.
    > "Like families, tidy datasets are all alike but every messy dataset is messy in its own way"

    Tidy datasets share a singular structure, but messy datasets are uniquely chaotic.
    > "For a given dataset, it’s usually easy to figure out what are observations and what are variables, but it is surprisingly difficult to precisely define variables and observations in general."

    It's a simple task to identify the observations and variables but to understand those observations and variables is a much more difficult task.

    4.
    - Value -- Measure (cell).
    - Variable -- Attribute across units (column).
    - Observation -- All measures for a unit (row).

    5. Tidy data is defined as variables being columns, observations being rows, and observational unit types being tables.

    6.
    - Problems -- Headers as values, multiple variables in one column, variables in rows/columns, mixed units, split units.
    - Headers -- (`<$10k`) are values of the "Income" variable, not variable names.
    - Melting -- Stacking columns into rows (converting Wide Data to Long Data).
    7.  Table 11 has the "Day" variable across headers (messy) while Table 12 has them altogether in the Date column (tidy/molten).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q4.**
    """)
    return


@app.cell
def _(pd, plt, sns):
    # Q4.1
    df_gifts = pd.read_csv('../data/ForeignGifts_edu.csv')

    # Q4.2
    plt.figure()
    sns.histplot(df_gifts['Foreign Gift Amount'], log_scale=True)
    plt.title("Gift Amounts (Log)")
    plt.show()
    print(df_gifts['Foreign Gift Amount'].describe())

    # Q4.3
    print(df_gifts['Gift Type'].value_counts(normalize=True))

    # Q4.4
    print(df_gifts['Country of Giftor'].value_counts().head(15))
    print(df_gifts.groupby('Country of Giftor')['Foreign Gift Amount'].sum().nlargest(15))

    # Q4.5
    inst_sums = df_gifts.groupby('Institution Name')['Foreign Gift Amount'].sum().nlargest(15)
    print(inst_sums)
    plt.figure()
    sns.histplot(df_gifts.groupby('Institution Name')['Foreign Gift Amount'].sum(), log_scale=True)
    plt.title("Total by Institution")
    plt.show()

    # Q4.6
    print(df_gifts.groupby('Giftor Name')['Foreign Gift Amount'].sum().nlargest(15))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q5.**
    """)
    return


@app.cell
def _(pd, plt, sns):
    # Q5.1
    df_college = pd.read_csv('../data/college_completion.csv')

    # Q5.2
    print(df_college.shape)
    print(df_college.head())

    # Q5.3: Private for-profit colleges are evenly split on being 2-year or 4-year programs. Private not-for-profit colleges are mostly 4-year programs. Public colleges are more likely to be 2-year programs.
    print(pd.crosstab(df_college['control'], df_college['level']))

    # Q5.4
    plt.figure()
    sns.kdeplot(data=df_college, x='grad_100_value', hue='control', fill=True)
    plt.show()
    print(df_college.groupby('control')['grad_100_value'].describe())

    # Q5.5: Private not-for-profit institutions appears to have aid vary positively with graduation rates.
    plt.figure()
    sns.scatterplot(data=df_college, x='aid_value', y='grad_100_value', hue='control', alpha=0.5)
    plt.show()

    numeric = ['aid_value', 'grad_100_value']
    print("Overall Corr:", df_college[numeric].corr().iloc[0, 1])
    print(df_college.groupby('control')[numeric].corr().iloc[0::2, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q6. (note: I use LaTeX daily for notes in a Marimo notebook, so I found this to be the easiest medium for my answers)**
    1.  **Mean:** $m(a+bX) = \frac{1}{N}\sum(a+bx_i) = a + b \frac{1}{N}\sum x_i = a + b m(X)$.
    2.  **Cov(X,X):** $\frac{1}{N}\sum(x_i-m(X))^2 = s^2$.
    3.  **Cov(X, a+bY):** $\text{cov}(X, a+bY) = b \cdot \text{cov}(X,Y)$ (constants vanish, scale factors out).
    4.  **Cov(a+bX, a+bY):** $b \cdot \text{cov}(a+bX, Y) = b \cdot b \cdot \text{cov}(X,Y) = b^2 \text{cov}(X,Y)$.
    5.  **Median/IQR:** Yes for both (if $b>0$). Linear transformations preserve order (Median) and scale distances (IQR).
    6.  **Non-linear:** $X=\{1,2,3\}$. $m(X)=2, m(X)^2=4$. $X^2=\{1,4,9\}, m(X^2)=4.66 \neq 4$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Q7.**
    """)
    return


@app.cell
def _(pd, plt, sns):
    # Q7.1
    df_ames = pd.read_csv('../data/ames_prices.csv')

    # Q7.2: Townhouses are the most expensive, on average. Single-family homes have the highest variance in transaction prices.
    plt.figure()
    sns.kdeplot(df_ames['price'])
    plt.show()
    print(df_ames['price'].describe())
    plt.figure()
    sns.kdeplot(data=df_ames, x='price', hue='Bldg.Type')
    plt.show()
    print(df_ames.groupby('Bldg.Type')['price'].describe().sort_values('mean', ascending=False))

    # Q7.3
    plt.figure()
    sns.ecdfplot(df_ames['price'])
    plt.show()
    print(df_ames['price'].quantile([0, 0.25, 0.5, 0.75, 1]))

    # Q7.4: Yes, there are lots of outliers. I don't see many patterns not already mentioned in my response to Q7.2. There are more outliers in Single-family houses than any other building type but that is simply due to the fact that they are overrepresented in our dataset.
    plt.figure()
    sns.boxplot(x=df_ames['price'])
    plt.show()
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df_ames['price'], y=df_ames['Bldg.Type'])
    plt.show()

    # Q7.5
    Q1 = df_ames['price'].quantile(0.25)
    Q3 = df_ames['price'].quantile(0.75)
    IQR = Q3 - Q1
    df_ames['is_outlier'] = ((df_ames['price'] < (Q1 - 1.5 * IQR)) | (df_ames['price'] > (Q3 + 1.5 * IQR))).astype(int)

    # Q7.6: The results change by letting us see a second minor peak around 325k that was not visible in the plot from Q7.2.
    p05 = df_ames['price'].quantile(0.05)
    p95 = df_ames['price'].quantile(0.95)
    df_ames['price_winsorized'] = df_ames['price'].clip(p05, p95)
    plt.figure()
    sns.kdeplot(df_ames['price_winsorized'])
    plt.show()
    print(df_ames['price_winsorized'].describe())
    return


if __name__ == "__main__":
    app.run()
