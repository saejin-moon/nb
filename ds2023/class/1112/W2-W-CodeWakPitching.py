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
    # MLB Pitching

    Data Source: https://www.kaggle.com/datasets/pschale/mlb-pitch-data-20152018/code
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import libraries
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return pd, plt, sns


@app.cell
def _(sns):
    sns.set(context='notebook', style='white')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Acquire remote data
    """)
    return


@app.cell
def _():
    # pip install kagglehub
    return


@app.cell
def _():
    import kagglehub
    path = kagglehub.dataset_download("pschale/mlb-pitch-data-20152018")
    print("Path to dataset files:", path)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Data Path:** `/home/sace/.cache/kagglehub/datasets/pschale/mlb-pitch-data-20152018/versions/2`
    """)
    return


@app.cell
def _():
    data_path = "/home/sace/.cache/kagglehub/datasets/pschale/mlb-pitch-data-20152018/versions/2"
    return (data_path,)


app._unparsable_cell(
    r"""
    ls {data_path}
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read in tables
    """)
    return


@app.cell
def _(data_path, pd):
    ATBATS = pd.read_csv(f"{data_path}/atbats.csv").set_index('ab_id')
    PITCHES = pd.read_csv(f"{data_path}/pitches.csv")
    PITCH_TYPES = pd.read_csv("./pitch-types.csv").set_index("pitch_type")
    PLAYERS = pd.read_csv(f"{data_path}/player_names.csv").set_index('id')
    PLAYERS['full_name'] = PLAYERS.first_name + ' ' + PLAYERS.last_name
    ALL = PITCHES.join(ATBATS, on='ab_id').join(PITCH_TYPES, on='pitch_type')
    ALL['pitcher_name'] = ALL.pitcher_id.map(PLAYERS.full_name)
    return (ALL,)


@app.cell
def _(ALL):
    ALL
    return


@app.cell
def _(ALL):
    ALL.columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparing Left and Right Handed Pitchers
    """)
    return


@app.cell
def _(ALL):
    PITCH_TYPE_HAND = ALL[ALL.include == 1].value_counts(['p_throws', 'pitch_desc']).unstack(fill_value=0).T
    return (PITCH_TYPE_HAND,)


@app.cell
def _(PITCH_TYPE_HAND):
    PITCH_TYPE_HAND_1 = (PITCH_TYPE_HAND / PITCH_TYPE_HAND.sum()).round(2).sort_values('R').tail(10)
    return (PITCH_TYPE_HAND_1,)


@app.cell
def _(PITCH_TYPE_HAND_1):
    PITCH_TYPE_HAND_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grouped bar chart
    """)
    return


@app.cell
def _():
    shared_chart_title = "Pitch Types by Pitcher's Handedness"
    return (shared_chart_title,)


@app.cell
def _(PITCH_TYPE_HAND_1):
    PITCH_TYPE_HAND_NARROW = PITCH_TYPE_HAND_1.stack().to_frame('p')
    return (PITCH_TYPE_HAND_NARROW,)


@app.cell
def _(PITCH_TYPE_HAND_NARROW):
    PITCH_TYPE_HAND_NARROW.head()
    return


@app.cell
def _(PITCH_TYPE_HAND_NARROW, plt, sns):
    g = sns.catplot(PITCH_TYPE_HAND_NARROW.sort_values('p', ascending=False), x= 'p', y='pitch_desc', hue='p_throws', kind='bar', aspect=1.5)
    sns.despine(left=True, bottom=True)
    plt.grid(axis='x', color='lightgray', lw=.5)
    plt.xlabel('')
    plt.ylabel('')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scatter plot
    """)
    return


@app.cell
def _(PITCH_TYPE_HAND_1, plt, shared_chart_title, sns):
    g_1 = sns.relplot(PITCH_TYPE_HAND_1, x='R', y='L', kind='scatter', height=8)
    g_1.ax.plot([-0.01, 0.38], [-0.01, 0.38], ls='--', color='red', lw=1)
    # Create a diagonal 
    for idx, row in PITCH_TYPE_HAND_1.iterrows():
    # g.ax.set_ylim(-.01,.38)
    # g.ax.set_xlim(-.01,.38)
        g_1.ax.annotate(idx, xy=(row.R, row.L), xytext=(row.R - 0.01, row.L + 0.004), c='gray', rotation=0)
    # Label the points
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    # Reduce ink
    plt.ylabel('')
    sns.despine(left=True, bottom=True)
    plt.text(0.15, 0.25, 'Left-handed\nbias', color='green', ha='center', va='center', rotation=45)
    plt.text(0.25, 0.15, 'Right-handed\nbias', color='green', ha='center', va='center', rotation=45)
    plt.title(shared_chart_title)
    # Add labels
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Butterfly Chart
    """)
    return


@app.cell
def _(PITCH_TYPE_HAND_1):
    X = PITCH_TYPE_HAND_1
    return (X,)


@app.cell
def _(X, plt, shared_chart_title, sns):
    # Define figure parameters
    fig = plt.figure(figsize=(10, len(X)/2), dpi=150)

    # Create an axes to draw on
    ax = plt.subplot()

    # Note you can do both at once ...
    # fig, ax = plt.subplots(figsize=(10, len(X)/2), dpi=150)

    # Plot the bars, flipping the left with negation
    ax.barh(y=X.index, width=-X.L, alpha=.75, color='lightblue', label="left")
    ax.barh(y=X.index, width=X.R, alpha=.75, label="right")

    # Create individual bar text labels
    text_props = {'c': 'gray', 'va': 'center'}
    for y in X.index:
        x1 = X.loc[y].L
        x2 = X.loc[y].R
        x1_label = str(round(x1 * 100))
        x2_label = str(round(x2 * 100))
        ax.text(-(x1 + .01 * len(x1_label)), y, x1_label, **text_props)
        ax.text(+x2, y, x2_label, **text_props)

    plt.legend(frameon=False, loc="lower right")

    # Reduce ink
    sns.despine(left=True, bottom=True)
    ax.set_xticks([])
    ax.grid(axis='y', color='#F0F0F0', ls='--')

    plt.title(shared_chart_title)
    plt.show()
    return


@app.cell
def _(ALL, plt, sns):
    g_2 = sns.relplot(ALL[ALL.include == 1].sample(10000), y='spin_rate', x='spin_dir', kind='scatter', col='pitch_desc', col_wrap=3, hue='p_throws')
    for i, ax_1 in enumerate(g_2.axes):
        ax_1.set_title(ax_1.get_title().split('=')[1])
    plt.show()
    return


@app.cell
def _(ALL, plt, sns):
    g_3 = sns.relplot(ALL[ALL.include == 1].sample(10000), x='break_angle', y='break_length', kind='scatter', col='pitch_desc', col_wrap=3, hue='p_throws')
    for i_1, ax_2 in enumerate(g_3.axes):
        ax_2.set_title(ax_2.get_title().split('=')[1])
    plt.show()
    return


if __name__ == "__main__":
    app.run()
