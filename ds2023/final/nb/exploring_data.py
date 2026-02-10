import marimo

__generated_with = "0.19.5"
app = marimo.App()


@app.cell
def _():
    # libraries
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import plotly.express as px

    # get dataset
    df = pd.read_csv("../data/data.csv")

    # get some percentages for text
    print((df["benefits"] == "Yes").mean() * 100)
    print((df["workplace_resources"] == "Yes").mean() * 100)

    # heatmap setup
    encoded_df = df.apply(lambda x: x.factorize()[0])
    corr = encoded_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # heatmap creation
    plt.figure(figsize=(24, 36))
    sns.heatmap(corr, mask=mask, cmap='Greys', cbar=False, linewidths=0)

    # heatmap modification
    plt.axis('off')

    # heatmap export
    plt.savefig("../images/background_texture.svg", format="svg", transparent=True, bbox_inches='tight')
    plt.close()

    # alluvial setup
    df["mh_coworker_discussion"] = df["mh_coworker_discussion"].sort_values(ascending=False).values
    df["mh_employer_discussion"] = df["mh_employer_discussion"].sort_values(ascending=False).values
    df['mh_coworker_discussion_color'] = df['mh_coworker_discussion'].map({'Yes': 1, 'No': 0})
    # alluvial creation
    fig = px.parallel_categories(df, 
            dimensions=['mh_employer_discussion', 'mh_coworker_discussion'],
            color="mh_coworker_discussion_color", color_continuous_scale=['#574748', '#fcb713'],
            labels={'mh_employer_discussion':'Talks to Employer', 'mh_coworker_discussion':'Talks to Coworker'})

    # alluvial modification
    fig.update_traces(
        line={'shape': 'hspline'},
        labelfont=dict(color='rgba(0,0,0,0)'), 
        tickfont=dict(color='rgba(0,0,0,0)')
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # alluvial export
    fig.write_image("../images/alluvial.svg")

    # loop used for all three stacked bar charts
    for col in ['benefits', 'workplace_resources', 'mh_employer_discussion']:
        # stacked bar chart prep
        plt.figure(figsize=(8, 1))
        data_counts = df[col].value_counts(normalize=True).to_frame().T
        if col == "mh_employer_discussion":
            data_counts = data_counts[['Yes', 'No']] 
            my_colors = ["#fcb713", "#574748"]
        else:
            data_counts = data_counts[["Yes", "I don't know", "No"]]
            my_colors = ["#fcb713", "#574748", "#574748"]

        # stacked bar chart creation
        ax = data_counts.plot(
            kind='barh', 
            stacked=True, 
            legend=False, 
            color=my_colors,
            edgecolor='none'
        )

        # stacked bar chart modification
        plt.axis('off')

        # stacked bar chart export
        plt.savefig(f"../images/{col}_stacked.svg", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()

    print("Assets generated successfully.")
    return


if __name__ == "__main__":
    app.run()
