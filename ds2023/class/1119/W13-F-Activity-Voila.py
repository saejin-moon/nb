import marimo

__generated_with = "0.19.5"
app = marimo.App()


@app.cell
def _():
    # W W13 F Code Walk: Using Voila 
    # DS 2023 | Communicating with Data

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import ipywidgets as widgets
    # from ipywidgets import Layout
    from IPython.display import HTML, display

    # Read in the data
    df = pd.read_csv("./pitching-sample.csv").set_index(['ab_id','pitch_num'])

    ### Plot 1 ### 

    def plot_pitches(p_hand, p_type):

        # The x and y values for the two scatter plots
        x1 = 'break_angle'
        y1 = 'break_length'
        x2 = 'spin_dir'
        y2 = 'spin_rate'

        # The filtered data frame
        X = df.query(f"pitch_type == '{p_type}' & p_throws == '{p_hand}'")

        # Handling the case where no rows are returned
        hand = {'L':'Left-handed', 'R':'Right-handed'}
        if len(X) == 0:
            display(f"There were no {p_type.lower()} pitches by {hand[p_hand].lower()}-handed pitchers.")

        # The viz
        else:

            # Create a figure with 2 axes objects
            fig, ax = plt.subplots(1, 2, figsize=(12,5))

            # Plot 1
            sns.scatterplot(X, x=x1, y=y1, hue='stand', ax=ax[0])
            ax[0].set_title("Break Angle and Length")
            ax[0].legend().remove()

            # Plot 2
            sns.scatterplot(X, x=x2, y=y2, hue='stand', ax=ax[1])
            ax[1].set_title("Spin Direction and Rate")

            sns.despine()

    pitch_type_widget = widgets.Dropdown(
        options= zip(list(df.pitch_name.unique()), list(df.pitch_type.unique())),
        value='SL',
        description='Pitch type:',
    )

    pitcher_hand_widget = widgets.RadioButtons(
        options= zip(['Left','Right'], list(df.p_throws.unique())),
        value='R',
        description='Pitcher hand:'
    )

    iplot1 = widgets.interactive(plot_pitches, p_hand=pitcher_hand_widget, p_type=pitch_type_widget)

    ### PLOT 2 ###

    def plot_pitches2(p_type, f_pair):
        x, y = f_pair
        X = df.query(f"pitch_type == '{p_type}'")
        plt.figure(figsize=(6,5))
        sns.scatterplot(X, x=x, y=y, hue='p_throws')
        sns.rugplot(X, x=x, y=y, ax=plt.gca(), hue='p_throws')
        sns.despine()

    feature_pairs = [
        ('break_angle', 'break_length'),
        ('spin_dir', 'spin_rate'),
        ('start_speed', 'end_speed'),
        ('px', 'pz'),
    ]
    feature_pairs_option_list = [(" vs ".join([o.replace("_"," ") for o in opt]), opt) for opt in feature_pairs]
    feature_pair_widget = widgets.Dropdown(
        options = feature_pairs_option_list,
        value = ('break_angle', 'break_length'),
        description = 'Features:')

    # Make a new version of the dropdown widget 
    # Otherwise, it will control *both* plots
    pitch_type_widget2 = widgets.Dropdown(
        options= zip(list(df.pitch_name.unique()), list(df.pitch_type.unique())),
        value='SL',
        description='Pitch type:',
    )

    iplot2 = widgets.interactive(plot_pitches2, p_type=pitch_type_widget2, f_pair=feature_pair_widget)

    # Create other page elements

    db_title = HTML("<h1>MLB Pitching: 2015 to 2025</h1>")
    iplot1_title = HTML("<h2>Pitch Types by Handedness</h2>")
    iplot2_title = HTML("<h2>Pitch Types by Feature Pair</h2>")

    # Display page elements
    display(db_title)
    display(iplot1_title)
    display(iplot1)
    display(HTML("<hr/>"))
    display(iplot2_title)
    display(iplot2)
    return


if __name__ == "__main__":
    app.run()
