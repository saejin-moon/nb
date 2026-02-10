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
    # W W13 M Activity: Challenge

    DS 2023 | Communicating with Data

    **Instructions**

    Using the provided data set `pitching-sample.csv`, create a simple dashboard using IPyWidgets, Matplotlib, and Seaborn that matches the following screenshot:

    ![](example.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Solution
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns

    import ipywidgets as widgets
    return pd, plt, sns, widgets


@app.cell
def _(pd):
    df = pd.read_csv("./pitching-sample.csv").set_index(['ab_id','pitch_num'])
    return (df,)


@app.cell
def _(df, display, plt, sns, widgets):
    PITCH_TYPE_MAP = {
        'CH': 'Changeup',
        'CU': 'Curveball',
        'EP': 'Eephus',
        'FC': 'Cutter',
        'FF': 'Four-seam Fastball',
        'FO': 'Pitchout (also PO)',
        'FS': 'Splitter',
        'FT': 'Two-seam Fastball',
        'IN': 'Intentional ball',
        'KC': 'Knuckle curve',
        'KN': 'Knuckleball',
        'PO': 'Pitchout (also FO)',
        'SC': 'Screwball',
        'SI': 'Sinker',
        'SL': 'Slider',
        'UN': 'Unknown'
    }

    REVERSE_PITCH_TYPE_MAP = {v: k for k, v in PITCH_TYPE_MAP.items()}

    p_throws_widget = widgets.RadioButtons(
        options=['Left', 'Right'],
        value='Right',
        description='Pitcher hand:',
        style={'description_width': 'initial'}
    )

    try:
        pitch_codes_in_data = set(df['pitch_type'].unique().dropna().tolist())
    
        valid_pitch_descriptions = sorted([PITCH_TYPE_MAP[code] for code in pitch_codes_in_data if code in PITCH_TYPE_MAP])

        default_description = 'Slider'
        if default_description not in valid_pitch_descriptions and valid_pitch_descriptions:
            default_description = valid_pitch_descriptions[0]
        elif not valid_pitch_descriptions:
            valid_pitch_descriptions = sorted(PITCH_TYPE_MAP.values())
            default_description = 'Slider'

    except Exception:
        valid_pitch_descriptions = sorted(PITCH_TYPE_MAP.values())
        default_description = 'Slider'
    
    pitch_type_widget = widgets.Dropdown(
        options=valid_pitch_descriptions,
        value=default_description,
        description='Pitch type:',
        style={'description_width': 'initial'}
    )

    def plot_dashboard(p_throws_selection, pitch_type_selection):
    
        p_throws_data_value = 'R' if p_throws_selection == 'Right' else 'L'

        pitch_code_for_filter = REVERSE_PITCH_TYPE_MAP.get(pitch_type_selection, pitch_type_selection)
    
        df_filtered = df[
            (df['p_throws'] == p_throws_data_value) &
            (df['pitch_type'] == pitch_code_for_filter)
        ].copy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        sns.scatterplot(
            data=df_filtered, 
            x='break_angle', 
            y='break_length', 
            hue='stand', 
            ax=axes[0], 
            s=40,
            legend=False
        )
        axes[0].set_title('Break Angle and Length')
        axes[0].set_xlim(-55, 35)
        axes[0].set_ylim(3, 15)   
    
        sns.scatterplot(
            data=df_filtered, 
            x='spin_dir', 
            y='spin_rate', 
            hue='stand', 
            ax=axes[1], 
            s=40,
            hue_order=['R', 'L']
        )
        axes[1].set_title('Spin Direction and Rate')
        axes[1].set_xlim(-10, 370)
        axes[1].set_ylim(-100, 2800)
    
        legend = axes[1].legend_
        if legend:
            legend.set_title('stand')
            legend.set_loc('upper right')
    
        sns.despine(fig=fig)
        plt.tight_layout()
        plt.show()

    interactive_plot = widgets.interactive(
        plot_dashboard,
        p_throws_selection=p_throws_widget,
        pitch_type_selection=pitch_type_widget
    )

    display(interactive_plot)
    return


if __name__ == "__main__":
    app.run()
