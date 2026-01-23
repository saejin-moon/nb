import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Programming Review
    Do Q1 and one other question.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q1.
    First, think about your priorities in life. What kind of salary do you want to make after graduation? Do you mind getting more schooling? What kind of work-life balance are you looking for? Where do you want to work, geographically? You don't have to write this down here, just think about it.

    1. Go to the Occupational Outlook Handbook at [https://www.bls.gov/ooh/](https://www.bls.gov/ooh/). Look up "Data Scientist." Read about the job and start collecting data about it from the job profile (e.g. salary, education required, work setting)
    2. Find 7-10 other jobs that appeal to you, and collect the same data as you did for Data Scientist. Put it all in a spreadsheet.
    3. Do any of your findings surprise you?
    4. Rank the jobs you picked from best to worst, and briefly explain why you did so.
    """)
    return


@app.cell
def _(pd):
    # (1 + 2)
    jobs_data = [
        {
            "title": "Data Scientist",
            "salary": 112590,
            "education": "Bachelor's degree",
            "growth": "34% (Much faster)",
            "experience": "None"
        },
        {
            "title": "Software Developer",
            "salary": 133080,
            "education": "Bachelor's degree",
            "growth": "15% (Much faster)",
            "experience": "None"
        },
        {
            "title": "Info Security Analyst",
            "salary": 124910,
            "education": "Bachelor's degree",
            "growth": "29% (Much faster)",
            "experience": "Less than 5 years"
        },
        {
            "title": "Actuary",
            "salary": 125770,
            "education": "Bachelor's degree",
            "growth": "22% (Much faster)",
            "experience": "None"
        },
        {
            "title": "Computer Systems Analyst",
            "salary": 103790,
            "education": "Bachelor's degree",
            "growth": "9% (Much faster)",
            "experience": "None"
        },
        {
            "title": "Database Administrator",
            "salary": 123100,
            "education": "Bachelor's degree",
            "growth": "9% (Faster than average)",
            "experience": "None"
        },
        {
            "title": "Statistician",
            "salary": 104350,
            "education": "Master's degree",
            "growth": "30% (Much faster)",
            "experience": "None"
        },
        {
            "title": "Web Developer",
            "salary": 95380,
            "education": "Bachelor's degree",
            "growth": "7% (Faster than average)",
            "experience": "None"
        }
    ]

    df_jobs = pd.DataFrame(jobs_data)

    df_jobs
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### (3)
    Despite their more universal application and my presumptions on their demand in the job market, I expected the Statistician to have a higher median income especially with the education expected from them.

    ### (4)
    #### Ranking
    1. Software Developer
    2. Information Security Analyst
    3. Data Scientist
    4. Actuary
    5. Database Administrator
    6. Computer Systems Analyst
    7. Statistician
    8. Web Developer

    **Reasoning:**
    I prioritized Salary and Growth.
    *   I ranked Software Developer the highest since it has the highest salary ($133k) with high growth as well.
    *   I ranked Info Security Analyst and Data Scientist right afterwards since both jobs have massive growth potential (>29%), which implies job security, though slightly lower pay than software developers.
    *   I ranked Statisticians lower due to the high entry cost of holding a Master's degree rather than a Bachelor's and for also holding one of the lower salaries despite the higher education requirement.
    *   I ranked Web Developer last primarily due to having the lowest salary in this specific group, and I foresee a future where this occupation is no longer necessary.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q2
    Being able to make basic plots to visualize sets of points is really helpful, particularly with data analysis. The pyplot plots are built up slowly by defining elements of the plot, and then using `plt.show()` to create the final plot. This question gives you some practice doing that **iterative** building process.
    """)
    return


@app.cell
def _(np, plt):
    # Q2 Solution: Plotting

    # (1) (imported in setup cell) I completed this assignment with Marimo so the Python setup is slightly different.

    # --- PART 1: Log and Exponential Scatter Plot ---
    # (2) didn't use 0 since it threw an error for being undefined
    x_part1 = np.linspace(0.01, 1, 50) 

    # (3)
    y_log = np.log(x_part1)
    z_exp = np.exp(x_part1)

    # (4)
    plt.figure(figsize=(8, 5))
    plt.scatter(x_part1, y_log, label='Natural Log', color='blue') # for (7)
    plt.scatter(x_part1, z_exp, label='Exponential', color='red')  # for (7)

    # (6)
    plt.xlabel('x values')
    plt.ylabel('Function values')
    plt.title('Natural Log and Exponential Functions')

    # (7)
    plt.legend(loc='lower right')

    # (5)
    plt.show()

    # --- PART 2: Sine and Cosine Line Plot ---

    # (8)
    x_part2 = np.linspace(-6.5, 6.5, 100)

    # (9)
    y_sin = np.sin(x_part2)
    y_cos = np.cos(x_part2)

    # (10 + 11)
    plt.figure(figsize=(8, 5))
    plt.plot(x_part2, y_sin, label='Sine')
    plt.plot(x_part2, y_cos, label='Cosine')

    # (12)
    plt.xlabel('x (radians)')
    plt.ylabel('Amplitude')
    plt.title('Sine and Cosine Waves')
    plt.legend(loc='lower left')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q3
    (skipped)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4
    (skipped)
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    return mo, np, pd, plt


if __name__ == "__main__":
    app.run()
