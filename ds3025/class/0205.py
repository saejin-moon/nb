import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo


@app.cell
def _():
    mo.md(r"""
    \[
    y \approx \hat{y} \\
    \hat{y} = \beta_0 + \beta_1x \to \sum_{i=1}^{n} |y_i - \hat{y}_i|^2 = \|\vec{y} - \hat{y}\|^2 \\
    \begin{bmatrix}
    y_1 - \hat{y}_1 \\
    y_2 - \hat{y}_2 \\
    \vdots \\
    y_n - \hat{y}_n
    \end{bmatrix} = Y - Ab
    \]
    """)
    return


if __name__ == "__main__":
    app.run()
