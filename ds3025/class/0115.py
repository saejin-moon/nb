import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # ds3025 | jan 15
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## distance formula
    """)
    return


@app.cell
def _(np):
    def _():
        from numpy import linalg as LA
        u = np.array([1., 2., 3., 4.])

        # by hand
        print(np.sqrt(np.sum(u ** 2)))
        # func
        return print(LA.norm(u))


    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## dot product
    """)
    return


@app.cell
def _():
    import numpy as np
    u = np.array([1., 2., 3., 4.])
    v = np.array([5., 4., 3., 2.])

    # by hand
    print(np.sum(u * v))
    # func
    print(np.dot(u, v))
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    inner product and dot product are the same.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## calculating norms
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    d = \text{the difference of} \\
    d(\vec{u}, \vec{v}) = ||\vec{v} - \vec{u}|| = \sqrt{(v_1 - u_1)^2 + (v_2 - u_2)^2} \text{ OR } \sqrt{\sum_{i=1}^{4} (v_i-u_i)^2} \\
    ||\vec{x}||_p = (\sum_{i=1}^{d} |x_i|^p)^{1/p}
    $$
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## matrices
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### basic notation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $$
    % matrix as a whole
    A = \begin{bmatrix}
    2 & 5 \\
    3 & 6 \\
    1 & 1
    \end{bmatrix} \text{is a} \space 2x3 \space \text{matrix} \\
    % individual items
    A_{12} = 5 \text{ and } A_{21} = 3 \\
    % matrix of a row/column
    A_{2,.} = \begin{bmatrix}
    3 & 6
    \end{bmatrix} \text{ and } A_{.,2} = \begin{bmatrix}
    5 \\
    6 \\
    1
    \end{bmatrix}
    $$
    """)
    return


if __name__ == "__main__":
    app.run()
