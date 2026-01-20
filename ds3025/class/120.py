import marimo

__generated_with = "0.19.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    when multiplying two matrices together, the first columns must match the number of rows. </br>
    so what you get is a 2x3 * 3x1 = 2x[3 * 3]x1 = 2x1 matrix.

    \[
    \begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23}
    \end{bmatrix} \cdot \begin{bmatrix}
    x_1 \\
    x_2 \\
    x_3
    \end{bmatrix} = \begin{bmatrix}
    \langle a_{1\cdot}, \vec{x} \rangle \\
    \langle a_{2\cdot}, \vec{x} \rangle
    \end{bmatrix} = \begin{bmatrix}
    a_{11}x_1 + a_{12}x_2 + a_{13}x_3 \\
    a_{21}x_1 + a_{22}x_2 + a_{23}x_3
    \end{bmatrix} = \begin{bmatrix}
    x_1a_{11} \\
    x_1a_{21}
    \end{bmatrix} + \begin{bmatrix}
    x_2a_{12} \\
    x_2a_{22}
    \end{bmatrix} + \begin{bmatrix}
    x_3a_{13} \\
    x_3a_{23}
    \end{bmatrix} = x_1\begin{bmatrix}
    a_{11} \\
    a_{21}
    \end{bmatrix} + x_2\begin{bmatrix}
    a_{12} \\
    a_{22}
    \end{bmatrix} + x_3\begin{bmatrix}
    a_{13} \\
    a_{23}
    \end{bmatrix}
    \]

    and this is a 2x3 * 3x2 matrices which becomes a 2x2 matrix.

    \[
    \begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23}
    \end{bmatrix}
    \begin{bmatrix}
    b_{11} & b_{12} \\
    b_{21} & b_{32} \\
    b_{31} & b_{32} \\
    \end{bmatrix} =
    \begin{bmatrix}
    \langle a_{1\cdot},b_{\cdot1} \rangle & \langle a_{1\cdot},b_{\cdot2} \rangle \\
    \langle a_{2\cdot},b_{\cdot1} \rangle & \langle a_{2\cdot},b_{\cdot2} \rangle
    \end{bmatrix}
    \]

    set D is open if it **does not** contain its boundary. </br>
    set D is closed if it **does** contain its boundary. </br>
    **open ball** of radius $r$ about $x$ is defined below. </br>
    (the colon below means such that.)

    \[B_r(\vec{x})=\{\vec{y}\in\mathbb{R}^d : \|\vec{y}-\vec{x}\|<r\}\]

    we also went over continuous functions and the definition of continuous using limits as defined below.

    \[\lim_{x \to a} f(x) = f(a)\]

    as well as the definition of a derivative which i will not bother defining.
    """)
    return


if __name__ == "__main__":
    app.run()
