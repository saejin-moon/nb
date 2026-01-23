import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    ## taylor's theorem
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    \[
    f'(x) > 0 \implies f(x) \space \text{is increasing} \\
    f'(x) < 0 \implies f(x) \space \text{is decreasing} \\
    f(a)=f(b) \space \text{then} \space c \in (a, b) \space | \space  f'(c) = 0 \\
    f'(c) = \frac{f(b) - f(a)}{b - a} \rightarrow f(b) = f(a) + f'(c)(b - a) \\
    f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2}(x - a)^2 + R(x,a) \space \text{where} \space R \space \text{is the remainder.} \\
    f(x) = \frac{f^{k}(a)}{k!}(x-a)^k + R \\
    f(x) = \sum_{k=0}^{\infty} \frac{f^{k}(a)}{k!}(x-a)^k \\
    \text{sin}(x) \approx \sum_{n=0}^{\infty} \frac{(-1)^{n}x^{2n+1}}{(2n+1)!} \\
    R_k = f(x) - T_k(x) \space \text{where} \space T \space \text{is the Taylor expansion.} \\
    |R_k| = |f(x) - T_k(x)| \le \frac{|f^{k+1}(c)|}{(k+1)!}(x-a)^{k+1} \space \text{where c is the value of} \space x \space \text{where} \space f^{k+1} \space \text{is at its maximum value.}
    \]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## gradient
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    \[
    f = f(x,y,z) \implies \nabla f = \begin{bmatrix}
    \frac{\partial f}{\partial x} \\
    \frac{\partial f}{\partial y} \\
    \frac{\partial f}{\partial z}
    \end{bmatrix} = \langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \rangle
    \]
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
