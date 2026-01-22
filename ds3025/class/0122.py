import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # taylor's theorem
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
    f(b) = f(a) + f'(a)(b - a) + \frac{f''(a)}{2}(b - a)^2 + R(b,a) \space \text{where} \space R \space \text{is the remainder.} \\
    f(b) = \frac{f^{k}(a)}{k!}(b-a)^k + R \\
    f(b) = \sum_{k=0}^{\infty} \frac{f^{k}(a)}{k!}(b-a)^k
    \]
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
