import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo


@app.cell
def _():
    mo.md(r"""
    # clustering
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## markov

    \[
    P(x>\beta) < \frac{E(x)}{\beta} \space \text{for} \space x \ge 0 \space \text{and} \space \beta > 0 \\
    E(x) = \int_{0}^{+\infty} xf(x)dx = \int_{0}^{\beta} xf(x)dx + \int_{\beta}^{+\infty} xf(x)dx > \int_{\beta}^{+\infty} xf(x)dx > \int_{\beta}^{+\infty} \beta f(x)dx = \beta\int_{\beta}^{+\infty} f(x)dx  = \beta P(x \ge \beta) \\
    E(x) > \beta P(x \ge \beta) \\
    \beta P(x \ge \beta) < E(x) \\
    P(x \ge \beta) < \frac{E(x)}{\beta} \\
    P(x > \beta) < \frac{E(x)}{\beta} \space \text{b/c the integral at one specific point is 0 so the} \space \ge \space \text{is unnecessary and we can reduce it to} \space > \\
    \]
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## chebyshev

    \[
    P(|x - E(x)| > \alpha) \le \frac{\text{Var}(x)}{\alpha^2} \\
    \]

    $B$ is the set of all points (vectors) in $\mathbb{R}^d$ such that their norm is less than or equal to a half.

    \[
    \mathbb{R}^d; B = \{\vec{x} \in \mathbb{R}^d : \|x\|_2 \le \frac{1}{2} \}
    \]

    \[
    C = [-\frac{1}{2}, \frac{1}{2}]^d
    \]

    in two-dimensional space, $B$ is the cirle with radius $\frac{1}{2}$ and $C$ is the box with radius $\frac{1}{2}$

    \[
    x \sim U[C] = \space \text{x is distributed uniformly throughout} \space C
    \]

    \[
    P(x \in B) \to 0 \space \text{as} \space d \to \infty \\
    x = \|x\|^2 \space \text{since the latter half of the eq. is still a rand. var.} \\
    P(x \in B) = P(\|x\|_2 < \frac{1}{2}) = P(\|x\|^2 < \frac{1}{4}) = P(\|x\|^2 - \mu < -\alpha) = P(|\|x\|^2 - \mu| > \alpha) \le \frac{\text{Var}(\|x\|^2)}{\alpha^2} \Rightarrow P(\|x\|^2 - \tilde{\mu} < -\alpha) =  P(\|x\| < \tilde{\mu} - \alpha) \\
    \tilde{\mu} - \alpha = \frac{1}{4} \to \alpha = \tilde{\mu} - \frac{1}{4} = d\mu - \frac{1}{4} > 0 \\
    P(\|x\|^2 < \frac{1}{4}) \le \frac{d\sigma^2}{(d\mu - \frac{1}{4})^2} \sim \lim_{d \to \infty} \frac{d\sigma^2}{(d\mu - \frac{1}{4})^2} \to \frac{1}{d} \to 0 \\
    \tilde{\mu} = E(\|x\|^2) = E(\sum_{i=1}^{d} x_i^2) = \sum_{i=1}^d E(x_i^2) \space \text{where we recall C and assert that} \space x_i \sim U[-\frac{1}{2}, \frac{1}{2}]; \mu = E(x_i^2) \therefore \sum_{i=1}^{d} \mu = d\mu \\
    \tilde{\sigma}^2 = \text{Var}(\|x\|^2) = \text{Var}(\sum_{i=1}^{d} x_i^2) = \sum_{i=1}^d \text{Var}(x_i^2) = \sum_{i=1}^{d} \sigma^2 = d\sigma^2
    \]
    """)
    return


@app.cell
def _():
    mo.md(r"""
    # vector space
    $V$ is a set of elements called vectors with the following properties:
    0. $\vec{x}, \vec{y}, \vec{z} \in V$
    1. $\vec{x} + \vec{y} \in V$
    2. $\alpha \in \mathbb{R}; \alpha \vec{x} \in V$
    3. $\vec{x} + (\vec{y} + \vec{z}) = (\vec{x} + \vec{y}) + \vec{z}$
    4. $\vec{x} + \vec{y} = \vec{y} + \vec{x}$
    5. $\exists \vec{0} \in V : \vec{0} + \vec{x} = \vec{x}$
    6. $\exists -\vec{x} : \vec{x} + (-\vec{x}) = \vec{0}$
    7. $(\alpha + \beta) \vec{x} = \alpha \vec{x} + \beta \vec{x}$

    ## subspace
    $\{\vec{v_1}, \vec{v_2}, \dots, \vec{v_k}\}$ if we include all possible linear combination of vectors then $\alpha_1\vec{v_1} + \alpha_2\vec{v_2} + \dots + \alpha_k\vec{v_k}$ (which is the span of the subspace)
    """)
    return


if __name__ == "__main__":
    app.run()
