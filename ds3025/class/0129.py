import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo


@app.cell
def _():
    mo.md(r"""
    1. generate clusters randomly
    2. clusters $\rightarrow$ new centroids
    3. centroids $\rightarrow$ new clusters
    4. new clusters $\rightarrow$ new centroids
    5. new centroids $\rightarrow$ new new clusters
    """)
    return


@app.cell
def _():
    mo.md(r"""
    \[
    \sum_{i=1}^{k} \sum_{j \in C_i} \|x_j-\mu_i\|^2 \\
    \text{for example: } \sum_{x_i \in C_1} \|x_i - \mu_1 \|^2 + \sum_{x_i \in C_2} \|x_i - \mu_2 \|^2 + \sum_{x_i \in C_3} \|x_i - \mu_3 \|^2 \space \text{where} \space C_1 = \{1,4,6,8\}, C_2 = \{2,3,7\}, C_3 = \{5\}, \mu_1 = \begin{pmatrix}
    -2 \\
    1
    \end{pmatrix}, \mu_2 = \begin{pmatrix}
    2 \\
    -1
    \end{pmatrix}, \mu_3 = \begin{pmatrix}
    10 \\
    -10
    \end{pmatrix} \\
    \sum_{1} = \|x_1 - \mu_1\|^2 + \|x_4 - \mu_1\|^2 + \|x_6 - \mu_1\|^2 + \|x_8 - \mu_1\|^2 \\
    \sum_{2} = \|x_2 - \mu_2\|^2 + \|x_3 - \mu_2\|^2 + \|x_7 - \mu_2\|^2 \\
    \sum_{3} = \|x_5 - \mu_3\|^2 \\
    \|x_1 - \mu_1\|^2 = (x_{11} - \mu_{11})^2 + (x_{12} - \mu_{12})^2 \\
    \begin{bmatrix}
    x_{11} - \mu_{11} & x_{12} - \mu_{12} \\
    x_{21} - \mu_{21} & x_{22} - \mu_{22} \\
    x_{31} - \mu_{21} & x_{32} - \mu_{22} \\
    x_{41} - \mu_{11} & x_{42} - \mu_{12} \\
    x_{51} - \mu_{31} & x_{52} - \mu_{32} \\
    x_{61} - \mu_{11} & x_{62} - \mu_{12} \\
    x_{71} - \mu_{21} & x_{72} - \mu_{22} \\
    x_{81} - \mu_{11} & x_{82} - \mu_{12}
    \end{bmatrix} = \begin{bmatrix}
    x_{11} & x_{12} \\
    x_{21} & x_{22} \\
    x_{31} & x_{32} \\
    x_{41} & x_{42} \\
    x_{51} & x_{52} \\
    x_{61} & x_{62} \\
    x_{71} & x_{72} \\
    x_{81} & x_{82}
    \end{bmatrix} - \begin{bmatrix}
    \mu_{11} & \mu_{12} \\
    \mu_{21} & \mu_{22} \\
    \mu_{21} & \mu_{22} \\
    \mu_{11} & \mu_{12} \\
    \mu_{31} & \mu_{32} \\
    \mu_{11} & \mu_{12} \\
    \mu_{21} & \mu_{22} \\
    \mu_{11} & \mu_{12} \\
    \end{bmatrix} = \begin{bmatrix}
    \vec{x}_1 \\
    \vec{x}_2 \\
    \vec{x}_3 \\
    \vec{x}_4 \\
    \vec{x}_5 \\
    \vec{x}_6 \\
    \vec{x}_7 \\
    \vec{x}_8
    \end{bmatrix} - \begin{bmatrix}
    \vec{\mu}_1 \\
    \vec{\mu}_2 \\
    \vec{\mu}_2 \\
    \vec{\mu}_1 \\
    \vec{\mu}_3 \\
    \vec{\mu}_1 \\
    \vec{\mu}_2 \\
    \vec{\mu}_1
    \end{bmatrix} \\
    \begin{bmatrix}
    \vec{\mu}_1 \\
    \vec{\mu}_2 \\
    \vec{\mu}_2 \\
    \vec{\mu}_1 \\
    \vec{\mu}_3 \\
    \vec{\mu}_1 \\
    \vec{\mu}_2 \\
    \vec{\mu}_1
    \end{bmatrix} = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 1 & 0 \\
    1 & 0 & 0 \\
    0 & 0 & 1 \\
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    1 & 0 & 0
    \end{bmatrix}\begin{bmatrix}
    \mu_{11} & \mu_{12} \\
    \mu_{21} & \mu_{22} \\
    \mu_{31} & \mu_{32}
    \end{bmatrix} = ZU \\
    \begin{bmatrix}
    \vec{x}_1 \\
    \vec{x}_2 \\
    \vec{x}_3 \\
    \vec{x}_4 \\
    \vec{x}_5 \\
    \vec{x}_6 \\
    \vec{x}_7 \\
    \vec{x}_8
    \end{bmatrix} = X \\
    \sum_{i=1}^{k} \sum_{j \in C_i} \|x_j-\mu_i\|^2 = \|X - ZU\|^{2}_{F} \space \text{where} \space F \space \text{is the frobenius norm.} \\
    \mathcal{G}(C_1, \dots, C_k) = \min_{\mu_1, \dots, \mu_k \in \mathbb{R}^d} \sum_{i=1}^{k} \sum_{j \in C_i} \|x_j-\mu_i\|^2\\
    P(Z>\beta) < \frac{E(Z)}{\beta}, Z \ge 0, P(Z>B) = 1 * P(\frac{Z}{\beta} > 1) \le I_A\frac{Z}{\beta}P(\frac{Z}{\beta}>1) + \frac{Z}{\beta}P(\frac{Z}{\beta}<1) = \frac{1}{\beta}\{ZP(\frac{Z}{\beta} > 1) + ZP(\frac{Z}{\beta}<1)\} \\
    I_{A^c}Z = \begin{cases}
    0 & x \in A \\
    Z & x \in A_c
    \end{cases}, I_AZ = \begin{cases}
    Z & x \in A \\
    0 & x \in A_c
    \end{cases}, I_{A^c}Z + I_AZ = Z \\
    \text{chebyshev} = P(|X - E(Z)| > \alpha) \le \frac{\text{Var}(X)}{\alpha^2} \\
    P(\underbrace{|X - E(x)|^2}_{Z} > \underbrace{\alpha^2}_{\beta}) = \frac{E(|X - E(X)|^2)}{\alpha^2}
    \]
    """)
    return


if __name__ == "__main__":
    app.run()
