import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## matrices and their inverses
    - column rank for linear independence is the same as row rank
    - two vectors are linearly independent if they are multiples of each other

    \[
    \textcolor{pink}{\text{solve linear systems} \space A\vec{x} = \vec{b}} \\
    A \space \text{is Square Matrix} \space A \in \mathbb{R}^{n \times n} \\
    A \space \text{is NOT SINGULAR if} \space rnk(A)=n \\
    n \space \text{cols are linearly independent.} \\
    \text{THEN there exists a matrix} \space A^{-1} \space \text{such that} \space A^{-1} \times A = A \times A^{-1} = I_{n \times n} = \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1
    \end{bmatrix} \\
    \text{span}(\text{col}(a)) = \mathbb{R}^n\\
    \vec{e}_1 = \begin{bmatrix}
    1 \\
    0 \\
    \vdots  \\
    0
    \end{bmatrix} \vec{e}_2 = \begin{bmatrix}
    0 \\
    1 \\
    \vdots  \\
    0
    \end{bmatrix} \vec{e}_n = \begin{bmatrix}
    0 \\
    0 \\
    \vdots  \\
    1
    \end{bmatrix} \\
    \begin{bmatrix}
    \vec{a}_1 & \vec{a}_2 & \dots & \vec{a}_n
    \end{bmatrix} \vec{b}_i = \vec{e}_i \\
    \begin{bmatrix}
    \vec{a}_1 & \vec{a}_2 & \dots & \vec{a}_n
    \end{bmatrix} \begin{bmatrix}
    \vec{b}_1 & \vec{b}_2 & \dots & \vec{b}_n
    \end{bmatrix} = \begin{bmatrix}
    \vec{e}_1 & \vec{e}_2 & \dots & \vec{e}_n
    \end{bmatrix} = I_{n \times n} \\
    I\vec{x} = A^{-1}\vec{b} \to \vec{x} = A^{-1}\vec{b}
    \]

    ### formula for 2x2 matrix

    \[
    A = \begin{pmatrix}
    a & b \\
    c & d
    \end{pmatrix} \\
    A^{-1} = \frac{1}{ad - bc} \begin{pmatrix}
    d & -b \\
    -c & a
    \end{pmatrix} \\
    \text{linear independence is true IF} \space \alpha_1 \begin{bmatrix}
    a \\
    c
    \end{bmatrix} + \alpha_2 \begin{bmatrix}
    b \\
    d
    \end{bmatrix} = 0 \to \begin{cases}
    \alpha_1a + \alpha_2b = 0 \\
    \alpha_1c + \alpha_2d = 0
    \end{cases} = \begin{cases}
    \alpha_1a = -\alpha_2b \\
    \alpha_1c = -\alpha_2d
    \end{cases} = \begin{cases}
    \alpha_1ac = -\alpha_2bc \\
    \alpha_1ac = -\alpha_2ad
    \end{cases} \to -\alpha_2bc = -\alpha_2ad \to bc = ad \to \text{we can pick any} \space \alpha_2 \therefore \space \text{not linearly independent} \\
    bc \neq ad \to \alpha_2 = 0 \\
    \text{determinant of A is} \space \text{det} \space A = ad - bc \\
    A^{-1} \space \text{EXISTS iff det} \space A \neq 0
    \]

    ## idfk

    \[
    (y, \vec{x}) \\
    \textcolor{green}{\text{minimize} \space \|Y - AB\|^2} \\
    \begin{bmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_n
    \end{bmatrix} = \begin{bmatrix}
    1 & x_1 \\
    1 & x_2 \\
    \vdots & \vdots\\
    1 & x_n
    \end{bmatrix} \begin{bmatrix}
    \beta_0 \\
    \beta_1
    \end{bmatrix} = span(\begin{bmatrix}
    1 \\
    1 \\
    \vdots \\
    1
    \end{bmatrix}\begin{bmatrix}
    x_1 \\
    x_2 \\
    \vdots \\
    x_n
    \end{bmatrix}) \\
    \underbrace{A^T}_{2 \times d}\underbrace{A}_{d \times 2}\underbrace{\vec{\beta}}_{2 \times 1} \approx \underbrace{A^T}_{2 \times d}\underbrace{Y}_{d \times 1} \\
    \underbrace{A^TA}_{2 \times 2}\vec{\beta} = \underbrace{A^TY}_{2 \times 1} \\
    A \in \mathbb{R}^{n \times m} \space | \space n \ge m \space | \space \text{rnk}(A) = m \to A^TA \space \text{is nonsingular} \\
    A \in \mathbb{R}^{d \times 2} \space | \space d \ge 2 \space | \space \text{rnk}(A) = 2 \to (A^TA)^{-1} \space \text{exists} \\
    \underbrace{\underbrace{(A^TA)^{-1}}_{2 \times 2} \underbrace{(A^TA)}_{2 \times 2}}_{I_{2 \times 2}} \vec{\beta} = (A^TA)^{-1}A^TY \to \vec{\beta} = (A^TA)^{-1}A^TY
    \]

    ## vectors smth smth

    \[
    \vec{e}_1 = \begin{bmatrix}
    1 \\
    0
    \end{bmatrix} \space \vec{e}_2 = \begin{bmatrix}
    0 \\
    1
    \end{bmatrix} \\
    \langle \vec{e}_1, \vec{e}_2 \rangle = \sum_{i=1}^{d} v_iw_i = 1 \cdot 0 + 0 \cdot 1 = 0 \space \therefore \space \vec{v} \space \text{and} \space \vec{w} \space \text{are orthogonal iff} \space \langle \vec{v}, \vec{w} \rangle = 0 \space \text{given that} \space \vec{v} \neq \vec{0}, \vec{w} \neq \vec{0} \\
    \|\vec{v} + \vec{w}\|^2 = \|\vec{v}\|^2 + 2 \langle \vec{v}, \vec{w} \rangle + \|\vec{w}\|^2 \\
    \vec{v} = \sum \alpha_i b_i \\
    \|\vec{v}\|^2 = \| \sum \alpha_i b_i \|^2 \\
    \text{if vec is orthogonal, inner products are 0}
    \]
    """)
    return


if __name__ == "__main__":
    app.run()
