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
    # Title: The Silent Gap
    ![infographic](https://github.com/saejin-moon/saejin-moon/blob/main/infographic.png?raw=true)
    ## Description
    This infographic displays the divide between tech companies investing in mental health resources for their employees and those employees using the mental health resources they need. The infographic also offers an explanation for this divide by depicting the gap between employees who talk to their coworkers versus their employers about their mental health.
    ## Manifest
    | Name | Description | Link |
    |------|-------------|------|
    | OSMH Survey Data | Data source for the project | [Link](https://data.mendeley.com/datasets/mmnzx4w8cg/1/files/bb86b80d-a979-45fe-9ece-08d42665878b)
    | OSMH Survey Notebook | Jupyter Notebook with preliminary visualizations from the team that gathered the data. | [Link](https://data.mendeley.com/datasets/mmnzx4w8cg/1/files/60f65ee5-53fd-4ba8-b658-a745857c2805)
    | Mental Health Infographic | An infographic on mental health that I used to derive majority of the color scheme of my infographic. | [Link](https://mir-s3-cdn-cf.behance.net/project_modules/disp/0f7af321106259.562fb93084673.jpg) |
    | Wired Infographic | A random infographic I found online that I selected the yellow shade from to use in my infographic. | [Link](https://media.wired.com/photos/5932fec1f682204f73698129/3:2/w_2240,c_limit/qq_language_f.jpg) |
    ## Link
    [Google Drive Link to All Files](https://drive.google.com/drive/folders/1NmOtywmDhD8cNLhO1zBese6vfakbYR-z?usp=sharing)
    """)
    return


if __name__ == "__main__":
    app.run()
