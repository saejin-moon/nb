import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
