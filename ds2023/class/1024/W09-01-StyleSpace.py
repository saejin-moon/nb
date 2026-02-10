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
    # W09: Exploring Style Space by Replacing Data Points with Pictures
    """)
    return


@app.cell
def _():
    # For data
    import pandas as pd
    import numpy as np

    # For images
    from matplotlib.image import PIL # The Python Image Libary is used by Matplotlib
    import colorsys # This built in library will allow use to extract hue, color, and brightness

    # For plotting
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg # These are image processing functions in Matplotlib
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox # This will all us to put images on our plots
    import seaborn as sns

    # For easy file access
    import glob
    return AnnotationBbox, OffsetImage, PIL, colorsys, glob, np, pd, plt, sns


@app.cell
def _(sns):
    sns.set_theme(context="notebook", style="dark")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example
    """)
    return


@app.cell
def _():
    artist = 'balla'
    # artist = 'okeeffe'
    return (artist,)


@app.cell
def _(artist, glob):
    image_files = glob.glob(f"{artist}/*.jpg")
    return (image_files,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Look at a single image
    """)
    return


@app.cell
def _(image_files):
    img_file = image_files[13]
    img_file
    return (img_file,)


@app.cell
def _(PIL, img_file):
    img = PIL.Image.open(img_file)
    img
    return (img,)


@app.cell
def _(img):
    img.thumbnail((256, 256))
    return


@app.cell
def _(img):
    img
    return


@app.cell
def _(img, plt):
    plt.imshow(img)
    plt.show()
    return


@app.cell
def _(img):
    img_data = img.getdata()
    return (img_data,)


@app.cell
def _(img_data):
    list(img_data)[:10]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get the mean of the RGB values.
    """)
    return


@app.cell
def _(img_data, pd):
    pd.DataFrame(list(img_data)).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define a function to extract mean R, G, and B values from an image.
    """)
    return


@app.cell
def _(pd):
    def extract_mean_colors(my_img):
        h = my_img.entropy()
        my_rgb = pd.DataFrame(list(my_img.getdata()), columns=['r','g','b']).mean().round()
        my_rgb['entropy'] = h
        return my_rgb
    return (extract_mean_colors,)


@app.cell
def _(extract_mean_colors, img):
    extract_mean_colors(img)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define a function to extract mean H, S, and V values from an image.
    """)
    return


@app.cell
def _(colorsys, np, pd):
    def extract_mean_hsv(my_img):
        pixels = my_img.getdata()
        hsv = [colorsys.rgb_to_hsv(*(np.array(pxl)/255.)) for pxl in pixels]
        my_hsv = pd.DataFrame(hsv, columns=['h','s','v']).mean()
        return my_hsv
    return (extract_mean_hsv,)


@app.cell
def _(colorsys, np, pd):
    def extract_std_hsv(my_img):
        pixels = my_img.getdata()
        hsv = [colorsys.rgb_to_hsv(*(np.array(pxl)/255.)) for pxl in pixels]
        my_hsv = pd.DataFrame(hsv, columns=['h_std','s_std','v_std']).std()
        return my_hsv
    return (extract_std_hsv,)


@app.cell
def _(extract_mean_hsv, img):
    extract_mean_hsv(img)
    return


@app.cell
def _(extract_std_hsv, img):
    extract_std_hsv(img)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Put all of the images in a data frame
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Put images file names in a data frame.
    """)
    return


@app.cell
def _(PIL, image_files, pd):
    # Create a data frame of images from the file names
    IMAGES = pd.DataFrame({"file_name":image_files})

    # Creates images from files (so we don't have to keep creating them)
    IMAGES['img'] = IMAGES.file_name.apply(PIL.Image.open)

    # Reduce size for efficiency
    IMAGES.img.apply(lambda x: x.thumbnail((256,256)))

    # Extract artist names from file names
    IMAGES['artist'] = IMAGES.file_name.str.split("/").str[2]

    # Extract years from file names
    IMAGES['year'] = IMAGES.file_name.str.extract(r"(\d{4})") #.astype(int)
    IMAGES = IMAGES.dropna(subset=['year'])
    IMAGES.year = IMAGES.year.astype(int)
    return (IMAGES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Extract mean HSV and RGB for all images.
    """)
    return


@app.cell
def _(IMAGES, extract_mean_colors, extract_mean_hsv, extract_std_hsv):
    IMAGES[['h', 's', 'v']] = IMAGES.img.apply(extract_mean_hsv)
    IMAGES[['h_std', 's_std', 'v_std']] = IMAGES.img.apply(extract_std_hsv)
    IMAGES['hsv_sum'] = IMAGES[['h','s','v']].sum(1)
    IMAGES[['r', 'g', 'b', 'entropy']] = IMAGES.img.apply(extract_mean_colors)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get a few more features for visualization.
    """)
    return


@app.cell
def _(IMAGES):
    IMAGES['rgb_sum'] = IMAGES[['r', 'g', 'b']].sum(1)
    IMAGES['rgb_norm'] = IMAGES.apply(lambda x: [x.r/255., x.g/255., x.b/255.], axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Make sure all images of year data.
    """)
    return


@app.cell
def _(IMAGES):
    int(IMAGES.year.isna().sum() / len(IMAGES) * 100)
    return


@app.cell
def _(IMAGES):
    IMAGES.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Show all the images.
    """)
    return


@app.cell
def _(IMAGES, plt):
    for _idx, _row in IMAGES.iterrows():
        plt.imshow(_row.img)
        plt.title(str(_idx) + ': ' + _row.file_name)
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try out some visualizations.
    """)
    return


@app.cell
def _(IMAGES, plt):
    rgb_sorted_index = IMAGES.rgb_sum.sort_values().index
    IMAGES.loc[rgb_sorted_index, ['r','g','b']]\
            .plot.bar(stacked=True, rot=0, color=['red', 'green', 'blue'], legend=False, figsize=(len(IMAGES)/3, 5))
    plt.show()
    return


@app.cell
def _(IMAGES, plt):
    hsv_sorted_index = IMAGES.hsv_sum.sort_values().index
    IMAGES.loc[hsv_sorted_index, ['h','s','v']]\
        .plot.bar(stacked=True, rot=0, legend=True, figsize=(len(IMAGES)/3, 5))
    plt.show()
    return


@app.cell
def _(IMAGES, plt):
    IMAGES.plot.scatter(x='h', y='s', c=IMAGES.entropy, s=64)
    plt.show()
    return


@app.cell
def _(IMAGES, plt):
    IMAGES.plot.scatter(x='h', y='v', c=IMAGES.entropy, s=64)
    plt.show()
    return


@app.cell
def _(IMAGES, plt):
    IMAGES.plot.scatter(x='s', y='v', c=IMAGES.entropy, s=64)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, visualize scatter plots with pictures as data points.
    """)
    return


@app.cell
def _(AnnotationBbox, IMAGES, OffsetImage, np, plt):
    # Choose two color features
    c1 = 'h'
    c2 = 's'
    fig, ax = plt.subplots(figsize=(10, 10))
    # Create a new figure and axes
    ax.scatter(IMAGES[c1], IMAGES[c2], alpha=0)
    for _idx, _row in IMAGES.iterrows():
    # Put a scatter plot on the axes
    # Use invisible points (alpha = 0) since we are going to use images 
        my_img = _row.img
        my_img_array = np.array(my_img) / 255.0
    # Note, we could have used Pandas to create our axes object
    # ax = IMAGES.plot.scatter(x=c1, y=c2, figsize=(20, 20), alpha=0)
        imagebox = OffsetImage(my_img_array, zoom=0.2)
    # Iterate through the data to assign images to each point
        x = _row[c1]
        y = _row[c2]
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)  # Create an image using PIL
        ax.add_artist(ab)
    plt.xlabel(c1)
    plt.ylabel(c2)  # Convert to array and normalize the values
    plt.savefig('my_plot.svg', format='svg')
    plt.show()  # Create an image box and adjust zoom as needed  # This wraps the image data and prepares it for matplotlib's artist system  # This also specifies the size of the image to plot  # Put the image in an "AnnotationBbox"  # This specifies *where* to put the image on the plot  # Add the annotationbbox to the plot  # An artist is any object that gets drawn on a figure
    return


@app.cell
def _(AnnotationBbox, OffsetImage, np, plt):
    def imageplot(X, c1='r', c2='g', figsize=(10, 10), zoom=0.2, rot=0):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(X[c1], X[c2], alpha=0)
        for _idx, _row in X.iterrows():  # Invisible points, just for axes
            my_img = _row.img
            my_img_array = np.array(my_img) / 255.0
            imagebox = OffsetImage(my_img_array, zoom=zoom)
            x = _row[c1]  # Convert to array and normalize
            y = _row[c2]  # Adjust zoom as needed
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0)
            ax.add_artist(ab)
        plt.xticks(rotation=rot)
        plt.xlabel(c1)
        plt.ylabel(c2)
        plt.show()
    return (imageplot,)


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES.sort_values('year'), 'year', 'h', figsize=(20,10), zoom=.25)
    return


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES.sort_values('year'), 'year', 's', figsize=(20,10))
    return


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES.sort_values('year'), 'year', 'v', figsize=(20, 10))
    return


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES, 'h', 's')
    return


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES, 's', 'v')
    return


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES, 'h', 'v')
    return


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES, 'h', 'h_std')
    return


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES, 's', 's_std')
    return


@app.cell
def _(IMAGES, imageplot):
    imageplot(IMAGES, 'v', 'v_std', figsize=(20,20), zoom=.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
