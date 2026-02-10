import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $k$-Nearest Neighbor
    ### Foundations of Machine Learning
    ### `github.com/ds4e/knn`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction
    - Today we start with predictive analytics, which is a vast field that doubles in size every 30 days
    - I'm going to use `cars_env.csv`, but `land_mines.csv` and `diabetes.csv` are also great choices; for today, we need data with two numeric explanatory variables $(x_1, x_2)$, and a categorical outcome label, $y$, that takes at least two distinct values
    - There are a lot of new ideas to take in, since we're breaking a lot of new ground; we'll revisit everything and explore the hidden depth throughout the rest of the course
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example
    - Let's start with an example of what we're talking about: Predicting vehicle class from price and carbon footprint
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd 
    import seaborn as sns

    df = pd.read_csv('./data/cars_env.csv')
    df.head()

    # Drop extremely expensive cars:
    q90 = np.quantile( df['baseline price'],.9) # Compute the .9 quantile
    print(q90)
    keep = df['baseline price'] < q90  # Logical condition asserting price < .9 quantile
    df = df.loc[keep,:] # Use locator function to filter on a Boolean conditional
    df.describe()
    sns.kdeplot(data=df,x='baseline price')
    plt.show()

    # Simplify the vehicle classification scheme:
    df['class'] = df['EPA class']
    df['class'] = df['class'].replace(['MIDSIZE CARS','COMPACT CARS','SUBCOMPACT CARS','TWO SEATERS','LARGE CARS'],'car')
    df['class'] = df['class'].replace(['SMALL STATION WAGONS','MIDSIZE STATION WAGONS'],'station wagon')
    df['class'] = df['class'].replace(['STANDARD PICKUP TRUCKS','SMALL PICKUP TRUCKS'],'truck')
    df['class'] = df['class'].replace(['VANS','MINIVAN'],'van')
    return df, plt, sns


@app.cell
def _(df, plt, sns):
    sns.kdeplot(x=df['baseline price'], hue = df['class'], common_norm=False)
    df.loc[:,['baseline price','class']].groupby('class').describe() # Baseline price by simplified vehicle class
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    sns.kdeplot(x=df['footprint'], hue = df['class'], common_norm=False)
    df.loc[:,['footprint','class']].groupby('class').describe() # Footprint by simplified vehicle class
    plt.show()
    return


@app.cell
def _(df, plt, sns):
    # Plot footprint against price
    this_plot = sns.scatterplot(data=df,x='footprint',y='baseline price',
                                hue='class',
                                style='class')
    sns.move_legend(this_plot, "upper left", bbox_to_anchor=(1, 1)) # Move legend off the plot canvas
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supervised Learning
    - There is a target/outcome variable $y$, which we are trying to predict using features/covariates $X$
    - All of machine learning works on a fairly simple premise: "If $X$ and $X'$ are similar, then $y$ and $y'$ are probably similar."
    - A **(point) predictive model** is the mapping from features/covariates $\hat{x}$ to target/outcome $\hat{y}$, so $\hat{y} = m(\hat{x})$
    - We often predict other things, like the distribution of $\hat{y}$ conditional on $\hat{x}$, which we'll talk about
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification versus Regression
    - If the target/outcome variable $y$ is...
        - Categorical: Then we are doing classification
        - Numeric: Then we are doing regression
    - Our focus today is on a supervised learning algorithm for classification, called **$k$ nearest neighbor**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outline
    1. $k$-NN Classification
    2. Selecting $k$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. $k$-NN Classification
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preliminaries: Distance
    - In one dimensional, distance means absolute value: $d(a,b) = |a-b| = \sqrt{ (a-b)^2}$
    - In many dimensions, distances usually means **Euclidean distance**. Imagine we have lists of variables for two observations, $a = [a_1, ..., a_L]$ and $b  = [b_1, ..., b_L]$. Then the distance between $a$ and $b$ is defined as:

    \[
    d(a,b) = ||a-b|| = \sqrt{ (a_1 - b_1)^2 + (a_2 - b_2)^2 + ... + (a_n - b_L)^2}
    \]

    - What's the distance between $(1,3)$ and $(4,-2)$?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preliminaries: Scaling
    - Since units are typically non-comparable across variables (e.g. age and height), we need to **normalize** or **scale** the control/feature variables before we chuck them into a model
    - This is a very common transformation step, and many people do it automatically
    - To **Min-Max Scale** a variable $x$, we compute, for each observation $i$,
    $$
    u_i = \dfrac{x_i - \min(x)}{\max(x)-\min(x)}
    $$
    - This transforms all the values of $x$ so that that they lie between 0 and 1, with the smallest value mapped to zero and the largest value mapped to 1
    - If you do not do this (or if the variables aren't already similarly scaled), neighbor methods simple won't work correctly
    - It's easier to transform the data and keep the original, untransformed version to compare with predictions later
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preliminaries: The `df.apply(fcn)` Method
    - If we want to use `fcn` on every column of `df` without looping over it, we can use the `.apply(fcn)` method from Pandas
    - So we either `from sklearn.preprocessing import MinMaxScaler` or
    ```python
    def MinMaxScaler(x):
        u = (x-min(x))/(max(x)-min(x))
        return u

    df = df.apply(MinMaxScaler)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $k$-NN Classification
    - Imagine we have a categorical variable $y$ of interest, that we want to predict based on features/covariates $X$: Disease based on patient symptoms, corporate or sovereign rating based on financial indicators, genus or species based on characteristics or DNA
    - This is a classical **classification** problem: Given variables $X$, what is the probability we would assign to each possible label $\ell$ for the variable $y$?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## $k$ Nearest Neighbor Classification
    - Imagine we have covariates/features $X = [x_1, x_2, ..., x_N]$, consisting of $N$ observations of $L$ variables, and observed outcomes/target variable $Y$
    - Consider a new case $\hat{x} = (\hat{x}_1,...,\hat{x}_L)$. We want to make a guess of what **class/label** it will likely take, $\hat{\ell}$
    - The *$k$ Nearest Neighbor Classification Algorithm* is:
      1. Compute the distance from $\hat{x}$ to each observation $x_i$ in the dataset
      2. Find the $k$ "nearest neighbors" $x_1^*$, $x_2^*$, ..., $x_k^*$ to $\hat{x}$ in the data in terms of distance, with outcome labels $\ell_{1}^*$, $\ell_2^*$, ..., $\ell_k^*$
      3. Return either a
            - **Hard Classification**: The modal/most common label among the nearest neighbor labels
            - **Soft Classification**: The proportions for which each of the labels occur among the nearest neighbors
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Getting Started
    - We need a categorical outcome $Y$ and (for now) two numeric covariates/features $(X_1, X_2)$
    - Make a Seaborn plot of the labels as a function of the covariates/features and their labels (Hint: Use the `markers=Y` argument)
    - How would kNN roughly work, looking at this picture?
    - Check in with your data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SciKit-Learn
    - Unless we are doing something tailored to a particular task, we generally don't want to code our own algorithms: It is time consuming, and existing implementations are typically more efficient or robust than what we would create (but there is a lot of value in coding your own algorithms and estimators)
    - The most popular Python machine learning library is called **SciKit-Learn**
    - You typically import it as `from sklearn.<model class> import <model name>`, where `<model class>` is the set of related models and `<model name>` is the desired algorithm
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scikit-Learn
    - To get the `from sklearn.neighbors import KNeighborsClassifier` for classification
    - The workflow with `sk` is that you use it to
      1. Create an untrained model object with a fixed $k$: `model = KNeighborsRegressor(n_neighbors=k)` or `model = KNeighborsClassifier(n_neighbors=k)`
      2. Fit that object to the data, $(X,y)$: `fitted_model = model.fit(X,y)`
      3. Use the fitted object to make predictions for new cases $\hat{x}$: `y_hat = fitted_model.predict(x_hat)` for hard classification and `y_hat = fitted_model.predict_proba(x_hat)` for soft classification
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Preamble and Read data

    Here's a template for fitting a $k$-NN classifier:

    ```python
    # Preamble:
    import numpy as np
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier

    def minmax(x):
        u = (x-min(x))/(max(x)-min(x))
        return u

    # Wrangle data:
    df = pd.read_csv(file.csv) # load your data
    df.head()
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Select Data and Normalize

    ```python
    # Select data:
    y = df['target'] # Set out outcome/target
    ctrl_list = [ var_1, var_2, ] # List of control variables
    x = df.loc[:, ctrl_list] # Set our covariates/features
    u = x.apply(MinMaxScaler) # Scale our variables
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Fit a kNN Classifier and Predict
    ```python
    # Set the number of neighbors, typically odd to "break ties":
    k = 5
    # Create a fitted model instance:
    model = KNeighborsClassifier(n_neighbors = k) # Create a model instance
    model = model.fit(u,y) # Fit the model
    # Make predictions:
    y_hat = model.predict(u) # Hard prediction
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing the Predictor
    - Sometimes, it is nice to try to visualize how the algorithm works
    ```python
    this_plot = sns.scatterplot(x=x1,y=x2,
                    hue = y,
                    style=y_hat)
    sns.move_legend(this_plot, "upper left", bbox_to_anchor=(1, 1))
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confusion Matrices
    - Did it... work?
    - Evaluating fit and understanding whether we're making good or bad predictions is a core skill in machine learning
    - For classification tasks, our most basic metric of "Did it work?" is a confusion matrix: Cross-tabulate the true labels with the predicted ones, and see if they align or not

    ```python
    ## Confusion Matrix:
    pd.crosstab(y,y_hat)

    from sklearn.metrics import confusion_matrix
    confusion_matrix(y, y_hat)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy
    - In many scenarios, we want to simply the confusion matrix into a single, summary number
    - Many, many ways of doing this have been proposed
    - The simplest and most obvious is, "What proportion of the cases did we predict correctly?" This is called **accuracy**
    - We sum up the number of cases on the "descending diagonal" of the confusion matrix, and divide by the total number of observations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Accuracy
    - Numpy:
    ```python
    np.sum( np.diag(pd.crosstab(y,y_hat)))/len(y)
    ```
    - From scikit:
    ```python
    from sklearn.metrics import accuracy_score
    accuracy_score(y,y_hat)
    ```
    or
    ```python
    model.score(u,y)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise
    - Pick a dataset, select appropriate target variable and features, fit a $k$-NN classifier, and compute the accuracy
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Selecting $k$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hyperparameters
    - The variable $k$ is our first **hyper-parameter**: There's not an obvious way to pick it, just from looking at the data and the model themselves
    - Our models are typically very **greedy**: If you let them decide, they tend to pick more variables, more complexity, more nuance
    - This leads to models that are overconfident and unreliable, when used in practice
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python

    import matplotlib.pyplot as plt

    k_grid = np.array([ (2*k+1)**2 for k in range(0,11)]) # Odd number grid
    accuracies = []
    for k in k_grid: # For each candidate value of k...
        model = KNeighborsClassifier(n_neighbors = k) # Create a model instance
        model = model.fit(u,y) # Fit the model
        y_hat = model.predict(u) # Hard prediction
        correct = (y==y_hat).astype(int)
        sns.scatterplot(x=df['baseline price'],y=df['footprint'],
                        hue = correct,
                        style= correct)
        plt.show() # Render plot
        acc = model.score(u,y) # Compute accuracy
        print( f'Accuracy for {k} neighbors is {acc}')
        accuracies.append(acc) # Save accuracy

    sns.lineplot(x=k_grid,y=accuracies).set(title='Accuracy',xlabel='Neighbors')

    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Over-fitting on the Test Set
    - What this says is: "Pick $k=1$! That means, when you compare the predicted label to the actual label for every instance in the training data, you get the right answer!"
    - As we increase $k$, we start making mistakes by taking the majority class label in a neighborhood of that size
    - This logic is independent of the application. Is $k=1$ always the answer?
    - Everything we did made sense, but we're missing something
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Keywords: Overfitting, Underfitting
    - If we pick $k$ too low, the model is overly sensitive to a handful of data points; this is **overfitting**
    - If we pick $k$ too high, the model averages over many observations and will give answers close to population proportions; this is **underfitting**
    - We want to avoid under/over-fitting by building robust, appropriately complex models
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What are we doing wrong?
    - In machine learning, at this level, we are trying to **predict**
    - So our thought experiment is this: "Tomorrow, a new case walks through the door and you use your model. Is it likely to be accurate, or not?"
    - What we were doing is asking, "On the **training data** which the model has already seen, does the model perform well or not?"
    - We want to replicate the experience of making predictions with the model, on new cases which the model has not yet seen
    - How do we replicate this kind of uncertainty?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train-Test Splits
    - A **train-test split** randomly partitions the data into two sets:
        1. Training data, usually about 80% of the sample, on which the model is trained
        2. Test data, usually about 20% of the sample, on which performance metrics like confusion matrices and accuracy are computed
    - We are imagining that the test set is *like* the future cases we have not yet observed (i.e. drawn from the same joint distribution, having the same data generating process), and model performance on the test set will be similar to future performance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train-Test Splits
    - A train test split with Scikit:
    ```python
    from sklearn.model_selection import train_test_split
    u_train, u_test, y_train, y_test = train_test_split(u,y, # Feature and target variables
                                                        test_size=.2, # Split the sample 80 train/ 20 test
                                                        random_state=100) # For replication purposes
    ```
    - A train-test split of the dataframe with Pandas:
    ```python
    df = df.sample(frac=1, random_state = 100)
    test_size = int( .2 * len(df) )
    df_test = df.iloc[:test_size,:].reset_index(drop=True)
    df_train = df.iloc[test_size:,:].reset_index(drop=True)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Training and Evaluation

    ```python
    from sklearn.model_selection import train_test_split
    u_train, u_test, y_train, y_test = train_test_split(u,y,
                                                        test_size=.2,
                                                        random_state=100)
    # Set the number of neighbors, typically odd to "break ties":
    k = 5
    # Create a fitted model instance:
    model = KNeighborsClassifier(n_neighbors = k) # Create a model instance
    model = model.fit(u_train,y_train) # Fit the model, training set
    y_hat = model.predict(u_test) # Predictions, test set
    print( pd.crosstab(y_test, y_hat)) # Confusion matrix
    model.score(u_test,y_test) # Test accuracy
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choosing $k$
    - Now we loop over $k$, evaluating the accuracy on the test set to see what values of $k$ are best when confronted with new data:

    ```python
    k_grid = np.array([ (2*k+3) for k in range(0,150)]) # Odd number grid
    test_accuracies = []
    train_accuracies = []
    for k in k_grid: # For each candidate value of k...
        model = KNeighborsClassifier(n_neighbors = k) # Create a model instance
        model = model.fit(u_train,y_train) # Fit the model
        y_hat = model.predict(u_test) # Predict values
        test_acc = model.score(u_test,y_test) # Compute test accuracy
        train_acc = model.score(u_train,y_train) # Compute trainin accuracy
        print( f'Test accuracy for {k} neighbors is {test_acc}; train accuracy for {k} neighbors is {train_acc}')
        test_accuracies.append(test_acc) # Save test results
        train_accuracies.append(train_acc) # Save training results

    sns.lineplot(x=k_grid,y=train_accuracies,label='Training Accuracy').set(xlabel='Neighbors',ylabel='Accuracy')
    sns.lineplot(x=k_grid,y=test_accuracies,label='Test Accuracy')
    plt.show()
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## "Optimal" $k$
    - Test Accuracy seems to peak between 35 and 65; it depends on the train-test split, and the granularity of the $k$ grid

    ```python
    is_optimal = test_accuracies == np.max(test_accuracies) # Maximizer Boolean
    optimal_indices = np.where( is_optimal ) # Indices that maximize accuracy
    k_optimal = k_grid[ optimal_indices ] # Values of k that maximize accuracy
    ```

    - I got $k^* = 43$ with an accuracy of .747
    - Training Accuracy peaks at $k=1$ and then declines, like we expected
    - This train-test split approach is somewhat noisy, and we'll introduce other, more robust tools
    - But this concept --- test model performance on data it has not been trained on --- is a core idea from machine learning that we will use over and over and explore in greater depth as the course goes on
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise:
    - Determine a good $k$ for your data from earlier.
    """)
    return


if __name__ == "__main__":
    app.run()
