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
    # W07 From Tidy Data to Graphs
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set Up
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import igraph as ig
    return ig, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get Data
    """)
    return


@app.cell
def _(pd):
    data_file = "winereviews-DOC_SHORT.csv"
    # data_file = "winereviews-DOC_MOD.csv"
    DOC = pd.read_csv(data_file)
    DOC = DOC.dropna(subset=['doc_taster', 'doc_country', 'doc_variety']) # Drop rows with nulls values in key cols
    return (DOC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Get shape of data set.
    """)
    return


@app.cell
def _(DOC):
    DOC.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Nodes and Edges
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are going to create a "bipartite" graph.

    This means we are going to see how two categorical variables are related to each other.

    So, let's begin by picking two columns we want to be nodes in our graph.

    We'll choose Country and Variety.
    """)
    return


@app.cell
def _():
    node_cols = ['doc_country', 'doc_taster']
    return (node_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's create a `N` table by extracting the set of nodes from `DOC` using melt and value counts.

    We'll also rename the columns and index of our resulting data frame.
    """)
    return


@app.cell
def _(DOC, node_cols):
    _N = DOC[node_cols].melt().value_counts().to_frame('n')
    _N.index.names = ['type', 'label']
    _N = _N.reset_index()
    _N.index.name = 'id'
    _N.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's create the `EDGES` table by extracting the edges from `DOC`.

    Edges are just relationships between nodes.

    Each row in `DOC` contains relationships between columns.

    We'll just focus on the two columns we picked above.
    """)
    return


@app.cell
def _(DOC, node_cols):
    EDGES = DOC[node_cols].value_counts().to_frame('n').reset_index()
    EDGES.index.name = 'id'
    EDGES.head()
    return


@app.cell
def _(DOC, node_cols):
    NODES = DOC[node_cols].melt().value_counts().to_frame('n')
    NODES.index.names = ['type', 'label']
    NODES = NODES.reset_index()
    NODES.index.name = 'id'
    NODES.head()
    return (NODES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let's create the `EDGES` table by extracting the edges from `DOC`.

    Edges are just relationships between nodes.

    Each row in `DOC` contains relationships between columns.

    We'll just focus on the two columns we picked above.
    """)
    return


@app.cell
def _(DOC, node_cols):
    EDGES_1 = DOC[node_cols].value_counts().to_frame('n').reset_index()
    EDGES_1.index.name = 'id'
    EDGES_1.head()
    return (EDGES_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's replace the names in the `EDGES` table with the ids from the `NODES` table.

    This is not strictly necessary, but it is a good practice.
    """)
    return


@app.cell
def _(EDGES_1, NODES, node_cols):
    def replace_names_with_ids(col_name):
        if col_name not in EDGES_1.columns:
            return False  # Check if we've already modified the EDGES table.
        X = NODES[NODES.type == col_name].reset_index().set_index('label')
        EDGES_1[col_name] = EDGES_1[col_name].map(X.id)
    for col in node_cols:
        replace_names_with_ids(col)  # We create a version of NODES that contains only rows associated with the node type,
    EDGES_1.columns = ['source', 'target', 'n']  # and where the index is the node label  # We use .map() to replace the key of the X with the id column in X.
    return


@app.cell
def _(EDGES_1):
    EDGES_1.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convert to Igraph
    Now we convert our data frames into a graph object using Python's Igraph package.
    """)
    return


@app.cell
def _(EDGES_1, NODES, ig):
    # Create graph from EDGES
    g = ig.Graph(edges=EDGES_1[['source', 'target']].values.tolist())
    g.vs['label'] = NODES.label.to_list()
    # Add vertex (node) attributes
    g.vs['count'] = NODES.n.to_list()  # [NODES.loc[i, 'label'] for i in range(len(NODES))]
    g.vs['node_type'] = NODES.type.to_list()  # [NODES.loc[i, 'n'] for i in range(len(NODES))]
    # Add edge attributes
    g.es['weight'] = EDGES_1['n'].tolist()  # [NODES.loc[i, 'type'] for i in range(len(NODES))]
    return (g,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Save as GraphML file.

    This file can be opened up in tools like Gephi and CytoScape.
    """)
    return


@app.cell
def _(g):
    g.write_graphml("wine-taster-country.graphml")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generalize
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Break out into two functions.
    """)
    return


@app.cell
def _(pd):
    def tidy_to_graph(df:pd.DataFrame, node_cols:list):
        """
        Converts columns from a tidy table into a graph comprised of NODES and EDGES data frames.

        Works for bipartite graphs -- graphs with connections between two columns.
        """

        # Create nodes
        NODES = df[node_cols].melt().value_counts().to_frame('n')
        NODES.index.names = ['type', 'label']
        NODES = NODES.reset_index()
        NODES.index.name = 'id'

        # Create edges
        EDGES = df[node_cols].value_counts().to_frame('n').reset_index()
        EDGES.index.name = 'id'

        # Function to replace node names with node ids in the EDGES data frame
        def replace_names_with_ids(col_name):
            # Temporary NODES table the selected nodes for column name and swaps the index to label        
            X = NODES[NODES.type == col_name].reset_index().set_index('label')
            # Replaces column value with id using .map()
            EDGES[col_name] = EDGES[col_name].map(X.id)

        # Replace names with ids in EDGES
        for col in node_cols:
            replace_names_with_ids(col)

        # Renames EDGES columns
        EDGES.columns = ['source', 'target', 'n']

        # Returns both data frames
        return NODES, EDGES
    return (tidy_to_graph,)


@app.cell
def _(ig, pd):
    def graph_to_igraph(NODES:pd.DataFrame, EDGES:pd.DataFrame, directed:bool=False):
        """
        Converts NODES and EDGES directly into an igraph Graph object.
            """
        # Create graph from EDGES
        g = ig.Graph(edges=EDGES[['source', 'target']].values.tolist())
        
        # Add vertex (node) attributes
        g.vs['label'] = NODES['label'].to_list() # [NODES.loc[i, 'label'] for i in range(len(NODES))]
        g.vs['count'] = NODES['n'].to_list() # [NODES.loc[i, 'n'] for i in range(len(NODES))]
        g.vs['node_type'] = NODES['type'].to_list() # [NODES.loc[i, 'type'] for i in range(len(NODES))]
        
        # Add edge attributes
        g.es['weight'] = EDGES['n'].tolist()    
    
        return g
    return (graph_to_igraph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Test out the functions.
    """)
    return


@app.cell
def _(DOC, graph_to_igraph, tidy_to_graph):
    _my_node_cols = ['doc_taster', 'doc_country']
    _E, _N = tidy_to_graph(DOC, _my_node_cols)
    _G = graph_to_igraph(_E, _N)
    _G.write_graphml('wine-taster-country.graphml')
    return


@app.cell
def _(DOC, graph_to_igraph, tidy_to_graph):
    _my_node_cols = ['doc_country', 'doc_variety']
    _E, _N = tidy_to_graph(DOC, _my_node_cols)
    _G = graph_to_igraph(_E, _N)
    _G.write_graphml('FOO-wine-country-variety.graphml')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Display with iGraph
    """)
    return


@app.cell
def _(EDGES_1, NODES, graph_to_igraph):
    g_1 = graph_to_igraph(NODES, EDGES_1)
    return (g_1,)


@app.cell
def _():
    # g = ig.Graph.Read_GraphML("wine-taster-country.graphml")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    See https://python.igraph.org/en/stable/tutorial.html#layout-algorithms for layout options.

    ![image.png](attachment:image.png)
    """)
    return


@app.cell
def _():
    layout_type = "rt_circular"
    # layout_type = 'dh'
    return (layout_type,)


@app.cell
def _(g_1, ig, layout_type, plt):
    _fig, _ax = plt.subplots(figsize=(20, 20))
    ig.plot(g_1, target=_ax, layout=layout_type, vertex_color=['#ccff99' if type == 'doc_country' else '#ffcc99' for type in g_1.vs['node_type']], vertex_frame_width=1.0, vertex_frame_color='gray', vertex_label=g_1.vs['label'], vertex_label_size=12.0)
    plt.title(f'Tasters and Countries\n({layout_type})', fontsize=20)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's make the node sizes and edge widths proportional to their degrees.
    """)
    return


@app.cell
def _(g_1, pd):
    egde_weight = pd.Series(g_1.es['weight'])
    node_count = pd.Series(g_1.vs['count'])
    return egde_weight, node_count


@app.cell
def _(egde_weight, g_1, ig, layout_type, node_cols, node_count, plt):
    _fig, _ax = plt.subplots(figsize=(20, 20))
    ig.plot(g_1, target=_ax, layout=layout_type, edge_width=egde_weight / egde_weight.sum() * 100, vertex_size=node_count / node_count.sum() * 500, vertex_color=['#ccff99' if type == node_cols[0] else '#ffcc99' for type in g_1.vs['node_type']], vertex_frame_width=1.0, vertex_frame_color='gray', vertex_label=g_1.vs['label'], vertex_label_size=12.0)
    plt.title(f'Tasters and Countries\nwith Sized Nodes and Edges\n({layout_type})', fontsize=20)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Write a helper function.
    """)
    return


@app.cell
def _(g_1, ig, pd, plt):
    def graphplot(my_g, node_cols, layout_type='rt_circular', sized=False, title=None, figsize=(20, 20)):
        if sized:
            egde_weight = pd.Series(g_1.es['weight'])
            node_count = pd.Series(g_1.vs['count'])
            e_w = egde_weight / egde_weight.sum() * 100
            v_s = node_count / node_count.sum() * 500
        else:
            e_w = None
            v_s = None
        _fig, _ax = plt.subplots(figsize=figsize)
        ig.plot(my_g, target=_ax, layout=layout_type, edge_width=e_w, vertex_size=v_s, vertex_color=['#ccff99' if type == node_cols[0] else '#ffcc99' for type in g_1.vs['node_type']], vertex_frame_width=1.0, vertex_frame_color='gray', vertex_label=g_1.vs['label'], vertex_label_size=12.0)
        plt.title(title, fontsize=20)
        plt.show()
    return (graphplot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Try it out.
    """)
    return


@app.cell
def _(g_1, graphplot, layout_type, node_cols):
    graphplot(g_1, node_cols=node_cols, layout_type=layout_type, title='My Graph', sized=True)
    return


if __name__ == "__main__":
    app.run()
