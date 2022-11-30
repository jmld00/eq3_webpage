import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

iris_raw= pd.read_csv("iris.data", header=None)
iris_raw.columns=["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "species"]
iris=iris_raw.loc[:, iris_raw.columns!='species']

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

controls = dbc.Card(
    [
        html.Div(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                    id="x-variable",
                    options=[
                        {"label": col, "value": col} for col in iris.columns
                    ],
                    value="sepal length in cm",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                    id="y-variable",
                    options=[
                        {"label": col, "value": col} for col in iris.columns
                    ],
                    value="sepal width in cm",
                ),
            ]
        ),
        html.Div(
            [
                dbc.Label("Cluster count"),
                dbc.Input(id="cluster-count", type="number", value=3),
            ]
        ),
       html.Div(
            [
                dbc.Label("Species Selector"),
                dcc.Dropdown(
                    id="speciesVariable",
                    options=[
                        {"label": col, "value": col} for col in iris_raw.species.unique()
                    ],
                    value="Iris-setosa",
                ),
            ]
        ),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        html.H1("Iris Dataset"),
        html.Hr(),
        dbc.Row(
            [   
                dbc.Col(controls, md=4),
                dbc.Col(dcc.Graph(id="cluster-graph"), md=8),
            ],
            align="center",
        ),
        dbc.Row(
            [   
                dbc.Col(html.Img(id="iris_imgs"), md={"width":4, "offset":1}, sm={"width":4, "offset":1}),
                dbc.Col(dcc.Graph(id="regresion-graph"), md=8),
            ],
            align="center",
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("cluster-graph", "figure"),
    [
        Input("x-variable", "value"),
        Input("y-variable", "value"),
        Input("cluster-count", "value"),
    ],
)
def kmeans_graph(x, y, n_clusters):
    # minimal input validation, make sure there's at least one cluster

    km = KMeans(n_clusters=max(n_clusters, 1))
    df = iris.loc[:, [x, y]]
    km.fit(df.values)
    df["cluster"] = km.labels_

    centers = km.cluster_centers_
	
    data = [
        go.Scatter(
            x=df.loc[df.cluster == c, x],
            y=df.loc[df.cluster == c, y],
            mode="markers",
            marker={"size": 8},
            name="Cluster {}".format(c),
        )
        for c in range(n_clusters)
    ]

    data.append(
        go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker={"color": "#000", "size": 12, "symbol": "diamond"},
            name="Cluster centers",
        )
    )

    layout = {"xaxis": {"title": x}, "yaxis": {"title": y}, "title":"Iris: K-means"}

    return go.Figure(data=data, layout=layout)


# make sure that x and y values can't be the same variable
def filter_options(v):
    """Disable option v"""
    return [
        {"label": col, "value": col, "disabled": col == v}
        for col in iris.columns
    ]


# functionality is the same for both dropdowns, so we reuse filter_options
app.callback(Output("x-variable", "options"), [Input("y-variable", "value")])(
    filter_options
)
app.callback(Output("y-variable", "options"), [Input("x-variable", "value")])(
    filter_options
)


@app.callback(
    Output("regresion-graph", "figure"),
    [
        Input("x-variable", "value"),
        Input("y-variable", "value"),
        Input("speciesVariable", "value"),
    ],
)
def regresion_graph(x, y, species):
    df = iris_raw[iris_raw.species==species]

    model = LinearRegression()
    model.fit(df[x].values.reshape(-1, 1), df[y])
    
    x_range = np.linspace(df[x].min(), df[x].max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

	
    data = [
        go.Scatter(
            x=df[x],
            y=df[y],
            mode="markers",
            marker={"size": 8},
        )
    ]

    data.append(
        go.Scatter(
            x=x_range,
            y=y_range,
            name="Regression",
        )
    )

    layout = {"xaxis": {"title": x}, "yaxis": {"title": y}, "title":"Reggresion: "+ x + " Vs "+ y}

    return go.Figure(data=data, layout=layout)

@app.callback(
    [Output("iris_imgs", "src"),
     Output("iris_imgs", "title"),
    ],
    Input("speciesVariable", "value"),
)
def images_iris(specie):
    speciesDict={'Iris-setosa':'https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/330px-Kosaciec_szczecinkowaty_Iris_setosa.jpg', 
                 'Iris-versicolor':'https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/330px-Iris_versicolor_3.jpg', 
                 'Iris-virginica':'https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/330px-Iris_virginica.jpg'}

    return speciesDict[specie], "Flower: "+specie

if __name__ == "__main__":
    app.run_server(debug=True)