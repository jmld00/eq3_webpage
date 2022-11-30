import random
import pandas as pd
import dash
from dash.dependencies import Input, Output
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Whirpool Data Analitics'
app._favicon = ("page_wicon.png")

app = dash.Dash(__name__, use_pages=True, pages_folder="pages")

app.layout = html.Div(
    [
        # main app framework
        html.Div("Python Multipage App with Dash", style={'fontSize':50, 'textAlign':'center'}),
        html.Div([
            dcc.Link(page['name']+"  |  ", href=page['path'])
            for page in dash.page_registry.values()
        ]),
        html.Hr(),

        # content of each page
        dash.page_container
    ]
)


if __name__ == "__main__":
    app.run(debug=True)