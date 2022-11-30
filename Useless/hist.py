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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# Datos
data = pd.read_excel('WhirlpoolLimpiaFinal.xlsx', index_col=0, engine='openpyxl')
data['Test Completion Date'] = pd.to_datetime(data['Test Completion Date'])

# FRONTEND
compartimentos = ['Refrigerador', 'Congelador']
variable = dcc.Dropdown(
        id='compartimento',
        options=[{'label': i, 'value': i} for i in compartimentos],
        value = 'Refrigerador',
        style={'width': '100%'},
        clearable=False
    )

content = html.Div([
    dcc.Graph(id='graph1', style={'width': '100%', 'height': '95vh', 'responsive': 'True'})
])


app.layout = html.Div([variable, content])

def def_color(compartimento, valor):
    if compartimento == 'Refrigerador':
        if valor > 40:
            color = '#dd6121'
            line = '#bc4d23'
            opacity= 1
        elif valor < 36:
            color = '#8bbfde'
            line = '#3a6d9f'
            opacity = 1
        else:
            color = '#fff'
            line = '#333'
            opacity = 0.7
    elif compartimento == 'Congelador':
        if valor > 4:
            color = '#dd6121'
            line = '#bc4d23'
            opacity= 1
        else:
            color = '#fff'
            line = '#333'
            opacity = 0.7
    
    return color, line, opacity

platforms = data['Platform'].unique()
groups = []
for platform in platforms:
  familias = list(data[data['Platform'] == platform]['Familia'].unique())
  groups.append(familias)
matrix = groups[0]
highland = groups[1]
pira = groups[2]
familias = highland + matrix + pira

# BACKEND
@app.callback(
    Output(component_id='graph1', component_property='figure'),
    Input(component_id='compartimento', component_property='value')
)

def update(compar):
    if compar == 'Refrigerador':
        variable = 'RC Temp Average °F (M/M)'
        y0 = 40
        y1 = 36
        y2 = 42.5

    elif compar == 'Congelador':
        variable = 'FC Temp Average °F (M/M)'
        y0 = -11
        y1 = 4
        y2 = 10
    
    puntos = []
    for familia, hor in zip(familias, range(10, 21, 1)):
        valores = data[data['Familia'] == familia][variable].sort_values(ascending=True).values
        for valor in valores:
            color, line, opacity = def_color(compar, valor)
            puntos.append(go.Scatter(
                x = [hor],
                y = [valor],
                opacity = opacity,
                mode = 'markers',
                marker_color = color,
                marker_size = 20,
                marker_line_width = 2,
                marker_line_color = line
            ))
    
    title = {'text': f'Temperaturas promedio en el <b>{compar}</b>', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 20}
    layout = {'title':title, 'xaxis': {'title': 'Familia'},
                'yaxis': {'title': 'Temperatura [°F]'}, 'showlegend':False, 'template': 'simple_white'}
    fig = go.Figure(data=puntos, layout = layout)
    fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(np.arange(10,21,1)),
        ticktext = familias
    ),
    margin=dict(r=20))
    fig.add_vline(x=10.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)
    fig.add_vline(x=16.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)
    for fam, x in zip(['Highland', 'Matrix', 'Pira XL'], [10, 13.5, 18.5]):
        fig.add_trace(go.Scatter(
            x = [x],
            y = [y2],
            mode = 'text',
            text = [f'<b>{fam}'],
            textposition='top center',
            textfont_size = 15
        ))

    fig.add_hrect(
        y0=y0, y1=y1, line_width=0, 
        fillcolor="#6aa342", opacity=0.2,
        annotation_text="Rango Óptimo",
        annotation_position="top left")

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
