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
data['Eficiencia'] = np.where(data['% Below Rating Point'] < 0, 'Ahorra', 'No ahorra')
data['Temperatura promedio'] = 'Normal'
data['Temperatura promedio'] = np.where((data['RC Temp Average °F (M/M)'] > 40) | (data['FC Temp Average °F (M/M)'] > 4), 'Muy Caliente',
                                      np.where((data['RC Temp Average °F (M/M)'] < 36), 'Muy Fría', 'Normal'))

# FRONTEND
familias = ['General'] + list(data['Familia'].unique())
variable = dcc.Dropdown(
        id='familia',
        options=[{'label': i, 'value': i} for i in familias],
        value = 'General',
        style={'width': '100%'},
        clearable=False
    )

content = html.Div([
    dcc.Graph(id='graph1', style={'width': '100%', 'height': '95vh', 'responsive': 'True'})
])

app.layout = html.Div([variable, content])

# BACKEND
@app.callback(
    Output(component_id='graph1', component_property='figure'),
    Input(component_id='familia', component_property='value')
)

def pie(familia):
    if familia != 'General':
        ahorra_count = pd.DataFrame(data[data['Familia'] == familia]['Temperatura promedio'].value_counts()).reset_index()
    else:
        ahorra_count = pd.DataFrame(data['Temperatura promedio'].value_counts()).reset_index()
    pies = go.Pie(
        labels = ahorra_count['index'].values,
        values = ahorra_count['Temperatura promedio'].values
    )
    title = {'text': f'Comportamiento de <b>Temperatura Promedio</b> <br> <b>Familia</b>: {familia}',
        'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 20}
    layout = {'title':title, 'xaxis': {'title': 'Familia'},
                    'yaxis': {'title': 'Temperatura [°F]'}, 'showlegend':True, 'template': 'simple_white'}
    fig = go.Figure(data=pies, layout=layout)
    colors = ['#bebebe', '#dd6121', '#8bbfde'] 
    fig.update_traces(hoverinfo='label+percent+value', textinfo='percent', textfont_size=20,
                    marker=dict(colors = colors, line=dict(color='#ffffff', width=5)))
    fig.update_layout(
                    showlegend = True,
                    legend_bordercolor = '#333',
                    legend_borderwidth = 2,
                    legend_orientation = 'h',
                    legend_yanchor = 'bottom',
                    legend_y = -0.2,
                    legend_xanchor = 'center',
                    legend_x = 0.5,
                    legend_font_size = 20)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)





