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
# Datos
data = pd.read_excel('WhirlpoolLimpiaFinal.xlsx', index_col=0, engine='openpyxl')
data['Test Completion Date'] = pd.to_datetime(data['Test Completion Date'])

# TABLERO 4: REGRESIONES
# 1. Opciones de entrada y salida 
column_list = data.columns.values.tolist() # Variables continuas 
objects = []
numeric = []
for column_name in column_list:
  if data.dtypes[column_name] == 'O':
    objects.append(column_name)
  else:
    numeric.append(column_name)
numeric.remove('Production Line')
numeric.remove('Target')
numeric.remove('Ability')
options = numeric

# 2. Opciones para agrupar 
group = ['General', 'Refrigerant', 'Supplier', 'E-star/Std.', 'Cluster']
familias = ['General'] + list(data['Familia'].unique()) 

# 3. última fecha de actualización
update = str(data['Test Completion Date'].max())[0:11]


sidebar_style = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '16rem',
    'padding': '1rem 1rem',
    'background-color': '#666666'
}

sidebar = html.Div([
    html.Div(html.H6('Última actualización', className= 'text-center', style= {'color': '#ffffff'}),
    style={'padding': '0px 0px 0px 0px'}),

    html.Hr(style={'color': '#ffffff', 'height': '2px'}),

    html.Div(html.H6(f'{update}', className= 'text-center', style= {'color': '#ffffff'}),
    style={'padding': '0px 0px 0px 0px'}),

    html.Hr(style={'color': '#ffffff', 'height': '2px'}),

    html.Div(html.H6('Variables', className= 'text-center', style= {'color': '#ffffff'}),
    style={'padding': '0px 0px 0px 0px'}),
    html.Hr(style={'color': '#ffffff', 'height': '2px'}),

    html.Div(
        [html.Div(html.H6('Familia', className= 'text-center text-white'),
        style={'padding': '0.1rem 0rem 0rem 0rem', 'color': '#ffffff'}),
        dcc.Dropdown(
        id='ddl_w',
        options=[{'label': i, 'value': i} for i in familias],
        value = 'General',
        style={'width': '100%'},
        clearable=False
    )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'}),
    
    html.Div(
        [html.Div(html.H6('Entrada', className= 'text-center text-white'),
        style={'padding': '0px 0px 0px 0px', 'color': '#ffffff'}),
        dcc.Dropdown(
        id='ddl_x',
        options=[{'label': i, 'value': i} for i in options],
        value='Energy Usage kWh/day (M/M)',
        style={'width':'100%'},
        clearable=False
    )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'}),

    html.Div(
        [html.Div(html.H6('Salida', className= 'text-center text-white'),
        style={'padding': '0.1rem 0rem 0rem 0rem', 'color': '#ffffff'}),
        dcc.Dropdown(
        id='ddl_y',
        options=[{'label': i, 'value': i} for i in options],
        value='Energy Consumed (kWh/yr)',
        style={'width':'100%'},
        clearable=False
    )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'}),


    html.Div(
        [html.Div(html.H6('Categoría', className= 'text-center text-white'),
        style={'padding': '0.1rem 0rem 0rem 0rem', 'color': '#ffffff'}),
        dcc.Dropdown(
        id='ddl_z',
        options=[{'label': i, 'value': i} for i in group],
        value = 'General',
        style={'width': '100%'},
        clearable=False
    )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'})
    
], style=sidebar_style)

content_style = {
    'margin-left': '16rem'
}

content = html.Div([
    dcc.Graph(id='graph1', style={'width': '176vh', 'height': '100vh', 'responsive': 'True'})
], style=content_style)

app.layout = html.Div([sidebar, content])


# BACKEND 
@app.callback(
    Output(component_id='graph1', component_property='figure'), 
    [
        Input(component_id='ddl_x', component_property='value'),
        Input(component_id='ddl_y', component_property='value'),
        Input(component_id='ddl_z', component_property='value'),
        Input(component_id='ddl_w', component_property='value')
    ]
)
def update_output(ddl_x_value, ddl_y_value, ddl_z_value, ddl_w_value):
    # Se selecciona familia y se selecciona grupo 
    if (ddl_w_value != 'General') and (ddl_z_value != 'General'):
        graphs = []
        colors = ['#eeb111', '#333']
        classes = list(data[data['Familia'] == ddl_w_value][ddl_z_value].unique())
        amount = data[data['Familia'] == ddl_w_value].shape[0]
        title_text = f'<b>Familia:</b> {ddl_w_value} <br> <b>Unidades totales</b>: {amount} <br>'
        for cls, i in zip(classes, range(len(classes))):
            x = data[(data[ddl_z_value] == cls) & (data['Familia'] == ddl_w_value)][ddl_x_value]
            y = data[(data[ddl_z_value] == cls) & (data['Familia'] == ddl_w_value)][ddl_y_value]
            model = LinearRegression()
            model.fit(x.values.reshape(-1,1), y)
            x_range = np.linspace(x.min(), x.max(), 100)
            y_range = model.predict(x_range.reshape(-1,1))
            graphs.append(go.Scatter(x=x, y=y, mode='markers', marker_color= colors[i], name=f'Datos {cls}'))
            graphs.append(go.Scatter(x=x_range, y=y_range, marker_color= colors[i], name=f'Tendencia {cls}'))
            cls_amount = data[(data[ddl_z_value] == cls) & (data['Familia'] == ddl_w_value)].shape[0]
            title_text = title_text + f'<b>{cls}:</b>{cls_amount}  '

    # Se selecciona familia pero no se selecciona grupo 
    elif (ddl_w_value != 'General') and (ddl_z_value == 'General'):
        x = data[data['Familia'] == ddl_w_value][ddl_x_value]
        y = data[data['Familia'] == ddl_w_value][ddl_y_value]
        model = LinearRegression()
        model.fit(x.values.reshape(-1,1), y)
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range = model.predict(x_range.reshape(-1,1))
        graphs = [go.Scatter(x=x, y=y, mode='markers', marker_color='#333', name='Datos')]
        graphs.append(go.Scatter(x=x_range, y=y_range, name='Tendencia', marker_color='#eeb111'))
        amount = data[data['Familia'] == ddl_w_value].shape[0]
        title_text = f'<b>Familia:</b> {ddl_w_value} <br> <b>Unidades totales</b>: {amount} <br>'
    
    # No se selecciona familia pero se selecciona grupo 
    elif (ddl_w_value == 'General') and (ddl_z_value != 'General'):
        amount = data.shape[0]
        graphs = []
        colors = ['#eeb111', '#333']
        classes = list(data[ddl_z_value].unique())
        title_text = f'<b>Familia:</b> {ddl_w_value} <br> <b>Unidades totales</b>: {amount} <br>'
        for cls, i in zip(classes, range(len(classes))):
            x = data[(data[ddl_z_value] == cls)][ddl_x_value]
            y = data[(data[ddl_z_value] == cls)][ddl_y_value]
            model = LinearRegression()
            model.fit(x.values.reshape(-1,1), y)
            x_range = np.linspace(x.min(), x.max(), 100)
            y_range = model.predict(x_range.reshape(-1,1))
            graphs.append(go.Scatter(x=x, y=y, mode='markers', marker_color= colors[i], name=f'Datos {cls}'))
            graphs.append(go.Scatter(x=x_range, y=y_range, marker_color= colors[i], name=f'Tendencia {cls}'))
            cls_amount = data[(data[ddl_z_value] == cls)].shape[0]
            title_text = title_text + f'<b>{cls}:</b>{cls_amount}   '

    # No se selecciona familia y no se selecciona grupo 
    else:
        amount = data.shape[0]
        x = data[ddl_x_value]
        y = data[ddl_y_value]
        model = LinearRegression()
        model.fit(x.values.reshape(-1,1), y)
        x_range = np.linspace(x.min(), x.max(), 100)
        y_range = model.predict(x_range.reshape(-1,1))
        graphs = [go.Scatter(x=x, y=y, mode='markers', marker_color='#333', name='Datos')]
        graphs.append(go.Scatter(x=x_range, y=y_range, name='Tendencia', marker_color='#eeb111'))
        amount = data.shape[0]
        title_text = f'<b>Familia:</b> {ddl_w_value} <br> <b>Unidades totales</b>: {amount} <br>'


    title = {'text': title_text, 
    'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_color': 'white'}
    layout = {'title':title, 'xaxis': {'title': f'{ddl_x_value}', 'color':'white'},
                'yaxis': {'title': f'{ddl_y_value}', 'color':'white'}}
    fig = go.Figure(data=graphs, layout=layout)
    fig.update_layout(plot_bgcolor = '#d6d6d6', paper_bgcolor='#b1b1b1',
                      legend_bordercolor = '#dbdbdb',legend_borderwidth = 2)

    return fig



if __name__ == '__main__':
    app.run_server(debug=True)