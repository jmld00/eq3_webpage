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
from datetime import datetime as dt

app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# Datos
df = pd.read_excel('WhirlpoolLimpiaFinal.xlsx', index_col=0, engine='openpyxl')
df['% Below Rating Point'] = df['% Below Rating Point'] * 100
df['Test Completion Date'] = pd.to_datetime(df['Test Completion Date'], dayfirst=False)
df['Eficiencia'] = np.where(df['% Below Rating Point'] < 0, 'Ahorra', 'No ahorra')
df['Temperatura promedio'] = 'Normal'
df['Temperatura promedio'] = np.where((df['RC Temp Average °F (M/M)'] > 40) | (df['FC Temp Average °F (M/M)'] > 4), 'Muy Caliente',
                                      np.where((df['RC Temp Average °F (M/M)'] < 36), 'Muy Fría', 'Normal'))
df['Consumo'] = np.where(df['% Below Rating Point'] > 10, 'Excesivamente bajo', np.where(df['% Below Rating Point'] < -3, 'Excesivamente alto', 'Normal'))
df['Second Point Temp Setting (W/W or C/C)'] = df['Second Point Temp Setting (W/W or C/C)'].str.replace('CC', 'C/C')
df['Second Point Temp Setting (W/W or C/C)'] = df['Second Point Temp Setting (W/W or C/C)'].str.replace('WW', 'W/W')

column_list = df.columns.values.tolist() # Variables continuas 
objects = []
numeric = []
for column_name in column_list:
  if df.dtypes[column_name] == 'O':
    objects.append(column_name)
  else:
    numeric.append(column_name)
numeric.remove('Production Line')
numeric.remove('Target')
numeric.remove('Ability')
numeric.remove('Test Completion Date')
options = numeric

# 2. Opciones para agrupar 
# Familias
familias = ['General'] + list(df['Familia'].unique()) 

# Refrigerantes
refris = list(df['Refrigerant'].unique())

# Posición
pos = ['Primera', 'Segunda']

# Setting
sets = ['Ambas'] + list(df['Second Point Temp Setting (W/W or C/C)'].unique())

# 3. última fecha de actualización
update = str(df['Test Completion Date'].max())[0:11]
min_date = df['Test Completion Date'].min()
max_date = df['Test Completion Date'].max()




#FRONTEND
sidebar_style = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '12rem',
    'padding': '1rem 1rem',
    'background-color': '#666666',
    'heigh': '100vh',
    'border-radius': '0px 7px 7px 0px'
}

sidebar = html.Div([
    html.Div(html.H6('Última actualización', className= 'text-center', style= {'color': '#ffffff'}),
    style={'padding': '0px 0px 0px 0px'}),

    html.Hr(style={'color': '#ffffff', 'height': '2px'}),

    html.Div(html.H6(f'{update}', className= 'text-center', style= {'color': '#ffffff'}),
    style={'padding': '0px 0px 0px 0px'}),

    html.Hr(style={'color': '#ffffff', 'height': '2px'}),


    html.Div(
        [html.Div(html.H6('Fecha', className= 'text-center text-white'),
        style={'padding': '0px 0px 0px 0px', 'color': '#ffffff'}),
        dcc.DatePickerRange(
            id='ddl_x',
            month_format='MM/DD/YYYY',
            show_outside_days=True,
            minimum_nights=0,
            min_date_allowed=min_date,
            max_date_allowed=max_date,
            start_date=min_date,
            end_date=max_date,
            with_portal=True,
            number_of_months_shown = 3,
            style={'fontSize': 5, 'color': 'red'}
        )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'}),

    html.Div(
        [html.Div(html.H6('Familia', className= 'text-center text-white'),
        style={'padding': '0.1rem 0rem 0rem 0rem', 'color': '#ffffff'}),
        dcc.Dropdown(
        id='ddl_y',
        value = 'General',
        style={'width': '100%'},
        clearable=False
    )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'}),
    
    html.Div(
        [html.Div(html.H6('Refrigernte', className= 'text-center text-white'),
        style={'padding': '0.1rem 0rem 0rem 0rem', 'color': '#ffffff'}),
        dcc.Dropdown(
        id='ddl_z',
        value = 'R600',
        style={'width': '100%'},
        clearable=False
    )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'}),

    html.Div(
        [html.Div(html.H6('Posición', className= 'text-center text-white'),
        style={'padding': '0.1rem 0rem 0rem 0rem', 'color': '#ffffff'}),
        dcc.Dropdown(
        id='ddl_w',
        options=[{'label': i, 'value': i} for i in pos],
        value = 'Primera',
        style={'width': '100%'},
        clearable=False
    )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'}),

    html.Div(
        [html.Div(html.H6('Setting', className= 'text-center text-white'),
        style={'padding': '0.1rem 0rem 0rem 0rem', 'color': '#ffffff'}),
        dcc.Dropdown(
        id='ddl_t',
        options=[{'label': i, 'value': i} for i in sets],
        value = 'Ambas',
        style={'width': '100%'},
        clearable=False
    )], style={'padding': '0.5rem 0.5rem 0.5rem 0.5rem'}),
    
], style=sidebar_style)

box_style = {'paddingTop': '1rem',
            'paddingRight': '0.5rem',
            'paddingLeft': '0.5rem',
            'box-shadow': '#e3e3e3 4px 4px 2px',
            'border-radius': '7px',
            'background-color':'white',
            'width': '22%',
            'height': '100%',
            'align': 'center',
            }
            
box_style2 = {'paddingTop': '1rem',
            'paddingRight': '0.5rem',
            'paddingLeft': '0.5rem',
            'box-shadow': '#e3e3e3 4px 4px 2px',
            'border-radius': '7px',
            'background-color':'white',
            'width': '100%',
            'height': '100h%',
            'align': 'center'}

graph_style1 = {'padding': '0.5rem',
            'boxShadow': '#e3e3e3 4px 4px 2px',
            'border-radius': '10px',
            'backgroundColor': 'white',
            'width': '49%',
            'height': '100%'
}

graph_style2 = {'padding': '0.5rem',
            'boxShadow': '#e3e3e3 4px 4px 2px',
            'border-radius': '10px',
            'backgroundColor': 'white',
            'width': '29%',
            'height': '100%'
}

graph_style3 = {'padding': '0.5rem',
            'boxShadow': '#e3e3e3 4px 4px 2px',
            'border-radius': '10px',
            'backgroundColor': 'white',
            'width': '69%',
            'height': '100%'
}


content = html.Div(
[  
    html.Div(children=[
    # Dato Duros
    html.Div(children=[
        html.H6(id='titlecount', className='text-center'),
        html.H6(id='unicount', className='text-center')
    ], style=box_style),

    html.Div(children=[
        html.H6(id='titlebelow', className='text-center'),
        html.H6(id='highcount', className='text-center')
    ], style=box_style),

    html.Div(children=[
        html.H6(id='titlehot', className='text-center'),
        html.H6(id='hotcount', className='text-center')
    ], style=box_style),

    html.Div(children=[
        html.H6(id='titlecold', className='text-center'),
        html.H6(id='coldcount', className='text-center')
    ], style=box_style),

    ], style={'height': '10vh', 'paddingTop':'0.5rem', 'paddingLeft': '0.5rem', 'paddingRight': '0.5rem','display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),

    # Análisis de energía
    html.Div(children=[
        html.Div(id='leftg',children=[
           dcc.Graph(id='cpk_graph', style={"height" : "100%", "width" : "100%"})
    ]),
        html.Div(id='rightg',children=[
            dcc.Graph(id='consumo_graph', style={"height" : "100%", "width" : "100%"})
    ]),
    ], style={'height': '45vh', 'paddingTop':'0.5rem', 'paddingLeft': '0.5rem', 'paddingRight': '0.5rem','display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),

    # Análisis de temperatura
    html.Div(children=[
        html.Div(children=[
           dcc.Graph(id='refri_graph', style={"height" : "100%", "width" : "100%"})
    ], style=graph_style1),
        html.Div(children=[
            dcc.Graph(id='conge_graph', style={"height" : "100%", "width" : "100%"})
    ], style=graph_style1),
    ], style={'height': '45vh', 'paddingTop':'0.5rem', 'paddingLeft': '0.5rem', 'paddingRight': '0.5rem', 'paddingBottom': '0.5rem','display': 'flex', 'justify-content': 'space-between', 'width': '100%', 'flex-wrap': 'wrap'}),
],
style={'height': '100%', 'background-color': '#D5d5d5', 'marginLeft': '12rem'})

app.layout = html.Div([sidebar, content], style={'background-color': '#D5d5d5'})

# Colores puntos en boxplots
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

#BACKEND
# Filtar las opciones a partir del rango de fechas
@app.callback(
    [
        Output('ddl_y', 'options'),
        Output('ddl_z', 'options')
    ],
    [
        Input('ddl_x', 'start_date'),
        Input('ddl_x', 'end_date')
    ]
)

def set_family_options(start_date, end_date):
    temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date)]
    fams = ['General'] + list(temp['Familia'].unique())
    refs = list(temp['Refrigerant'].unique())
    return [{'label': i, 'value':i} for i in fams], [{'label': i, 'value':i} for i in refs]

# Filtrar los datos duros 
@app.callback(
    [
        Output('unicount', 'children'),
        Output('titlecount', 'children'),
        Output('titlecount', 'style'),

        Output('highcount', 'children'),
        Output('titlebelow', 'children'),
        Output('titlebelow', 'style'),

        Output('hotcount', 'children'),
        Output('titlehot', 'children'),
        Output('titlehot', 'style'),

        Output('coldcount', 'children'),
        Output('titlecold', 'children'),
        Output('titlecold', 'style'),
    ]
    ,
    [
        Input('ddl_x', 'start_date'),
        Input('ddl_x', 'end_date'),
        Input('ddl_y', 'value'),
        Input('ddl_z', 'value')
    ]
)

def duros(start_date, end_date, fam, ref):
    if fam != 'General':
        temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Familia'] == fam)].copy()
        count_title = 'Unidades Probadas'
        count_style = {'paddingTop': '.1rem'}
        count = temp.shape[0]

        waste_title = 'Ahorro promedio'
        waste_style = {'paddingTop': '.1rem', 'color': 'green'}
        waste = temp['% Below Rating Point'].mean()
        waste = f'{round(waste,2)}%'

        hot_title = 'RC Promedio'
        hot_style = {'paddingTop': '.1rem', 'color': '#e76920'}
        hot = temp['RC Temp Average °F (M/M)'].mean()
        hot = f'{round(hot,2)} °F'

        cold_title = 'FC Promedio'
        cold_style = {'paddingTop': '.1rem', 'color': '#74a9d1'}
        cold = temp['FC Temp Average °F (M/M)'].mean()
        cold = f'{round(cold,2)} °F'

    else:
        temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref)].copy()

        count_title = 'Unidades Probadas'
        count_style = {'paddingTop': '.1rem'}
        count = temp.shape[0]

        waste_title = 'Consumo elevado'
        waste_style = {'paddingTop': '.1rem', 'color': 'red'}
        waste = temp[temp['Consumo'] == 'Excesivamente alto'].shape[0]

        hot_title = 'Muy calientes'
        hot_style = {'paddingTop': '.1rem', 'color': '#e76920'}
        hot = temp[temp['Temperatura promedio'] == 'Muy Caliente'].shape[0]

        cold_title = 'Muy Frías'
        cold_style = {'paddingTop': '.1rem', 'color': '#74a9d1'}
        cold = temp[temp['Temperatura promedio'] == 'Muy Fría'].shape[0]

    return f'{count}', count_title, count_style, f'{waste}', waste_title, waste_style, f'{hot}', hot_title, hot_style, f'{cold}', cold_title, cold_style

# Cambiar estilo energético
@app.callback(
    [
        Output('leftg', 'style'),
        Output('rightg', 'style')
    ],
    Input('ddl_y', 'value')
)

def size(fam):
    if fam != 'General':
        return graph_style2, graph_style3
    else:
        return graph_style1, graph_style1


# Filtrar CPK Graph
@app.callback(
        Output('cpk_graph', 'figure'),
    [
        Input('ddl_x', 'start_date'),
        Input('ddl_x', 'end_date'),
        Input('ddl_y', 'value'),
        Input('ddl_z', 'value')
    ]
)

def graph(start_date, end_date, fam, ref):
    if fam != 'General':
        temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Familia'] == fam)].copy()
        fam_data = temp.sort_values('Test Completion Date').copy()
        fam_data['R'] = abs(fam_data['Energy Consumed (kWh/yr)'].diff())

        cpks = []

        # Calculo CPK
        target = fam_data['Target'].max()
        usl = target * 1.03
        X = fam_data.tail(15)['Energy Consumed (kWh/yr)'].mean()
        R = fam_data.tail(15)['R'].mean()
        dev = R/1.128
        cpu = (usl - X)/(3 * dev)
        cpk = cpu
        cpks.append(cpk)

        if cpk > 1:
            color = '#109618'
        elif (cpk >= 0.8) and (cpk <= 1):
            color = '#ff9900'
        elif (cpk < 0.8):
            color = '#dc3912'

        gauge = go.Indicator(
        domain = {'x': [0,1], 'y': [0,1]},
        value = round(cpk,2),
        mode = 'gauge + number',
        title = {'text': f'<b>CPK</b>: {fam}', 'font': {'size': 10}},
        gauge = {'axis': {'range': [None, round(max(cpks))+1]},
                'bar': {'color': color},
                'steps':[
                    {'range': [0, 0.8], 'color': '#999999'},
                    {'range': [0.8, 1], 'color': '#cccccc'},
                    {'range': [1, round(max(cpks))+1], 'color': '#eeeeee'}]
                })
        layout = {'template': 'simple_white'}
        fig = go.Figure(data=gauge, layout=layout)
        fig.update_layout(
            margin=dict(l=0, b=5, t=5, r=0))

        return fig

    else:
        temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref)].copy()
        platforms = list(temp['Platform'].unique())
        familias = []
        for platform in platforms:
            fams = list(temp[temp['Platform'] == platform]['Familia'].unique())
            for fam in fams:
                familias.append(fam)

        bars = []
        cpks = []
        for familia, hor in zip(familias, range(10, 21, 1)):
            fam_data = temp[temp['Familia'] == familia].sort_values('Test Completion Date')
            ref =  fam_data.iloc[fam_data.shape[0] - 1]['Refrigerant']
            fam_data = fam_data[fam_data['Refrigerant'] == ref]
            fam_data['R']  = abs(fam_data['Energy Consumed (kWh/yr)'].diff())

            # Calculo CPK
            target = fam_data['Target'].max()
            usl = target * 1.03
            X = fam_data.tail(15)['Energy Consumed (kWh/yr)'].mean()
            R = fam_data.tail(15)['R'].mean()
            dev = R/1.128
            cpu = (usl - X)/(3 * dev)
            cpk = cpu
            cpks.append(cpk)

            if cpk > 1:
                color = '#109618'
            elif (cpk >= 0.8) and (cpk <= 1):
                color = '#ff9900'
            elif (cpk < 0.8):
                color = '#dc3912'

            bars.append(go.Bar(
                x = [hor],
                y = [cpk],
                marker_color = color,
                text = [f'<b>{round(cpk, 2)}</b>']
            ))

            ytop = max(cpks) + 0.5

        title = {'text': 'CPK POR <b>FAMILIA', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 10,
                'font_family': 'Arial'}
        layout = {'title':title, 'xaxis': {'title': 'FAMILIA', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}},
                        'yaxis': {'title': 'CPK', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}, 'range': [0, ytop]}, 'showlegend':False, 'template': 'simple_white'}
        fig = go.Figure(data=bars, layout = layout)
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = list(np.arange(10,21,1)),
                ticktext = familias
            ),
            margin=dict(l=0, b=0, t=30, r=0))
        fig.update_traces(textposition='outside')
        fig.update_yaxes(showgrid=True)

        return fig
    
# Filtrar Consumo Graph
@app.callback(
        Output('consumo_graph', 'figure'),
    [
        Input('ddl_x', 'start_date'),
        Input('ddl_x', 'end_date'),
        Input('ddl_y', 'value'),
        Input('ddl_z', 'value')
    ]
)

def consumo(start_date, end_date, fam, ref):
    if fam != 'General':
        temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Familia'] == fam)].copy()
        fix = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Familia'] == fam)].copy()
        min_x = fix['% Below Rating Point'].min()
        max_x = fix['% Below Rating Point'].max()
        x = temp['% Below Rating Point']
        hist = go.Histogram(x=x, marker_color='#eeb111', opacity=0.9)
        title = {'text': 'HISTOGRAMA DE <b>AHORRO DE ENERGÍA</b>', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 10}
        layout = {'title':title, 'xaxis': {'title': 'AHORRO DE ENERGÍA [%]', 'range': [min_x, max_x], 'tickfont': {'size': 10}, 'titlefont':{'size': 10}},
                        'yaxis': {'title': 'RECUENTO DE UNIDADES', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}}, 'showlegend':False, 'template': 'simple_white'}
        fig = go.Figure(data=hist, layout=layout)
        fig.update_yaxes(showgrid=True)
        fig.update_layout(margin=dict(l=0, b=0, t=30, r=0))
        median = x.median()
        fig.add_vline(x=median, line_width=2, line_color="#333", opacity=1,
            annotation_text=f"Mediana: {round(median,2)}%")
        return fig
    else:
        temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref)].copy()
        consumo_count = pd.DataFrame(temp['Consumo'].value_counts()).reset_index()
        tipo_consumo = list(consumo_count['index'].unique())
        bars = []
        ys = []
        for consumo, hor in zip(tipo_consumo, range(1,4)):
            if consumo == 'Normal':
                color = '#8ace7e'
            elif consumo == 'Excesivamente bajo':
                color = '#51b364'
            else:
                color = '#e03531'
            y = consumo_count[consumo_count['index'] == consumo]['Consumo'].max()
            ys.append(y)
            data = go.Bar(
                x= [hor],
                y = [y],
                text = [f'<b>{y}'],
                marker_color = color
            )
            bars.append(data)
        topy = max(ys) + max(ys)*0.5


        title = {'text': 'UNIDADES POR <b>TIPO DE CONSUMO</b>', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 10}
        layout = {'title':title, 'xaxis': {'title': 'TIPO DE CONSUMO', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}},
                        'yaxis': {'title': 'UNIDADES', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}, 'range': [0, topy]}, 'showlegend':False, 'template': 'simple_white'}


        fig = go.Figure(data=bars, layout=layout)
        fig.update_yaxes(showgrid=True)
        fig.update_traces(textposition='outside')
        fig.update_yaxes(showgrid=True)
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = list(np.arange(1,4,1)),
                ticktext = tipo_consumo
            ),
            margin=dict(l=0, b=0, t=30, r=0))
    
        return fig

# Filtrar Temps Refri

@app.callback(
        Output('refri_graph', 'figure'),
    [
        Input('ddl_x', 'start_date'),
        Input('ddl_x', 'end_date'),
        Input('ddl_y', 'value'),
        Input('ddl_z', 'value'),
        Input('ddl_w', 'value'),
        Input('ddl_t', 'value')
    ]
)

def graph(start_date, end_date, fam, ref, pos, sett):
    if fam != 'General':
        if sett == 'Ambas':
            temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Familia'] == fam)].copy()
        else:
            temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Familia'] == fam) & (df['Second Point Temp Setting (W/W or C/C)'] == sett)].copy()
        if pos == 'Primera':
            sensores = ['RC1 Temp °F', 'RC2 Temp °F', 'RC3 Temp °F']
        elif pos == 'Segunda':
            sensores = ['RC1 Temp °F 2nd P','RC2 Temp °F 2nd P', 'RC3 Temp °F 2nd P']
        puntos = []
        for sensor, hor in zip(sensores, range(1,4)):
            valores = temp[sensor].values
            for valor in valores:
                color, line, opacity = def_color('Refrigerador', valor)
                puntos.append(go.Scatter(
                    x = [hor],
                    y = [valor],
                    opacity=opacity,
                    mode = 'markers',
                    marker_color = color,
                    marker_size = 20,
                    marker_line_width = 2,
                    marker_line_color= line
                    ))
        title = {'text': 'TEMPERATURA POR SENSOR EN <b>REFRIGERADOR</b> ', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 10}
        layout = {'title':title, 'xaxis': {'title': 'SENSOR', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}},
                'yaxis': {'title': 'TEMPERATURA [°F]', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}}, 'showlegend':False, 'template': 'simple_white'}

        fig = go.Figure(data=puntos, layout=layout)
        fig.update_layout(
        xaxis = dict(
        tickmode = 'array',
        tickvals = list(np.arange(1,4,1)),
        ticktext = ['RC1', 'RC2', 'RC3']
        ),
        margin=dict(l=0, b=0, t=30, r=0))
        fig.update_yaxes(showgrid=True)
        fig.add_hrect(
                y0=36, y1=40, line_width=0, 
                fillcolor="#6aa342", opacity=0.2,
                annotation_text="Rango Óptimo",
                annotation_position="top left")
        fig.add_vline(x=1.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)
        fig.add_vline(x=2.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)
        
        return fig
                
    else:
        if sett == 'Ambas':
            temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref)].copy()
        else:
            temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Second Point Temp Setting (W/W or C/C)'] == sett)].copy()
        platforms = list(temp['Platform'].unique())
        familias = []
        for platform in platforms:
            fams = list(temp[temp['Platform'] == platform]['Familia'].unique())
            for fam in fams:
                familias.append(fam)

        puntos = []
        for familia, hor in zip(familias, range(10, 21, 1)):
            if pos == 'Primera':
                valores = temp[temp['Familia'] == familia]['RC Temp Average °F (M/M)'].sort_values(ascending=True).values
            elif pos == 'Segunda':
                valores = temp[temp['Familia'] == familia]['RC Temp Average °F (W/W or C/C)'].sort_values(ascending=True).values
            for valor in valores:
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

        title = {'text': 'TEMPERATURA PROMEDIO EN <b>REFRIGERADOR</b>', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 10}
        layout = {'title':title, 'xaxis': {'title': 'FAMILIA', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}},
                        'yaxis': {'title': 'TEMPERATURA [°F]', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}}, 'showlegend':False, 'template': 'simple_white'}
                
        fig = go.Figure(data=puntos, layout=layout)
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = list(np.arange(10,21,1)),
                ticktext = familias
            ),
            margin=dict(l=0, b=0, t=30, r=0))
        fig.update_yaxes(showgrid=True)
        fig.add_hrect(
                y0=36, y1=40, line_width=0, 
                fillcolor="#6aa342", opacity=0.2,
                annotation_text="Rango Óptimo",
                annotation_position="top left")

        fig.add_vline(x=10.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)
        fig.add_vline(x=16.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)

        return fig 

# Filtrar Temps Conge

@app.callback(
        Output('conge_graph', 'figure'),
    [
        Input('ddl_x', 'start_date'),
        Input('ddl_x', 'end_date'),
        Input('ddl_y', 'value'),
        Input('ddl_z', 'value'),
        Input('ddl_w', 'value'),
        Input('ddl_t', 'value')
    ]
)

def graph(start_date, end_date, fam, ref, pos, sett):
    if fam != 'General':
        if sett == 'Ambas':
            temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Familia'] == fam)].copy()
        else:
            temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Familia'] == fam) & (df['Second Point Temp Setting (W/W or C/C)'] == sett)].copy()
        if pos == 'Primera':
            sensores = ['FC1 Temp °F', 'FC2 Temp °F', 'FC3 Temp °F']
        elif pos == 'Segunda':
            sensores = ['FC1 Temp °F 2nd P','FC2 Temp °F 2nd P', 'FC3 Temp °F 2nd P']
        puntos = []
        for sensor, hor in zip(sensores, range(1,4)):
            valores = temp[sensor].values
            for valor in valores:
                color, line, opacity = def_color('Congelador', valor)
                puntos.append(go.Scatter(
                    x = [hor],
                    y = [valor],
                    opacity=opacity,
                    mode = 'markers',
                    marker_color = color,
                    marker_size = 20,
                    marker_line_width = 2,
                    marker_line_color= line
                    ))
        title = {'text': 'TEMPERATURA POR SENSOR EN <b>CONGELADOR</b> ', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 10}
        layout = {'title':title, 'xaxis': {'title': 'SENSOR', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}},
                'yaxis': {'title': 'TEMPERATURA [°F]', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}}, 'showlegend':False, 'template': 'simple_white'}

        fig = go.Figure(data=puntos, layout=layout)
        fig.update_layout(
        xaxis = dict(
        tickmode = 'array',
        tickvals = list(np.arange(1,4,1)),
        ticktext = ['FC1', 'FC2', 'FC3']
        ),
        margin=dict(l=0, b=0, t=30, r=0))
        fig.update_yaxes(showgrid=True)
        fig.add_hrect(
                y0=-10, y1=4, line_width=0, 
                fillcolor="#6aa342", opacity=0.2,
                annotation_text="Rango Óptimo",
                annotation_position="top left")
        fig.add_vline(x=1.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)
        fig.add_vline(x=2.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)
        
        return fig
                
    else:
        if sett == 'Ambas':
            temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref)].copy()
        else:
            temp = df[(df['Test Completion Date'] >= start_date) & (df['Test Completion Date'] <= end_date) & (df['Refrigerant'] == ref) & (df['Second Point Temp Setting (W/W or C/C)'] == sett)].copy()
        platforms = list(temp['Platform'].unique())
        familias = []
        for platform in platforms:
            fams = list(temp[temp['Platform'] == platform]['Familia'].unique())
            for fam in fams:
                familias.append(fam)

        puntos = []
        for familia, hor in zip(familias, range(10, 21, 1)):
            if pos == 'Primera':
                valores = temp[temp['Familia'] == familia]['FC Temp Average °F (M/M)'].sort_values(ascending=True).values
            elif pos == 'Segunda':
                valores = temp[temp['Familia'] == familia]['FC Temp Average °F (W/W or C/C)'].sort_values(ascending=True).values
            for valor in valores:
                if valor > 4:
                    color = '#dd6121'
                    line = '#bc4d23'
                    opacity= 1
                else:
                    color = '#fff'
                    line = '#333'
                    opacity = 0.7
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

        title = {'text': 'TEMPERATURAS PROMEDIO EN <b>CONGELADOR</b>', 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor': 'top', 'font_size': 10}
        layout = {'title':title, 'xaxis': {'title': 'FAMILIA', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}},
                        'yaxis': {'title': 'TEMPERATURA [°F]', 'tickfont': {'size': 10}, 'titlefont':{'size': 10}}, 'showlegend':False, 'template': 'simple_white'}
                
        fig = go.Figure(data=puntos, layout=layout)
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = list(np.arange(10,21,1)),
                ticktext = familias
            ),
            margin=dict(l=0, b=0, t=30, r=0))
        fig.update_yaxes(showgrid=True)
        fig.add_hrect(
                y0=-10, y1=4, line_width=0, 
                fillcolor="#6aa342", opacity=0.2,
                annotation_text="Rango Óptimo",
                annotation_position="top left")

        fig.add_vline(x=10.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)
        fig.add_vline(x=16.5, line_width=2, line_dash="dash", line_color="#333", opacity=1)

        return fig 

# RUN SERVER
if __name__ == '__main__':
    app.run_server(debug=True)