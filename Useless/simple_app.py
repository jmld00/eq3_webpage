import dash
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Hello, World!'),
    html.H2('Hello', style={'color':'red', 'fontSize': 14, 'font':'arial'}),
    html.H2('Hello2', style={'color':'red', 'fontSize': 14, 'font':'arial'})
])

if __name__ == '__main__':
	app.run_server(debug=True)