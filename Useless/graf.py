import plotly 
import plotly.express as px
import pandas as pd
experience_dict = {'Programming Language': ['763254', 'A', 'Tableau', 'R', 'H71263', 'PowerShell'], 'Years of Experience (As of April 2022)': [8,4,3,2,1,1]}

fig = px.bar(experience_dict, x='Programming Language', y='Years of Experience (As of April 2022)',color_discrete_sequence=["black"])

fig.update_layout(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor = "rgba(0,0,0,0)",
    font_color = 'black',
    font_family = 'verdana',
    font_size = 20,
    )
fig.write_html("skills.html")
print(fig)