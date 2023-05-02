"""
Created on Wed Apr 12 16:58:27 2023

"""
#This project includes energy data from Sweden as follows
    #First tab of the dashboard contains the historical data of total energy consumption, energy consumption of different sectors
    #that were used to forecast future energy consumption based on the values from Swedish government's climate goals
    #As well as forecast of future energy consumption based on

    #The code includes part where data is tried to collect from Fingrid API, but unfortunately that never worked



import dash
from dash import html
import matplotlib.pyplot as plt
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import metrics
import numpy as np
import plotly.graph_objs as go
import geopandas
from shapely.geometry import Point
from datetime import datetime

# Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load data

#CO2 emission in Sweden
df_CO2 = pd.read_csv('Sweden co2 emissions.csv')
y1 = df_CO2['Fossil CO2 emissions (Tons)'].values
#df2 = df_real.iloc[:, [1,2,3]]
X1 = df_CO2.values
fig1 = px.line(df_CO2, x="year", y=df_CO2.columns[1])  # Creates a figure with the raw data

#Energy consumption by sector in Sweden
df_econsumption = pd.read_csv('Final_df.csv')
y2 = df_econsumption['Total Energy'].values
df_EC = df_econsumption.iloc[:, [1,2,3,4]]
X2 = df_EC.values
fig2 = px.line(df_econsumption, x="Year", y=df_EC.columns[0:4])  # Creates a figure with the raw data

#Energy consumption forecast based on the national climate goals. Source: Sweden’s draft integrated national energy and climate plan
df_eforecast = pd.read_csv('Energy consumption forecast.csv',delimiter=';', header=0)
df_eforecast = df_eforecast.set_index('Year', drop=True)


#Energy carriers in Sweden
df_real = pd.read_csv('Sweden consumption by fuel.csv')
y_Fuel = df_real['Total(TWh)'].values
df_Fuel = df_real.iloc[:, [2,6,7]]
X3 = df_Fuel.values
fig3 = px.line(df_real, x="year", y=df_real.columns[0:6])  # Creates a figure with the raw data

#Share of renewable energy of total energy
df_real = pd.read_csv('Sweden renewables share.csv')
y_share = df_real['Share of energy from renewable sources'].values
df_share = df_real
X4 = df_share.values
fig4 = px.line(df_real, x="year", y=df_real.columns[1])  # Creates a figure with the raw data


#Real time data from transmission loads between Sweeden and Finland
import requests

url = "https://api.fingrid.fi/v1/variable/87/events/latest"

headers = {
    "x-api-key": "S2a6uC3tud9XDuX26tPv87bxGVSdXy0934f7WQPd"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print("Error: Could not retrieve data.")


# Load RF model for energy consumption by sector
with open('RF_model.pkl', 'rb') as file:
    RF_model = pickle.load(file)

# Get the most recent value of total energy consumption
total_energy_consumption = df_econsumption['Total Energy'].iloc[-1]


# Initialize a list to store the predicted values
predicted_values = [total_energy_consumption]

# Loop through the next 9 years and make predictions
for i in range(2022, 2030):
    # Get the industry's energy consumption and energy consumption of domestic transport for the current year
    industry_energy_consumption = df_eforecast.loc[i, 'Industry']
    domestic_transport_energy_consumption = df_eforecast.loc[i, 'Domestic Transport']

    # Combine the features into a single array
    features = np.array([total_energy_consumption, industry_energy_consumption, domestic_transport_energy_consumption]).reshape(1, -1)

    # Use the model to make a prediction for the next year
    prediction = RF_model.predict(features)[0]

    # Store the predicted value
    predicted_values.append(prediction)

    # Update the most recent value of total energy consumption for the next iteration
    total_energy_consumption = prediction

# Combine the predicted values with the historical values
y_pred = np.array(predicted_values)
y_all = np.concatenate((y2, y_pred))

# Assume that 'y2' contains the historical data and 'predicted_values' contains the predicted data
years = list(range(1972, 2031))

# Create a dictionary of data that contains the years and the corresponding data
data = {'Year': years, 'Total Energy Consumption': y_all}

# Create a list of colors to differentiate historical and predicted data
colors = ['blue'] * len(y2) + ['red'] * len(predicted_values)


# Create a dataframe from the data dictionary and the color list
df = pd.DataFrame(data=data)
df['color'] = colors

# Create the figure
fig5 = px.line(df, x='Year', y='Total Energy Consumption', color='color')


# Plot the historical and predicted values
plt.plot(range(1972, 2022), y2, label='Historical')
plt.plot(range(2021, 2030), predicted_values, label='Predicted')

# Add a title and axis labels
plt.title('Total Energy Consumption by Sector')
plt.xlabel('Year')
plt.ylabel('Total Energy Consumption (TWh)')

# Add a legend
plt.legend()
plt.show()

# Load and run model for renewable energy production

with open('regrren.pkl', 'rb') as file:
    ren_model = pickle.load(file)

y2_pred_ren = ren_model.predict(X3)

df_pred_ren = pd.DataFrame(y2_pred_ren)



def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

# Define auxiliary functions


#ehkä poista suppress_callback_exceptions=True
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div(
        style={
            'background-image': 'url(https://media.istockphoto.com/id/1012202960/pt/foto/sun-above-the-solar-farm.jpg?s=1024x1024&w=is&k=20&c=dmM3NSmI0ymjfw2bL6dTB4Qbf9kCH1kYuNHfCjqBtmc=)',
            'background-size': 'cover',
            'background-position': 'center',
            'background-repeat': 'no-repeat'
        },
        children=[
            html.H1('Swedish Energy Data', style={'color': 'white', 'font-weight': 'bold'}),
            dcc.Tabs(
                id='tabs',
                value='tab-1',
                children=[
                    dcc.Tab(label='Energy consumption per sector', style={'color': '#0B0BA0','font-weight': 'bold'}, value='tab-1'),
                    dcc.Tab(label='Sweden co2 emissions', style={'color': '#0B0BA0','font-weight': 'bold'}, value='tab-2'),
                    dcc.Tab(label='Sweden consumption by fuel', style={'color': '#0B0BA0','font-weight': 'bold'}, value='tab-3'),
                    dcc.Tab(label='Share of energy from renewable sources', style={'color': '#0B0BA0','font-weight': 'bold'}, value='tab-4'),
                    dcc.Tab(label='Level of renewable electricity production', style={'color': '#0B0BA0','font-weight': 'bold'}, value='tab-5'),
                ]
            ),
        ]
    ),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('Energy consumption per sector', style={'color': '#FFA500', 'font-weight': 'bold'}),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
            ),
            html.H4('Total energy consumption forecast', style={'color': '#FFA500', 'font-weight': 'bold'}),
            dcc.Graph(
                id='renewable-data',
                figure=fig5,
            ),

        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('Sweden co2 emissions', style={'color': '#FFA500', 'font-weight': 'bold'}),
            dcc.Graph(
                id='yearly-data',
                figure=fig1,
            ),
        ])

    elif tab == 'tab-3':
        return html.Div([
            html.H4('Sweden consumption by fuel', style={'color': '#FFA500', 'font-weight': 'bold'}),
            dcc.Graph(
                id='yearly-data',
                figure=fig3,
            ),
        ])

    elif tab == 'tab-4':
        return html.Div([
            html.H4('Sweden renewables share', style={'color': '#FFA500', 'font-weight': 'bold'}),
            dcc.Graph(
                id='yearly-data',
                figure=fig4,
            ),

        ]),




if __name__ == '__main__':
    app.run_server()