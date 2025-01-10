# -------------------------------------------------------------------------------
# Name          Heat Flux Plots
# Description:  Collection of plotting functions
# Author:       Chandler Engel
#               US Army Corps of Engineers
#               Cold Regions Research and Engineering Laboratory (CRREL)
#               Chandler.S.Engel@usace.army.mil
# Created:      27 December 2024
# Updated:       - 
#
# -------------------------------------------------------------------------------

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast_heat_fluxes(energy_df):
    """
    Create an interactive Plotly line plot of heat fluxes, highlighting 'net flux'
    in bold black.
    """
    # Convert wide DataFrame to long-form
    energy_long = pd.melt(energy_df.reset_index(), id_vars='index')

    # Create the Plotly Express line chart
    fig = px.line(
        energy_long,
        x='index',
        y='value',
        color='variable',
        color_discrete_sequence=px.colors.qualitative.Bold,
        # If you want consistent coloring per variable, you could specify a dict:
        # color_discrete_map={
        #     'downwelling SW': 'blue',
        #     'downwelling LW': 'orange',
        #     'upwelling LW': 'green',
        #     'sensible heat': 'red',
        #     'latent heat': 'purple',
        #     'net flux': 'black'
        # },
    )

    # Make the 'net flux' line thicker and black
    for trace in fig.data:
        if trace.name == 'net flux':
            trace.line.width = 3
            trace.line.color = 'black'
        else:
            # Optionally make other lines thinner or semi-transparent
            trace.line.width = 2
            trace.opacity = 0.8

    # Customize layout
    fig.update_layout(
        title='Forecast Heat Fluxes',
        xaxis_title='',
        yaxis_title='Heat Flux (W/m²)',
        legend_title_text='Flux Type',
        template='plotly_white'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig

# def plot_met(df):
#     # Prepare the data for plotting
#     columns = df.columns
#     df_met = df[[columns[0], columns[3], columns[6], columns[8]]]
#     df_met = df_met.rename(columns={columns[0]: 'Temperature F'})
#     df_met['Temperature C'] = (df_met['Temperature F'] - 32) * (5 / 9)
#     df_met['windspeed ms'] = df_met['Surface Wind (mph)'] * 0.44704
#     df_met = df_met.drop(['Temperature F', 'Surface Wind (mph)'], axis=1)
#     df_met = pd.melt(df_met.reset_index(), id_vars='date')
#     df_met = df_met.rename(columns={0: 'variable'})

#     # Create a Plotly figure
#     fig = go.Figure()

#     # Add traces for each variable
#     variables = df_met['variable'].unique()
#     colors = ['blue', 'orange', 'green', 'red']  # Set distinct colors for each variable

#     # Create a subplot for each variable
#     fig = make_subplots(
#         rows=len(variables),
#         cols=1,
#         shared_xaxes=True,
#         vertical_spacing=0.02,
#         subplot_titles=[var for var in variables]
#     )
    
#     # Add a line trace for each variable on its own subplot row
#     for i, (var, color) in enumerate(zip(variables, colors), start=1):
#         subset = df_met[df_met['variable'] == var]
#         fig.add_trace(
#             go.Scatter(
#                 x=subset['date'], 
#                 y=subset['value'], 
#                 mode='lines', 
#                 name=var, 
#                 line=dict(color=color)
#             ),
#             row=i,
#             col=1
#         )

#     # Update layout
#     fig.update_layout(
#         height=300 * len(variables),  # Scale the height if you want
#         title="Meteorological Data",
#         xaxis_title="Date",
#         showlegend=False,  # Legend can be hidden since each subplot is labeled
#         template="plotly_white"
#     )

#     # Show grid lines for all subplots
#     for i in range(len(variables)):
#         fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i+1, col=1)
#         fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i+1, col=1)

#     return fig

def plot_met(df):
    # Prepare the data for plotting
    columns = df.columns
    df_met = df[[columns[0], columns[3], columns[6], columns[8]]]
    df_met = df_met.rename(columns={columns[0]: 'Temperature F'})
    df_met['Temperature C'] = (df_met['Temperature F'] - 32) * (5 / 9)
    df_met['windspeed ms'] = df_met['Surface Wind (mph)'] * 0.44704
    df_met = df_met.drop(['Temperature F', 'Surface Wind (mph)'], axis=1)
    df_met = pd.melt(df_met.reset_index(), id_vars='date')
    df_met = df_met.rename(columns={0: 'variable'})

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for each variable
    variables = df_met['variable'].unique()
    colors = ['blue', 'orange', 'green', 'red']  # Set distinct colors for each variable

    # Create subplots: one row per variable
    fig = make_subplots(
        rows=len(variables),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[var for var in variables]
    )
    
    # Add a line trace for each variable on its own subplot row
    for i, (var, color) in enumerate(zip(variables, colors), start=1):
        subset = df_met[df_met['variable'] == var]
        fig.add_trace(
            go.Scatter(
                x=subset['date'], 
                y=subset['value'], 
                mode='lines', 
                name=var, 
                line=dict(color=color)
            ),
            row=i,
            col=1
        )

        fig.add_shape(
        type='rect',
        xref='x domain', x0=0, x1=1,
        yref='y domain', y0=0, y1=1,
        line=dict(color='black', width=2),
        row=i, col=1
)

        # If this variable is "Temperature C", add a horizontal line at y=0
        if var == "Temperature C":
            fig.add_hline(
                y=0,
                line_color="black",
                row=i,          # Must match the row of the Temperature C subplot
                col=1           # Only one column
            )

    # Update layout (overall figure settings)
    fig.update_layout(
        height=300 * len(variables),  # Adjust the height if you want
        title="Meteorological Data",
        showlegend=True,             # Each subplot is labeled, so legend can be hidden
        template="plotly_white"
    )

    fig.update_xaxes(
        title_text="Date",
        row=len(variables), 
        col=1
    )

    fig.update_layout(
        font=dict(color='black'),               # sets default text color (annotations, axis labels, etc.)
        title_font=dict(color='black', size=16) # sets figure title style
    )

    # Show grid lines for all subplots
    for i in range(len(variables)):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i+1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', row=i+1, col=1)

    fig.update_xaxes(tickfont=dict(color='black'), title_font=dict(color='black'))
    fig.update_yaxes(tickfont=dict(color='black'), title_font=dict(color='black'))

    return fig

# def plot_met(df):
#     # Prepare the data for plotting
#     columns = df.columns
#     df_met = df[[columns[0], columns[3], columns[6], columns[8]]]
#     df_met = df_met.rename(columns={columns[0]: 'Temperature F'})
#     df_met['Temperature C'] = (df_met['Temperature F'] - 32) * (5 / 9)
#     df_met['windspeed ms'] = df_met['Surface Wind (mph)'] * 0.44704
#     df_met = df_met.drop(['Temperature F', 'Surface Wind (mph)'], axis=1)
#     df_met = pd.melt(df_met.reset_index(), id_vars='date')
#     df_met = df_met.rename(columns={0: 'variable'})

#     # Create a Plotly figure
#     fig = go.Figure()

#     # Add traces for each variable
#     variables = df_met['variable'].unique()
#     colors = ['blue', 'orange', 'green', 'red']  # Set distinct colors for each variable

#     for var, color in zip(variables, colors):
#         subset = df_met[df_met['variable'] == var]
#         fig.add_trace(go.Scatter(
#             x=subset['date'],
#             y=subset['value'],
#             mode='lines',
#             name=var,
#             line=dict(color=color)
#         ))

#     # Customize layout
#     fig.update_layout(
#         title="Meteorological Data",
#         xaxis_title="Date",
#         yaxis_title="Value",
#         legend_title="Variable",
#         template="plotly_white"
#     )

#     fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
#     fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

#     return fig

def plot_cooling_rate(cooling_rate):
    fig, ax = plt.subplots(figsize=(15, 5))
    ax = sns.lineplot(cooling_rate)
    ax.axhline(-1.29 * 10 ** -3, linestyle='--', color='k')
    plt.ylabel('Cooling Rate (C/min)', fontsize=12)
    return fig


def plot_parcel_cooling(cooling_rate, T_water_C):
    cooling_rate_hr = cooling_rate * 60

    temps = pd.DataFrame(cooling_rate_hr)
    cooling_cumsum = cooling_rate_hr.cumsum()
    for i in range(100):
        c = cooling_cumsum - cooling_cumsum[i]
        c[0:i] = 0
        temps[i] = c
    temps = temps + T_water_C
    temps[temps < 0] = 0
    temps = pd.melt(temps.reset_index(), id_vars='index')
    fig, ax = plt.subplots(figsize=(15, 5))
    ax = sns.lineplot(data=temps, x="index", y="value", hue='variable')
    plt.ylabel('Water Temp (C)', fontsize=12)
    return fig