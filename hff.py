# -------------------------------------------------------------------------------
# Name          Heat Flux App Frontend
# Description:  Streamlit app to display results of heat flux
#				calculations using NWS hourly forecast data
# Author:       Chandler Engel
#               US Army Corps of Engineers
#               Cold Regions Research and Engineering Laboratory (CRREL)
#               Chandler.S.Engel@usace.army.mil
# Created:      20 December 2022
# Updated:      -
#               
# --

import streamlit as st
import numpy as np
import pandas as pd
from pvlib.location import Location
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from utils import *

st.set_page_config(layout="wide")

st.title('6.5 Day Heat Flux Forecast')
st.write('Forecast met input data from NOAA Hourly Tabular Forecast Data')
st.write('Heat Flux forecast from CRREL HeatFluxForecast model')

lat = st.sidebar.number_input('latitude', value=41.1242)
lon = st.sidebar.number_input('longitude', value=-101.3644337)
T_water_C = st.sidebar.number_input('water temperature (C)', value=2)
D = st.sidebar.number_input('characteristic depth (m)', value=2)

if st.sidebar.button('Compute Heat Fluxes'):

    df = get_full_forecast(lat, lon)

    first_forecast_time = df.index[0]
    timezone = first_forecast_time.tz
    time_now = pd.Timestamp.now(tz=timezone)

    #st.write(f'current timezone: {timezone}')
    st.write(f'Current Time: {time_now}')
    st.write(f'Current Forecast Start Time: {first_forecast_time}')
    if time_now - first_forecast_time > pd.Timedelta(hours=1):
        get_full_forecast.clear()
        df = get_full_forecast(lat, lon)

    q_sw, q_atm, q_b, q_l, q_h, q_net = calc_fluxes(df, T_water_C, lat, lon)

    energy_df = build_energy_df(q_sw, q_atm, q_b, q_l, q_h)
    fig = plot_forecast_heat_fluxes(energy_df)
    st.write(fig)

    g = plot_met(df)
    st.pyplot(g)

    cooling_rate = calc_cooling_rate(q_net, D)

    st.write(plot_cooling_rate(cooling_rate))
    st.write(plot_parcel_cooling(cooling_rate, T_water_C))
