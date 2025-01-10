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
from plots import *

def blurb():
    blurb_text = """

    **Downwelling SW (Shortwave Radiation):** This represents the heat flux from incoming solar radiation that reaches the river's surface. Its magnitude fluctuates daily with the solar cycle, peaking during midday when sunlight is strongest.

    **Downwelling LW (Longwave Radiation):** This flux captures the longwave radiation emitted by the atmosphere and surroundings toward the river. It tends to be relatively steady compared to shortwave radiation, influenced by cloud cover and atmospheric conditions.

    **Upwelling LW (Longwave Radiation):** The heat flux emitted from the river's surface back into the atmosphere. This depends on the river's surface temperature, with warmer water emitting more longwave radiation.

    **Sensible Heat:** The heat exchange between the river surface and the air due to differences in temperature. Positive values indicate heat transfer from the air to the river, while negative values indicate heat loss from the river to the air.

    **Latent Heat:** The heat exchange associated with water evaporation or condensation at the river's surface. Evaporation (heat loss) is typically the dominant process, driven by humidity and wind.

    **Net Flux:** The overall heat budget combining all the fluxes. A positive net flux indicates heat gain by the river, while a negative net flux indicates heat loss."""

    return blurb_text

st.set_page_config(page_title="Heat Flux Forecast", page_icon="🌡️", layout="wide")

st.title('6.5 Day Heat Flux Forecast')
st.write('Forecast met input data from NOAA Hourly Tabular Forecast Data')
st.write('Heat Flux forecast from CRREL HeatFluxForecast model')
st.write('Based on Water Quality Module calculations in HEC-RAS')

lat = st.sidebar.number_input('latitude', value=41.1242)
lon = st.sidebar.number_input('longitude', value=-101.3644337)
T_water_C = st.sidebar.number_input('water temperature (C)', value=2)
D = st.sidebar.number_input('characteristic depth (m)', value=2)

if st.sidebar.button('Compute Heat Fluxes'):

    df = get_full_forecast(lat, lon)
    df = df.replace([np.inf, -np.inf], np.nan)

    first_forecast_time = df.index[0]
    timezone = first_forecast_time.tz
    time_now = pd.Timestamp.now(tz=timezone)

    with st.expander("**Notes on Flux Terms**", expanded=False):
        st.write(blurb())

    #st.write(f'current timezone: {timezone}')
    st.write(f'Current Time: {time_now}')
    st.write(f'Current Forecast Start Time: {first_forecast_time}')
    if time_now - first_forecast_time > pd.Timedelta(hours=1):
        get_full_forecast.clear()
        df = get_full_forecast(lat, lon)

    q_sw, q_atm, q_b, q_l, q_h, q_net = calc_fluxes(df, T_water_C, lat, lon)

    energy_df = build_energy_df(q_sw, q_atm, q_b, q_l, q_h)
    fig = plot_forecast_heat_fluxes(energy_df)
    st.plotly_chart(fig, use_container_width=True)

    g = plot_met(df)
    st.plotly_chart(g, use_container_width=True)

    cooling_rate = calc_cooling_rate(q_net, D)

    with st.expander("Experimental Plots", expanded=False):
        st.write(plot_cooling_rate(cooling_rate))
        st.write(plot_parcel_cooling(cooling_rate, T_water_C))


