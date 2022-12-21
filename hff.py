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
col1, col2 = st.columns(2)

with col1:
   lat = st.number_input('latitude',value=41.1242)

with col2:
   lon = st.number_input('longitude',value=-101.3644337)
   T_water_C = st.number_input('water temperature (C)',value=2)

if st.button('Get Current Forecast'):
	df = get_full_forecast(lat,lon)

	T_water_C = 0
	q_sw, q_atm, q_b, q_l, q_h, q_net = calc_fluxes(df,T_water_C,lat,lon)

	energy_df = build_energy_df(q_sw, q_atm, q_b, q_l, q_h)
	fig = plot_forecast_heat_fluxes(energy_df)
	st.write(fig)

	g = plot_met(df)
	st.pyplot(g)