import requests
import pandas as pd
from xml.etree import ElementTree as ET
from datetime import datetime

def get_digital_dwml_forecast(lat, lon):
     # URL for DWML format
    url = f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=digitalDWML"

    # Fetch the DWML data
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data. HTTP status: {response.status_code}")

    # Parse the XML content
    root = ET.fromstring(response.content)

    # Extract time-layouts
    time_layouts = {}
    for layout in root.findall(".//time-layout"):
        layout_key = layout.find("layout-key").text
        times = [
            datetime.fromisoformat(t.text) for t in layout.findall("start-valid-time")
        ]
        time_layouts[layout_key] = times

    # Helper function to extract data for a parameter
    def extract_parameter_data(tag_name, attribute_value=None):
        param = root.find(f".//{tag_name}[@type='{attribute_value}']")
        if param is None:
            print(f"Missing parameter: {tag_name} with attribute '{attribute_value}'")
            return None, []
        layout_key = param.find("time-layout")
        if layout_key is None or layout_key.text is None:
            print(f"Missing time-layout for {tag_name} with attribute '{attribute_value}'")
            return None, []
        values = [v.text for v in param.findall("value")]
        return layout_key.text, values

    # Extract weather data
    data = {}
    for param, attr, label in [
        ("temperature", "hourly", "Temperature (F)"),
        ("temperature", "dew point", "Dewpoint (F)"),
        ("wind-speed", "sustained", "Wind Speed (mph)"),
        ("cloud-amount", "total", "Sky Cover (%)"),
    ]:
        layout_key, values = extract_parameter_data(param, attr)
        if layout_key and layout_key in time_layouts:
            timestamps = time_layouts[layout_key]
            data[label] = pd.Series(
                [int(v) if v else None for v in values], index=timestamps
            )

    # Combine all parameters into a DataFrame
    df = pd.DataFrame(data)
    df.index.name = "datetime"

    return df