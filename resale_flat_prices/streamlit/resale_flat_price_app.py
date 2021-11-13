import numpy as np

import streamlit as st

from io import BytesIO
import joblib
import requests

from resale_price_utils import *

st.markdown("## 2021 Singapore Resale Flat Prices")

st.markdown("##### Have you ever wondered how much a particular flat in Singapore might go on the resale market? Look no further! This AI powered streamlit app estimates the resale price using only 5 parameters!")

st.markdown("### Apartment Details")

# 1. User inputs to model.
# Town
town = st.selectbox("● Location of apartment.", options = sorted(median_price_town_rank.keys()))

# Flat type
flat_type = st.selectbox("● Type of apartment.", 
                         options = ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI GENERATION"],
                         index = 2)

# Storey range
storey_range = st.slider("● Apartment storey.", min_value = min_storey_range, max_value = max_storey_range,
                         value = 5)

# Floor area
floor_area = st.slider("● Floor area of apartment in square metres.", min_value = min_floor_area, max_value = max_floor_area,
                       value = 95)

# Age
age = st.slider("● Age of apartment in years.", min_value = min_age, max_value = max_age, value = 5)

# Pre-process the input data.
town = town_rank(town)
flat_type = flat_type_formatter(flat_type)
storey_range = storey_formatter(storey_range)
floor_area = floor_area_scaler(floor_area)
age = age_scaler(age)

X = np.array([[town, flat_type, storey_range, floor_area, age]])

# 2. Load model from GitHub. Not the ideal method, but should work for now.
# To-do: implement a more efficient method.

@st.cache
def load_model():  
    model_load_state = st.text("Loading price prediction AI...")
    mlink = "https://github.com/natsunoyuki/Data_Science/blob/master/resale_flat_prices/resale_price_lgb.pkl?raw=true"
    mfile = BytesIO(requests.get(mlink).content)
    model = joblib.load(mfile)
    model_load_state.text('AI loaded!')
    return model

model = load_model()

def get_prediction(model, X):
    y = y_descaler(model.predict(X))
    return y

if st.button("Get resale price"):
    resale_price = get_prediction(model, X)
    st.markdown("##### Estimated resale price: SGD {}.".format(resale_price.astype(int)[0]))
    
st.markdown("#### Links")
st.markdown('* <a href="https://github.com/natsunoyuki/blog_posts/blob/main/data_science/Singapore_resale_flat_prices.ipynb" target="_blank">Jupyter notebook on Github</a> explaining how I created the prediction model.', unsafe_allow_html = True)

st.markdown("#### Disclaimer")
st.markdown("The information contained in this website is for general information purposes only. While we endeavour to keep the information up to date and correct, we make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the website or the information contained on the website for any purpose. Any reliance you place on such information is therefore strictly at your own risk.")

st.markdown("In no event will we be liable for any loss or damage including without limitation, indirect or consequential loss or damage, or any loss or damage whatsoever arising from loss of data or profits arising out of, or in connection with, the use of this website.")

st.markdown("Every effort is made to keep the website up and running smoothly. However, we take no responsibility for, and will not be liable for, the website being temporarily unavailable due to technical issues beyond our control.")

st.markdown("Disclaimer text taken from https://www.nibusinessinfo.co.uk/content/sample-website-disclaimer.")