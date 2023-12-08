import streamlit as st
import pandas as pd
import math
import sklearn
import pickle
import numpy as np
from matplotlib import pyplot as plt
import pydeck as pdk
import warnings
warnings.filterwarnings('ignore')


#insert introduction to research question
st.title("The Anthropogenic Effects on Air Quality")
st.subheader("What human qualities can impact air quality the most?")
st.write("The Air Quality Index (AQI) is a numerical scale that communicates the "
         "level of air pollution in a specific area, assessing the concentration of "
         "pollutants such as particulate matter (PM2.5 and PM10), ground-level "
         "ozone, carbon monoxide, sulfur dioxide, and nitrogen dioxide. The AQI " 
         "provides a standardized way to convey the potential health impacts associated "
         "with different pollution levels, ranging from Good to Hazardous. Human "
         "activities significantly contribute to air quality degradation through "
         "the release of pollutants from sources like vehicle emissions, industrial processes"
         " and burning of fossil fuels. Exposure to poor air quality can have severe health "
         "implications, causing respiratory and cardiovascular problems, exacerbating pre-existing "
         "conditions, and even leading to premature death. Therefore, understanding and addressing human "
         "impacts on air quality are crucial for safeguarding public health and the environment.")
st.image("heatmap_aqi.png")

#heat maps from exploratory data analysis
st.subheader("Pollutant Parameter Distributions")
tab1, tab2, tab3 = st.tabs(["Ozone", "PM10", "PM2.5"])
with tab1:
    st.image('heatmap_ozone.png')
with tab2:
    st.image('heatmap_pm10.png')
with tab3:
    st.image('heatmap_pm25.png')
        
st.subheader("Demographic Distributions")
tab1, tab2, tab3, tab4 = st.tabs(["Black Population Percentages", "Hispanic Population Percentages", 
                            "Urban Population Percentages", "Average Salary"])
with tab1:
    st.image('heatmap_black.png')
with tab2:
    st.image('heatmap_his.png')
with tab3:
    st.image('heatmap_urban.png')
with tab4:
    st.image('heatmap_salary.png')

st.subheader("Utilize our advanced machine learning app to predict"
             " air quality index.")


# Reading the pickle files that we created before 
# Decision Tree
dt_pickle = open('dt_aqi.pickle', 'rb') 
dt_model = pickle.load(dt_pickle) 
dt_pickle.close()

#AdaBoost
ad_pickle = open('ad_aqi.pickle', 'rb') 
ad_model = pickle.load(ad_pickle) 
ad_pickle.close()

# Random Forest
rf_pickle = open('rf_aqi.pickle', 'rb') 
rf_model = pickle.load(rf_pickle) 
rf_pickle.close()

# Loading default dataset
aqi = pd.read_csv('mlm_aqi_data.csv')

# User input form
state = st.selectbox("Select state", options = aqi['State'].unique().tolist())
filter_by_state = aqi[aqi['State'] == state]

with st.form("user_inputs"):
    county = st.selectbox("Select county", options = filter_by_state['County'].unique().tolist())
    CO_perc = st.number_input("Percentage of CO", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    NO2_perc = st.number_input("Percentage of NO2", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    PM25_perc = st.number_input("Percentage of PM2.5", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    PM10_perc = st.number_input("Percentage of PM10", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    O3_perc = st.number_input("Percentage of O3", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    ml_model = st.selectbox("Select model",
                            options = ["Decision Tree", "Random Forest", "AdaBoost"],
                            placeholder = 'Choose an option')
    st.form_submit_button()

#Return row in original data to get demographics
user_df = aqi.loc[aqi["State"].isin([state]) & aqi['County'].isin([county])]
#Replace values with input values
user_df.loc[user_df['CO_perc']!= CO_perc, 'CO_perc'] = CO_perc
user_df.loc[user_df['NO2_perc']!= NO2_perc, 'NO2_perc'] = NO2_perc
user_df.loc[user_df['O3_perc']!= O3_perc, 'O3_perc'] = O3_perc
user_df.loc[user_df['PM2.5_perc']!= PM25_perc, 'PM2.5_perc'] = PM25_perc
user_df.loc[user_df['PM10_perc']!= PM10_perc, 'PM10_perc'] = PM10_perc

st.dataframe(user_df)

encode_df = aqi.copy()
# Combine the list of user data as a row to default_df
encode_df = pd.concat([encode_df, user_df])
encode_df = encode_df.drop(columns=['AQI'])
# Create dummies for encode_df
cat_var = ["State", "County"]
encode_dummy_df = pd.get_dummies(encode_df, columns = cat_var)
# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

if ml_model == "Decision Tree":
    # Using DT to predict() with encoded user data
    new_prediction_dt = dt_model.predict(user_encoded_df)
    #new_prediction_dt = math.floor(new_prediction_dt)
    # Show the predicted cost range on the app
    st.write("Decision Tree AQI Prediction: {}".format(*new_prediction_dt))
    # showing additional feature importance plot
    st.subheader("Feature Importance")
    st.image("dt_feature_imp.svg")
    
elif ml_model == "AdaBoost":
    # Using AdaBoost to predict() with encoded user data
    new_prediction_dt = ad_model.predict(user_encoded_df)
    # Show the predicted cost range on the app
    st.write("AdaBoost AQI Prediction: {}".format(*new_prediction_dt))
    # showing additional feature importance plot
    st.subheader("Feature Importance")
    st.image("ad_feature_imp.svg")

else:
    # Using RF to predict() with encoded user data
    new_prediction_rf = rf_model.predict(user_encoded_df)
    # Show the predicted cost range on the app
    st.write("Random Forest AQI Prediction: {}".format(*new_prediction_rf))
    # showing additional feature importance plot
    st.subheader("Feature Importance")
    st.image("rf_feature_imp.svg")