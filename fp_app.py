import streamlit as st
import pandas as pd
import sklearn
import pickle
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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
state = st.selectbox(aqi['State'].unique().tolist())
filter_by_state = aqi[aqi['State'] == state]
with st.form("user_inputs"):
    county = st.selectbox(filter_by_state['County'].unique().tolist())
    CO_perc = st.number_input("Percentage of CO", min_value=0, max_value=1, value=0.1)
    NO2_perc = st.number_input("Percentage of NO2", min_value=0, max_value=1, value=0.1)
    PM25_perc = st.number_input("Percentage of PM2.5", min_value=0, max_value=1, value=0.1)
    PM10_perc = st.number_input("Percentage of PM10", min_value=0, max_value=1, value=0.1)
    O3_perc = st.number_input("Percentage of 3O", min_value=0, max_value=1, value=0.1)
    ml_model = st.selectbox("Select model",
                            options = ["Decision Tree", "Random Forest", "AdaBoost"],
                            placeholder = 'Choose an option')
    
    st.form_submit_button()

#Return row in original data to get demographics
user_df = aqi.loc[aqi["State"].isin([state]) & aqi['County'].isin([county])]
#Replace values with input values

encode_df = aqi.copy()
# Combine the list of user data as a row to default_df
encode_df = pd.concat([encode_df, user_df])
encode_df = encode_df.drop(['AQI'])
# Create dummies for encode_df
cat_var = ["State", "County"]
encode_dummy_df = pd.get_dummies(encode_df, columns = cat_var)
# Extract encoded user data
user_encoded_df = encode_dummy_df.tail(1)

if ml_model == "Decision Tree":
    # Using DT to predict() with encoded user data
    new_prediction_dt = dt_model.predict(user_encoded_df)
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