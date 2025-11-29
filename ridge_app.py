import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk


model = pk.load(open(r'C:\Users\Utkarsh kumar\Desktop\Abes\project_made\House_Price_Prediction\house_ridge_model.pkl', 'rb'))


st.header('🏠 Real Estate House Price Prediction - Bangalore')


data = pd.read_csv(r'C:\Users\Utkarsh kumar\Desktop\Abes\project_made\House_Price_Prediction\Final_housing_data.csv')


locate = st.selectbox('📍 Choose the Location', data['location'].unique())
sqft = st.number_input('📏 Enter the total sq ft of the house', min_value=100.0, value=1000.0)
bathrooms = st.number_input('🚿 Enter the total number of bathrooms', min_value=1, max_value=10, value=2, step=1)
# balconies = st.number_input('🌤️ Enter the total number of balconies', min_value=0, max_value=5, value=1, step=1)
bedrooms = st.number_input('🛏️ Enter the total number of bedrooms', min_value=1, max_value=10, value=2, step=1)


input_df = pd.DataFrame([[locate, sqft, bathrooms, bedrooms]], 
                        columns=['location', 'total_sqft', 'bath', 'bhk'])


if st.button('🔍 Predict House Price'):
    output = model.predict(input_df)[0]
    output = max(output, 0)  
    
    st.success(f'🏡 Estimated House Price: ₹ {output:.2f} lakhs')
