import pandas as pd
import numpy as np
import streamlit as st
import pickle as pk

model = pk.load(open(r'C:\Users\Utkarsh kumar\Desktop\Abes\project_made\House_Price_Prediction\House_prediction.model.pkl', 'rb'))

st.header('Real Estate House Price Prediction - Bangalore ')

data=pd.read_csv(r'C:\Users\Utkarsh kumar\Desktop\Abes\project_made\House_Price_Prediction\Cleaned_Data.csv')

locate=st.selectbox('Choose The Location ',data['location'])
sqft=st.number_input('Enter the total sq ft of the house')
bathrooms=st.number_input('Enter the total number of the bathrooms')
balconies=st.number_input('Enter the total number of the balconies')
bedrooms=st.number_input('Enter the total sq ft of the bedrooms')

input_df=pd.DataFrame([[locate,sqft,bathrooms,balconies,bedrooms]],columns=['location','total_sqft','bath','balcony','bedrooms'])

if st.button('Predict House Price'):
    prediction = model.predict(input_df)
    st.success(f'🏡 Estimated House Price: ₹ {prediction[0]:,.2f} lakhs')
