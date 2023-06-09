# model deployment
import streamlit as st
import numpy as np
import pickle

#load model
model = pickle.load(open('model1.pkl', 'rb'))
st.title('What is the Gold price?')
SPX = st.slider("Stocks price",676.530029,2872.870117)
USO = st.slider("Oil price",7.960000,117.480003)
SLV = st.slider("Silver price",8.850000,47.259998)
EUR_USD = st.slider("Euro to USD conversion",1.039047,1.598798)

def predict():
    float_features = [float(x) for x in [SPX, USO, SLV, EUR_USD]]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    label = prediction[0]
    
    print(type(label))
    print(label)

    st.success('The Gold price is : ' + str(label) + ' :thumbsup:')
    
trigger = st.button('Predict', on_click=predict)
