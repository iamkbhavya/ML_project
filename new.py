# model deployment
import streamlit as st
import numpy as np

#load model
model = pickle.load(open('model.pkl', 'rb'))
st.title('What is the Gold price?')
SPX = st.slider("Stocks price",0.1,5.8)
USO = st.slider("Oil price",0.1,5.8)
SLV = st.slider("Silver price",0.1,5.8)
EUR_USD = st.slider("Euro to USD conversion",0.1,5.8)

def predict():
    float_features = [float(x) for x in [SPX, USO, SLV, EUR_USD]]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    label = prediction[0]
    
    print(type(label))
    print(label)

    st.success('The Gold price is : ' + str(label) + ' :thumbsup:')
    
trigger = st.button('Predict', on_click=predict)
