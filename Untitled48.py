#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn import metrics
import matplotlib.pyplot as plt


# In[4]:


gold_data=pd.read_csv('C:/download/archive/gld_price_data.csv')


# In[5]:


# Separating the dependent and independent variables
X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']


# In[6]:


print("The independent features in the dataset are:-")
print(X)


# In[7]:


print("The dependent features in the dataset is:-")
print(Y)


# In[8]:


# Splitting into Training data and Test Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)


# In[9]:


# Model training: Linear Regression
lr_model = LinearRegression() 
lr_model.fit(X_train,Y_train)
lr_pred = lr_model.predict(X_test)
error_score = metrics.r2_score(Y_test,lr_pred)
print("R squared score for Linear regression is : ", error_score)


# In[10]:


# Model training: Decision Tree
dt_model=DecisionTreeRegressor()
dt_model.fit(X_train,Y_train)
dt_pred = dt_model.predict(X_test)
error_score = metrics.r2_score(Y_test,dt_pred)
print("R squared score for Decision tree is : ", error_score)


# In[11]:


# Model training: KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train,Y_train)
knn_pred = knn_model.predict(X_test)
error_score = metrics.r2_score(Y_test,knn_pred)
print("R squared score for KNN is : ", error_score)


# In[12]:


# Model training: Random Forest
rf_model=RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train,Y_train)
rf_pred = rf_model.predict(X_test)
error_score = metrics.r2_score(Y_test,rf_pred)
print("R squared score for Random forest is : ", error_score)


# In[13]:


# Actual and predicted values plot for Linear Regression
Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(lr_pred, color='green', label='Predicted Value')
plt.title('Actual Price vs Predicted Price for Linear Regression')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[14]:


# Actual and predicted values plot for Decision Tree
Y_test = list(Y_test)
plt.plot(Y_test, color='red', label = 'Actual Value')
plt.plot(dt_pred, color='yellow', label='Predicted Value')
plt.title('Actual Price vs Predicted Price for Decision Tree')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[15]:


# Actual and predicted values plot for KNN
Y_test = list(Y_test)
plt.plot(Y_test, color='blue', label = 'Actual Value')
plt.plot(knn_pred, color='grey', label='Predicted Value')
plt.title('Actual Price vs Predicted Price for KNN')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[16]:


# Actual and predicted values plot for Random forest
Y_test = list(Y_test)
plt.plot(Y_test, color='violet', label = 'Actual Value')
plt.plot(rf_pred, color='brown', label='Predicted Value')
plt.title('Actual Price vs Predicted Price for Random forest')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()


# In[79]:


# Among all the models that we have used Random forest has given us the best possible result
# So, now we will apply hyperperameter tuning to check weather it's accuracy can be improved further or not.

from sklearn.model_selection import GridSearchCV


param_grid = {
'n_estimators': [50, 100, 200],
'max_depth': [5, 10, 20, 30],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4]
}
from joblib import parallel_backend

with parallel_backend('threading', n_jobs=10):

    rf = RandomForestRegressor()
    rf_tuned = GridSearchCV(rf, param_grid, cv=5)
    rf_tuned.fit(X_train, Y_train)
    best_params = rf_tuned.best_params_
    print(f"Best Parameters: {best_params}")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


rf_tuned = RandomForestRegressor(**best_params)
rf_tuned.fit(X_train, Y_train)
Y_pred = rf_tuned.predict(X_test)
r2 = metrics.r2_score(Y_test, y_pred)
print(f"Tuned Random Forest:")
print(f"R squared value: {r2}")


# In[17]:


# We can see from the rbove r squared value that after hyperperameter tuning the accuracy of random forest has decreased.
# So, we will use the older version of random forest for our model deployment


# In[18]:


# Now we will save our random forest model into a file that we will use for model deployment
import pickle
fileName="model.pkl"
# save model
pickle.dump(rf_model, open(fileName, "wb"))


# In[19]:


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
    float_features = [float(x) for x in [SPX, USO, SLV, EUR/USD]]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    label = prediction[0]
    
    print(type(label))
    print(label)

    st.success('The Gold price is : ' + str(label) + ' :thumbsup:')
    
trigger = st.button('Predict', on_click=predict)


# In[20]:





# In[ ]:




