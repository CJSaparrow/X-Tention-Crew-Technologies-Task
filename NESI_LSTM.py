#!/usr/bin/env python
# coding: utf-8

# # Comprehensive Stock Analysis and Prediction Task

# #### Importing the libraries.

# In[44]:


import tensorflow as tf
from tensorflow import keras
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime

import warnings
warnings.filterwarnings('ignore')


# ## 1. Data Validation and Cleaning:
# ####    Loading dataset (NSEI.csv)
# 

# In[45]:


df = pd.read_csv('NSEI.csv')
df.info


# In[46]:


df.tail()


# In[47]:


df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index = df['Date']
df.tail()


# # 2. Stock Analysis:
# #### Statistical measures
# #### Exploratory Data Analysis (EDA)

# In[48]:


df.describe()


# In[49]:


dataset = pd.DataFrame(df[['Close']])
dataset.tail()


# #### Analyzing the field "close" of the dataset.

# In[50]:


plt.figure(figsize=(14,7))
plt.plot(dataset,label='Close Price history')


# In[51]:


dataset.info()


# In[52]:


dataset = dataset.dropna()
dataset.info()


# In[53]:


dataset.tail()


# ## 3. Predictive Modeling: (LSTM Model)
# #### Splitting data for training

# In[54]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)
scaled_data[:5]


# In[55]:


window_size = 50

x_data, y_data = [],[]

for i in range(window_size, len(scaled_data)):
    x_data.append(scaled_data[i-window_size:i,0])
    y_data.append(scaled_data[i,0])

x_data, y_data = np.array(x_data), np.array(y_data)

x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
y_data = np.reshape(y_data, (y_data.shape[0], 1))

x_data.shape, y_data.shape


# In[56]:


train_size = 0.8

x_train, x_test = x_data[:(int)(len(x_data)*train_size)], x_data[(int)(len(x_data)*train_size):]
y_train, y_test = y_data[:(int)(len(y_data)*train_size)], y_data[(int)(len(y_data)*train_size):]

y_original = dataset[window_size + len(y_train):]

x_train.shape, y_train.shape, x_test.shape, y_test.shape, y_original.shape


# ## 4. Deployment and Monitoring: (LSTM model)

# In[57]:


import keras
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

model=Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.1))
model.add(LSTM(units=50))
model.add(Dense(1))

model.summary()


# In[58]:


from keras.losses import mean_squared_error
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=60, batch_size=128, verbose=2)


# In[59]:


y_pred = model.predict(x_test)
print('Loss: ', sum(keras.losses.mean_squared_error(y_test, y_pred))/len(y_pred))


# In[60]:


closing_price = scaler.inverse_transform(y_pred)
print(closing_price[:5])


# In[62]:


validation = y_original
validation['Predictions'] = closing_price

plt.figure(figsize=(15,8))
plt.plot(validation['Close'], label="Actual Close Price")
plt.plot(validation['Predictions'], label="Predicted Close Price")
plt.title("Actual Close vs Predicted Close")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig("Output.png") #save the output graph in the png format


# In[ ]:





# In[ ]:



model.save('model.keras')

