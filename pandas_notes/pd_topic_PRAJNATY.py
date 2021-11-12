#!/usr/bin/env python
# coding: utf-8

# ## Working with Python APIs
# 
# _Tianyu Jiang (prajnaty@umich.edu)_
# 
# ### Background knowledge: What is an application programming interface (API)
# 
# - An API is a set of defined rules that
# explain how computers or applications communicate with one another.
# 
# - APIs sit between an application and the web server, acting as an intermediary layer that processes data transfer between systems.
#     1. __A client application initiates an API call__ to retrieve information — also known as a request.
#     This request is processed from an application to the web server
#     via the API’s Uniform Resource Identifier (URI) 
#     and includes a request verb, headers, and sometimes, a request body.
#     1. After __receiving a valid request__,
#     the API __makes a call__ to the external program or web server.
#     1. The server sends __a response__
#     to the API with the requested information.
#     1. __The API transfers the data__ 
#     to the initial requesting application.
# 
# Ref: https://www.ibm.com/cloud/learn/api
# 
# ### Import libraries
# 1. __requests__ library helps us get the content 
# from the API by using the ```get()``` method.
# The ```json()``` method converts the API response to JSON format for easy handling.
# 1. __json__ library is needed so that we can work with the JSON content we get from the API.
# In this case, we get a dictionary for each Channel’s information such as name, id, views and other information.
# 1. __pandas__ library helps to create a dataframe
# which we can export to a .CSV file in correct format with proper headings and indexing.

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import requests
import json


# - Here I created a json dict,
# if you want to get json from web url, you need to use an API key.
#     - If you don't use an API key, you may get
#     [this](https://stackoverflow.com/questions/54783076/you-must-use-an-api-key-to-authenticate-each-request-to-google-maps-platform-api) error message.
#     - You may use ```requests.get(url).json()```
#     to get the response from the API for the url (in json format).
# 
# 
# - Reference: https://stackoverflow.com/questions/46578128/pandas-read-jsonjson-url

# In[2]:


json_dict = {
"message": "hello world",
"result": [{"id":12312312, "TimeStamp":"2017-10-04T17:39:53.92","Quantity":3.03046306,},
           {"id": 2342344, "TimeStamp":"2017-10-04T17:39:53.92","Quantity":3.03046306,}]
}

df = pd.json_normalize(json_dict['result'])

# Sanity check
print(df)


# In[3]:


# Process the timestamp and move it to front
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df = df.set_index('TimeStamp')

# Sanity check
print(df)


# ### Alternative approach
# 
# - Use ```pd.DataFrame(df['result'].values.tolist())```

# In[4]:


df = pd.read_json('https://bittrex.com/api/v1.1/public/getmarkethistory?market=BTC-ETC')
df = pd.DataFrame(df['result'].values.tolist())

# Process the timestamp and move it to front
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df = df.set_index('TimeStamp')

# Sanity check
print(df.head())

