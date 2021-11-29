#!/usr/bin/env python
# coding: utf-8


# ## Topics in Pandas
# **Stats 507, Fall 2021**
#

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using
# the exact title with spaces replaced by a dash.
#
# + [Topic Title](#Topic-Title)
# + [Topic 2 Title](#Topic-2-Title)
# + [Data Cleaning](#Data-Cleaning)

# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide.

# ## Data Cleaning
# **Jiaxi Chen**

# References: 
# * https://www.w3schools.com/python/pandas/pandas_cleaning.asp
# * https://pandas.pydata.org/docs/reference/index.html
#
#
# ### Finding Duplicates
# - Duplicate rows are rows that have been registered more than one time.
#
# - `pandas.DataFrame.duplicated` returns boolean series denoting duplicate rows.
# - `keep` determines which duplicates (if any) to mark.
#     - `keep = False` : Mark all duplicates as True.
#     - `keep = first` : Mark duplicates as True except for the first occurrence.
#     - `keep = last` : Mark duplicates as True except for the last occurrence.

import pandas as pd
import numpy as np
from pydataset import data

df = data('iris')
df.head()
print(df[df.duplicated(keep=False)])
print(df[df.duplicated(keep='first')])

# ### Removing Duplicates
# - We can use the `drop_duplicates()` method to remove duplicates
# - The `inplace = True` will make sure that the method does NOT return
# a new DataFrame, but it will remove all duplicates from the original
# DataFrame.

print(df.shape)
df.drop_duplicates(inplace = True)
print(df.shape)

# ### Finding NaN
# - Check for NaN under a single DataFrame column:
# `df['your column name'].isnull().values.any()`
# - Count the NaN under a single DataFrame column:
# `df['your column name'].isnull().sum()`
# - Check for NaN under an entire DataFrame: `df.isnull().values.any()`
# - Count the NaN under an entire DataFrame: `df.isnull().sum().sum()`

data = {'set_of_numbers': [1,2,3,4,5,np.nan,6,7,np.nan,8,9,10,np.nan]}
df = pd.DataFrame(data)
check_for_nan = df['set_of_numbers'].isnull()
print (check_for_nan)
nan_count = df['set_of_numbers'].isnull().sum()
print (nan_count)
nan_exist = df['set_of_numbers'].isnull().values.any()
print(nan_exist)

# ### Replacing NaN with mean, median or mode
# - A common way to replace empty cells, is to calculate the mean, median or mode value of the column.
# - Pandas uses the `mean()` `median()` and `mode()` methods to calculate the respective values for a specified column

x = df["set_of_numbers"].mean()
df["set_of_numbers"].fillna(x, inplace = True)
print(df)
nan_count = df['set_of_numbers'].isnull().sum()
print (nan_count)


# In[ ]:


############################ pd_topic_group2_4b_berlands ##############################

# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Topic Title](#Topic-Title) 
# + [Topic 2 Title](#Topic-2-Title)
# + [DataFrame Method: `select_dtypes()`](#DataFrame-Method:-`select_dtypes()`) 

# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide. 

# ## DataFrame Method: `select_dtypes()`
# *Brody Erlandson*
# berlands@umich.edu
#
# We can select the columns with a specific type(s), and/or exclude columns
# with a specific type(s).
#   
# - Parameters:
#   - include: scalar or list-like. Default None.
#   - exclude: scalar or list-like. Default None.
# - Returns: DataFrame

# ## `select_dtypes()` Example

df = pd.DataFrame({"strings1" : ["a", "b", "c"], "ints" : [1, 2, 3],
                   "floats" : [.1, .2, .3], 
                   "category" : ["cat1", "cat1", "cat2"],
                   "string2" : ["x", "y", "z"]})
df = df.convert_dtypes()
df["category"] = df["category"].astype("category")
df.select_dtypes(include=["string", "category"])

# Similarly:

df.select_dtypes(exclude=[float, int])

# ## When to use `select_dtypes()`
#
# Say you have a lot of columns of different types. You'd like to apply some
# function to only one type. Instead of finding all the indices or names
# of the columns, we can use `select_dtypes()` to get these columns.  
# 
# For example:   

df.select_dtypes(include="string").apply(lambda x : x + "_added")


# In[ ]:


############################ pd_topic_group2_3e_yangyli ##############################


########################### pd_topic_group2_4j_aayushi #############################
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021**
#

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using
# the exact title with spaces replaced by a dash.
#
# + [Topic Title](#Topic-Title)
# + [Topic 2 Title](#Topic-2-Title)

# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide.


# Aayushi Sinha
#
# aayushi@umich.edu
#
#
# ##  Timedeltas in Pandas

# ### Class pd.Timedelta
#
# * It represents a duration of time or date. Easy to use for date or time manipulations.
# * It is a similar to the python class of datetime.timedelta.
# * It is expressed in difference units, e.g. days, hours, minutes, seconds.

# In[1]:


import pandas as pd
from datetime import datetime, timedelta

#Calculating the number of days between two dates.
present_date = datetime.now()
print ("Present Date", str(present_date))
future_date = present_date + timedelta(days = 6)
print("Future Date", str(future_date) )

print("Days left", str(present_date - future_date))


# ### Parsing
# * Prasing through a string or an integer with units in the argument can create a timedelta object.
# * pd.to_timedelta can convert a scalar, array, list, or series from a recognized timedelta
#   format/ value into a Timedelta type.

# In[3]:


#Parsing and to_timedelta
print (pd.Timedelta('2 days 2 hours 15 minutes 30 seconds'))
#Converting to timedelta object
pd.to_timedelta("1 days 06:05:01.00003")


# ### Operations
#
# * You can construct timedelta64[ns] Series/Dataframes through subtraction operations and operate on the series/dataframes.
# * NaT values are supported in timedelta series.
# * min, max and the corresponding idxmin, idxmax operations are supported. Even negate can be used.

# In[5]:


# Operation Examples
s = pd.Series(pd.date_range('2012-1-1', periods=3, freq='D'))
td = pd.Series([ pd.Timedelta(days=i) for i in range(3) ])
df = pd.DataFrame(dict(A = s, B = td))
df['C']=df['A']+df['B']
df['D']=df['C']-df['B']
print(df)


############################ pd_topic_group2_4i_lydiajr ##############################

# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Topics in Pandas
# **Stats 507, Fall 2021**
#

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using
# the exact title with spaces replaced by a dash.
#
# + [Topic Title](#Topic-Title)
# + [Topic 2 Title](#Topic-2-Title)

# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide.

#!/usr/bin/env python
# coding: utf-8

# + [Timedeltas: Representing changes in time](#Topic-Title)
# *Lydia Rogers*, lydiajr@umich.edu
#
# Reference for timedeltas can be found [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html)

# ## Intro to Timedeltas
# * In addition to the datetime functionality that works well with pandas, there are occasions where we will want to analyze changes in time rather than dates.
# * We can represent these changes in time as timedeltas using the `datetime` module in addition to pandas.

# In[1]:


# Load required modules
import pandas as pd
import datetime

# Initializing timedeltas
# Can initialize using pd.Timedelta...
t1 = pd.Timedelta("2 days")
t2 = pd.Timedelta(2, unit = "d")
print(t1 == t2)

# null = NaT
t3 = pd.Timedelta("nan")
t4 = pd.Timedelta("nat")
print(t3, t4)

# negative times also supported
t5 = pd.Timedelta("-1us")
t6 = pd.Timedelta("-3hr")
print(t5, t6)


# ... or from a time string using to_timedelta
t7 = pd.to_timedelta("1 days 06:05:01.00003")
t8 = pd.to_timedelta("247ms")
print(t7, t8)


# ## Operations using timedeltas
# * Timedeltas can be used in arithmetic operations such as addition and subtraction with other timedeltas or datetimes, as well as in scalar multiplication.
# * Other operations supported include minimum and maximum calculations, and absolute values.
# * We can convert timedeltas to different scales (hours, minutes, seconds) by dividing one timedelta object by another.

# In[2]:


# Create example dataframe
sundays = pd.Series(pd.date_range("2021-10-24", periods=6, freq="W"))
td = pd.Series([pd.Timedelta(days=i*7) for i in range(6)])
df = pd.DataFrame({"dates": sundays, "days": td})

# Timedelta arithmetic
df["new_date"] = df["dates"] + 2 * df["days"]
df["new_timedelta"] = df["dates"] - df["new_date"]
df["scaled_timedelta"] = df["new_timedelta"] * 0.4
df["seconds_timedelta"] = df["scaled_timedelta"] / pd.Timedelta(1, "s")
print(df)

# Additional operations
print("min:", df["new_date"].min())
print("max:", df["new_date"].max())
print(abs(df["new_timedelta"]))


# ## Summarizing Timedeltas
# * Timedeltas and datetimes are also supported by most reduction operations such as mean, median, quantile, and sum.
# * These reduction operations, implemented in pandas, offer easy ways to summarize changes in time.

# In[3]:


print("mean timedelta:", df["seconds_timedelta"].mean())
print("mean date:", df["new_date"].mean())

print("median timedelta:", df["seconds_timedelta"].median())
print("median date:", df["new_date"].median())

print("timedelta 75th quantile:", df["seconds_timedelta"].quantile(0.75))
print("date 75th quantile:", df["new_date"].quantile(0.75))

print("timedelta sum:", df["seconds_timedelta"].sum())


############################ pd_topic_yangyli ##############################

#!/usr/bin/env python
# coding: utf-8

# ## STAT 507 Problem Set 4
# ## Name: Yang Li (46933158)
# ## Email: yangyli@umich.edu
# > ## Question 0 - Topics in Pandas [25 points]
#

# In[1]:


import pandas as pd
import numpy as np
import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
from scipy import stats


# ## Time series
# - Pandas contains extensive capabilities and features for working with time series data for all domains.
# - Parsing time series information from various sources and formats.

# In[2]:


dti = pd.to_datetime(["1/1/2021",
                      np.datetime64("2021-01-01"),
                      datetime.datetime(2021, 1, 1)])
dti


# ## Time series
# - Generate sequences of fixed-frequency dates and time spans.
# - Manipulating and converting date times with timezone information.
#

# In[3]:


dti = pd.date_range("2021-01-01", periods=2, freq="H")
print(dti)
dti1 = dti.tz_localize("UTC")
print(dti1)
dti1.tz_convert("US/Pacific")


# ## Time series
#
# - Resampling or converting a time series to a particular frequency.

# In[4]:


idx = pd.date_range("2021-01-01", periods=4, freq="H")
ts = pd.Series(range(len(idx)), index=idx)
ts


# ## Time series
#
# - Performing date and time arithmetic with absolute or relative time increments.

# In[5]:


friday = pd.Timestamp("2021-10-22")
print(friday.day_name())
# Add 1 day
saturday = friday + pd.Timedelta("1 day")
print(saturday.day_name())


# In[ ]:


############################ pd_topic_group2_2a_pxchen ##############################


# In[ ]:


"""
@author: Panxi Chen (pxchen)
@email: pxchen@umich.edu 
"""

### PS4 - Question 0 - Topics in Pandas


# In[1]:


# Module imports
import pandas as pd
import numpy as np
from datetime import *
from IPython.core.display import display, HTML


# ## Timedeltas
# ### Overview
# - Using Timedeltas
# - Filter data with Timedelta
#     - Example - number of homework
# - Example - age
# - Arithmetic operation
# - Example - Arithmetic Operation

# ## Timedeltas
# - Timedeltas is a type that shows the difference between values of date times. For instance, the difference between years, months, days, hours, minutes, seconds, even milliseconds, and so on. Also, the result will give in the Timedelta type.
# - the form of the datetime is not restricted. It can be either mm-dd-yyyy or yyyy-mm-dd, "-" or "/".  
# - if we do not assign month or year, then it will have default value 1, which is first day and January.

# In[2]:


# Construct Timedelta by string
td0 = pd.Timedelta('1 days 7 hours 45 minutes 30 seconds')
print(td0)


# In[3]:


td2 = pd.date_range('2020-01-24', periods=7, freq='15min')
print("The frequence is ", pd.to_timedelta(td2.freq))


# In[4]:


# Construct Timedelta
to_td = pd.to_timedelta('30min')

# transform from minutes to seconds
td_second = to_td.total_seconds()

# transform from seconds to time
td_period = str(pd.Timedelta(seconds=700000))

# print results
print("seconds:", int(td_second), "s")
print("period:", td_period)


# In[5]:


td_mdy = pd.to_datetime('8-18-2021') - pd.to_datetime('12-21-2020')

td_n = pd.to_datetime('2020-12-24') - pd.to_datetime('2019')
td_ny = pd.to_datetime('2020-12-24') - pd.to_datetime('2019-1')
td_nyr = pd.to_datetime('2020/12/24') - pd.to_datetime('2019-1-1')

print("datetime in mm-dd-yyyy with assigned date:", td_mdy)
print("datetime in yyyy-mm-dd with assigned year and month:", td_ny)
print("datetime in yyyy-mm-dd with assigned year:", td_n)
print("datetime in yyyy-mm-dd with assigned date on Jan. 1st:", td_n)


# ## Filter data with Timedelta
# - We can use Timedelta to filter the date data to obtain the results we need.

# ## Example - number of homework
# - In following example, we are going to figure out the number of homework that will due in a week.
# - From the dataframe df_dues, we could see from index 1 to 3, the homework is closed.

# In[6]:


dues = {'due_date': ['2021/10/07', '2021/10/11', 
                     '2021/10/22', '2021/10/24', 
                     '2021/10/27', '2021/10/31',], 
        'hw_number': [3, 1, 2, 5, 4, 2]}
df = pd.DataFrame(dues, index=[1, 2, 3, 4, 5, 6])
display(HTML(df.to_html()))


# In[7]:


df['due_date'] = pd.to_datetime(df['due_date'])
df = df[(df['due_date'] - datetime.now()) > pd.Timedelta(days=0)]
df = df[(df['due_date'] - datetime.now()) < pd.Timedelta(days=7)]
display(HTML(df.to_html(index=False)))
print(df['hw_number'].sum(), "homework needs to be finished in one week.")


# ## Example - age
# - We could apply Timedelta to calculate the age of a person given his birth date.
# - In this example, we use Timedelta to find the age for someone born on 06/27/1998.

# In[8]:


age = int((datetime.now() 
           - pd.to_datetime('6-27-1998')) 
          / pd.Timedelta(days=365))
print("age:", age)


# ## Arithmetic Operation
# - This example shows the arithmetic operation of Timedelta. Also, a time series is constructed further on the basis of Timedelta.

# ## Example - Arithmetic Operation

# In[9]:


time = pd.date_range(start='2021-9-7', periods=7, freq='7D')
print("Time period from {} to {}".format(time.min(), time.max()))


# In[10]:


# add 5 minutes
time_plus = time + pd.Timedelta(minutes=5)

# subtract datetime(year, month, day, hour, minute)
time_minus = time - datetime(2020, 1, 21, 3, 5)

#print results
print("Addition: ", time_plus)
print("Subtraction: ", time_minus)


# In[11]:


td1 = pd.Series([pd.Timedelta(days=3*(i+1)) for i in range(7)])
df1 = pd.DataFrame(dict(date=time, period=td1))

# operations
df1['addition']=df1['date'] - df1['period']
df1['subtraction']=df1['date'] - df1['period']
display(HTML(pd.DataFrame(df1).to_html()))


# In[12]:


# Construct Time Series
ts1 = pd.Series(np.random.randn(7), index=time)
print('Time seris (Oct. 5th):', ts1['2021-10-05'])
display(HTML(pd.DataFrame(ts1).to_html()))


# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
#
# + [Dropping](#Dropping) 
# + [Pandas Aggregation](#Pandas-Aggregation)
# + [Pandas with Time](#Pandas-with-Time)
# + [Window Functions](#Window-Functions) 
# + [Rolling Functions](#Rolling-Functions) 
# + [Expanding Functions](#Expanding-Functions) 
# + [Time series](#Time-series)
# + [Timestamps vs time spans](#Timestamps-vs-time-spans)

# ## Dropping
#
# __Pengfei Liu__    
# 2021/11/16

# ## Dropping In Pandas
# - Delete (some parts of) data samples under certain conditions using `pd.drop()`.
# - Use `axis` to choose to drop from the index(`0` or `index`) or columns(`1` or `columns`).
# - General usage: `df.drop(df[<some boolean condition>].index)`

import numpy as np
import pandas as pd
df1 = pd.DataFrame(np.arange(24).reshape(6, 4), columns=['A', 'B', 'C', 'D'])
print(df1)
#drop columns
print(df1.drop(['B','C'], axis = 1))
#drop the items that is less than 6 in column A.
print(df1.drop(df1[df1['A'] < 6].index))

# - In the previous test, we showed the importance of `labels`, which was `df1[df1['A'] < 6].index` there.
# - we can also use the parameter `level` to make multiple indexing.
#

midx = pd.MultiIndex(levels=[['lama', 'cow', 'falcon'],
                             ['speed', 'weight', 'length']],
                     codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
                            [0, 1, 2, 0, 1, 2, 0, 1, 2]])
df = pd.DataFrame(index=midx, columns=['big', 'small'],
                  data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
                        [250, 150], [1.5, 0.8], [320, 250],
                        [1, 0.8], [0.3, 0.2]])
print(df)
print(df.drop(index = 'cow', columns = 'small'))

# - We also have other relative functions such as `pd.dropna()` and `pd.drop_duplicates()`.

# ## Pandas Aggregation
#
# __LiHsuan Lin__    
# 2021/11/16

# ####  An essential piece of analysis in large dataset is efficient summarization
# #### There are a few aggregation operations in Pandas that are useful
# #### Aggregate using one or more operations over the specified axis

# +


# create df
df = pd.DataFrame([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [np.nan, np.nan, np.nan]],
                  columns=['A', 'B', 'C'])
df

# +


# sum and min aggregation operation
df.agg(['sum', 'min'])


# ### Normally, it is used hand in hand with group_by to find the aggregation on groups
# 
# - first specify the column (group) and then apply agg

# +


# create example df
data1 = {'Name':['Max', 'Bill', 'Max', 'Princi', 
                 'Gaurav', 'Bill', 'Princi', 'Tom'], 
        'salary(K)':[80, 78, 56, 110, 
               78, 87, 150, 32], 
        'Address':['Nagpur', 'Kanpur', 'Allahabad', 'Kannuaj',
                   'Jaunpur', 'Kanpur', 'Allahabad', 'Aligarh'], 
        'Qualification':['MS', 'MA', 'MCA', 'Phd',
                         'B.Tech', 'Phd', 'MA', 'MS']}
df = pd.DataFrame(data1)
df

# +


# group by Name
df.groupby(['Name']).mean()

# +


df.groupby(['Name','Qualification']).sum()


# - multiple aggregrations
# - use {Column:Operation} to specify the group and which operations to use

# +


df.groupby('Name').agg({'salary(K)':['min', 'max']})
# -

# ## Pandas with Time
# *Stats 507, Fall 2021*  
#    
# __Jinhuan Ke__  
# jhgoblue@umich.edu  
# 10/19/2021
#   
# - Timestamp
# - Period
# - DatetimeIndex and PeriodIndex
# - Generating ranges of timestamps

import pandas as pd
import numpy as np
import datetime

# ## Timestamp
# - Timestamped data is the most basic type of time series data that associates values with points in time. 
# - For pandas objects it means using the points in time.
# - use `pd.Timestamp()` to generate time series data save in seconds.
# - time variable can be `year, month, day` or `"year-month=day"`.

print(pd.Timestamp("2021-10-21"))
print(pd.Timestamp(2021, 10, 21))

# ## Period
# - Time spans can be represented by `Period`.
# - use `pd.Period()` to generate time span variables.
# - the `freq` parameter in the function specify the frequency of the time series.

print(pd.Period("2021-10-21", freq='Y'))
print(pd.Period("2021-10-21", freq='M'))
print(pd.Period("2021-10-21", freq='D'))

# ## DatetimeIndex and PeriodIndex
# - Timestamp and Period can serve as an index. 
# - Lists of Timestamp and Period are automatically coerced to DatetimeIndex and PeriodIndex respectively.

dates = [
    pd.Timestamp("2021-10-21"),
    pd.Timestamp("2021-10-22")
]
t1 = pd.Series(np.random.randn(2), dates)
print(t1.index)
print(type(t1))
t1

periods = [pd.Period("2021-10-21"), pd.Period("2021-10-22")]
t2 = pd.Series(np.random.randn(2), periods)
print(t2.index)
print(type(t2))
t2

periods_m = [pd.Period("2021-10-21", 'M'), pd.Period("2021-10-22", 'M')]
t3 = pd.Series(np.random.randn(2), periods_m)
print(t3.index)
print(type(t3))
t3

# ## Generating ranges of timestamps
# - we can use the `date_range()` and `bdate_range()` functions to create a DatetimeIndex. 
# - The default frequency for `date_range` is a calendar day,
# - while the default for `bdate_range` is a business day.
# - `freq` parameter in function can refer to the [Date Offset][do]
#
# [do]: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

# +
start = datetime.datetime(2021, 10, 21)
end = datetime.datetime(2021, 12, 21)
dtindex_1 = pd.date_range(start, end)
print(dtindex_1)

dtindex_2 = pd.bdate_range(start, end)
print(dtindex_2)

dtindex_3 = pd.bdate_range(start, periods=10, freq='BM') #business month end
print(dtindex_3)
# -

# # Question 0 - Topics in Pandas

# ##  Window Functions
# * Provide the calculation of statistics 
# * Primarily used in signal processing and time series data
# * Perform desired mathematical operations on consecutive values at a time

# ## Rolling Functions
# * Can apply on Series and Dataframe type
# * For DataFrame, each function is computed column-wise
# * Here is an example for calculation of rolling sum
# * Similar way to calculate the mean, var, std ...

series = pd.Series(range(5))
df = pd.DataFrame({"A": series, "B": series ** 3})
df.rolling(3).sum()

# ## Parameters
# * **win_type** changes the window function (equally weighted by default)
# * You can set the minimum number of observations in window required to have a
# value by using **min_periods=k**
# * Set the labels at the center of the window by using **center==True** 
# (set to the right edge by default)

print(df.rolling(3, win_type='gaussian').sum(std=3))
print(df.rolling(3, win_type='gaussian', center=True).sum(std=3))
print(df.rolling(3, win_type='gaussian', min_periods=2, 
                 center=True).sum(std=3))

# ## Expanding Functions
# * Calculate the expanding sum of given DataFrame or Series
# * Perform desired mathematical operations on current all previous values
# * Similar way to calculate the mean, var, std...
df.expanding(min_periods=2).sum()

# + [General Time Series Functions](#General-Time-Series-Functions)
# modules: -------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind_from_stats
from scipy.stats import norm
from Stats507_hw4_helper import ci_prop
from Stats507_hw4_helper import ci_mean
from numpy import arange
import itertools
from scipy.stats import binom
from collections import defaultdict
import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import datetime
# + [Convert to Timestamps](#Convert-to-Timestamps) [markdown]
# ## Time Series 
# **Ran Yan**
# -
# # Question 0 - Topics in Pandas
#
# ##  General Time Series Functions
# * pandas contains extensive capabilities and features for working with time series data for all domains.
# * can parse time series information in different formats and sources
# * can manipulating and converting date times with timezone information

# +
# example 1
dti = pd.to_datetime(
    ["5/20/2000", np.datetime64("2000-05-20"), datetime.datetime(2000, 5, 20)]
)
dti

# example 2
dti = pd.date_range("2000-01", periods=5, freq="D")
dti

# example 3
dti = dti.tz_localize("UTC")
dti
# -

# ## Convert to Timestamps
# * when convert to timestamps, convert series to series and convert list-like data to DatetimeIndex
# * return a single timestamp if pass a string
# * can use format argument to ensure specific parsing, here is an example below

pd.to_datetime("2021/10/23", format="%Y/%m/%d")

# ## Other Useful Time Series Functions
# - `between_time` 
#     - to select rows in data frame that are only between certain time range
#     - here is an example below to select time from 9:00 to 10:00
# - `date_range`
#     - convert timestamp to a 'unix' epoch
# - `bdate_range`
#     - create DatetimeIndex for in a range of business days

# +
rng = pd.date_range('2/3/2000', periods=24, freq='H')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts.iloc[ts.index.indexer_between_time(datetime.time(9), datetime.time(10))]

stamps = pd.date_range("2021-10-23 20:15:05", periods=2, freq="H")
stamps

start = datetime.datetime(2021, 10, 20)
end = datetime.datetime(2021, 10, 23)
index = pd.bdate_range(start, end)
index
# -

# Eduardo Ochoa Rivera \
# eochoa@umich.edu \
# November 20, 2021

# ## Module imports

import numpy as np
import pandas as pd
import datetime

# ## Time series

# pandas contains extensive capabilities and features for working with time series data for all domains.
# For example: 
# * Parsing time series information from various sources and formats

dti = pd.to_datetime(["1/1/2018", 
                      np.datetime64("2018-01-01"), 
                      datetime.datetime(2018, 1, 1)])
dti

# * Generate sequences of fixed-frequency dates and time spans

dti = pd.date_range("2018-01-01", periods=3, freq="H")
dti

# * Performing date and time arithmetic with absolute or relative time increments. It can be specially usefull for financial data analysis.
#

friday = pd.Timestamp("2018-01-05")
print(friday.day_name())
saturday = friday + pd.Timedelta("1 day")
print(saturday.day_name())
monday = friday + pd.offsets.BDay()
print(monday.day_name())

# ## Timestamps vs time spans

# Timestamped data is the most basic type of time series data that associates values with points in time.

pd.Timestamp(datetime.datetime(2012, 5, 1))

# However, in many cases it is more natural to associate things like change variables with a time span instead.
#

pd.Period("2011-01"), pd.Period("2012-05", freq="D")

# **Timestamp** and **Period** can serve as an index.

# +
dates = [
    pd.Timestamp("2012-05-01"),
    pd.Timestamp("2012-05-02"),
    pd.Timestamp("2012-05-03"),
]
ts = pd.Series(np.random.randn(3), dates)
print(ts.index) 
ts

periods = [pd.Period("2012-01"), 
           pd.Period("2012-02"), 
           pd.Period("2012-03")]
ts = pd.Series(np.random.randn(3), periods)
print(ts.index) 
ts

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

