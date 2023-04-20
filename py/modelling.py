#import necessary libraries and files 
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
import time
import datetime as dt
from datetime import date

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
#import folium

#from matplotlib import pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import svm

from sklearn import metrics

import os
#import glob
#%%
#home = str(os.Path.home())
print(os.getcwd())
old_dir = os.getcwd()
#os.path.join(os.path.curdir, 'selecting_stations.py')
os.path.join(os.path.curdir, 'selecting_stations.py')
os.chdir(os.path.join(os.path.curdir, 'Documents/GitHub/ML-Project'))
print(os.getcwd())

#%%
print(os.getcwd())
#%%
data = pd.read_csv("data/station_data.csv")
#%%
data.head()
#%%
#data.isnull().sum()
data.isna().sum()
#%%
# constant value switches
MAKE_FILES=True
READ_FILES = False
#%%
data  = data.sort_values(['NAME', 'TIME' ])

#%%
print((data['NAME']).unique())

#%%
data['usage'] = abs(data['AVAILABLE BIKES'].diff())/data['BIKE STANDS']
#%%
data.to_csv("data/station_data.csv", index=False)
#%%

round(data.describe(), 3)

#%%
ct = pd.crosstab(index=data['NAME'], columns=data['cluster'])

print(ct)
#%%
#data = data[data['STATUS'] == 'Open']

data.drop_duplicates(keep= 'first',inplace=True)

dates = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in data["TIME"]]
#get date and time columns
data['datetime'] = dates
data['time'] = data.datetime.dt.time
#data['time'] = [dt.datetime.time(d) for d in data['DATETIME']] 
data['date'] = data.datetime.dt.date
#data['date'] = [dt.datetime.date(d) for d in data['DATETIME']] 
data['date_for_merge'] = data.datetime.dt.round('d')


startDate = min(dates).date()
#create important features
data['OCCUPANCY_PCT'] =  data['AVAILABLE BIKES'] / data['BIKE STANDS']
data['FULL'] = np.where(data['OCCUPANCY_PCT'] == 0, 1,0 )
data['EMPTY'] = np.where(data['OCCUPANCY_PCT'] == 1, 1,0 )

### create time aggregates needed for clustering
# weekday/saturday/sunday
data['day_number'] = data.datetime.dt.dayofweek
data['day_type'] = np.where(data['day_number'] <= 4, 'Weekday', (np.where(data['day_number'] == 5, 'Saturday', 'Sunday')))

def bin_time(x):
    if x.time() < dt.time(6):
        return "Overnight "
    elif x.time() < dt.time(11):
        return "6AM-10AM "
    elif x.time() < dt.time(16):
        return "11AM-3PM "
    elif x.time() < dt.time(20):
        return "4PM-7PM "
    elif x.time() <= dt.time(23):
        return "8PM-11PM "
    else:
        return "Overnight "

data["time_type"] = data.datetime.apply(bin_time)
data['hour'] = data.datetime.dt.hour
data['month'] = data.datetime.dt.month
data['week'] = data.datetime.dt.week
data['year'] = data.datetime.dt.year
data['dayIndex'] = [(d - startDate).days for d in data['date']]
data['yearWeek'] = data.year *100+data.week

data.sample(5)
#%%
print(data.columns.values)

#%%
#training set
training_data = data[data['year'] < 2020]
#len(training_data)
training_data.to_csv("data/training.csv", index=False)#%%


ct= pd.crosstab([training_data['NAME'], training_data['BIKE STANDS'], training_data['STATION ID']],training_data['cluster'])
print(ct)
#%%
df = training_data.copy()
df = df.dropna()
#range_start = date(2019, 12, 15)
#range_end = date(2020, 2, 1)

#m = (df.date >= range_start) & (df.date <= range_end)
#df_short = df[m].sort_values(by='datetime')

#%%
#df['yearWeek'] = df.year *100+df.week

# Remove columns with information that we don't need for the clustering
df = df.drop(columns = {'NAME','STATUS','ADDRESS', 'LATITUDE','LONGITUDE', 'LAST UPDATED','AVAILABLE BIKE STANDS',\
                        'time_type', 'hour', 'dayIndex', 'year',  'OCCUPANCY_PCT', 'FULL', 'EMPTY',\
                        'STATION ID','BIKE STANDS', 'AVAILABLE BIKES', 'day_type',\
                        'month', 'day_number', 'datetime', 'week', 'date_for_merge', 'time', 'TIME',\
                        'date'})
#%%
    
#df = df.drop(columns = {'TIME'})
#%%
df_avg = df.groupby(['cluster','yearWeek']).agg('mean')
df_avg = df_avg.reset_index()

#print(df_avg)
#%%
plt.plot(df_avg.yearWeek, df_avg.usage)
#%%
# Reshape to get each time of the day in a column (features) and each station in a row (data-points)
X = df_avg.pivot(index='cluster' , columns='yearWeek', values='usage')
print(X.shape)
X
#%%
print(X.columns)

#%%
#%%

#%%

#%%

df = training_data.copy()
#%%
df = df.dropna()

#%%
df = df.drop(columns = {'NAME','STATUS','ADDRESS', 'LATITUDE','LONGITUDE', 'LAST UPDATED','AVAILABLE BIKE STANDS',\
                        'time_type', 'hour', 'dayIndex', 'OCCUPANCY_PCT', 'FULL', 'EMPTY',\
                        'STATION ID','BIKE STANDS', 'AVAILABLE BIKES', 'time_type',\
                        'datetime', 'date_for_merge', 'time', 'TIME', 'day_type', 'date'})
#%%
df.head(1)
#%%
X = df.copy().drop(columns = 'usage')
y = df.usage
#%%
#df = df.drop(columns='datetime')
#%%
#The script splits the dataset into 80% train data and 20% test data.
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#%%
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'actual': y_test, 'predicted': y_pred})

df1 = df.head(40)
df1['index'] = df1.reset_index().index
#print(df1)
#%%
ax = plt.gca()

df1.plot(kind='line',x='index',y='actual', color='green',ax=ax)
df1.plot(kind='line',x='index',y='predicted', color='red', ax=ax)
plt.title('Actual vs predicted usage (Linear Regresion Method)')
plt.xlabel('index')
plt.ylabel('bike usage')

plt.show(block=True)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#https://github.com/Panchop10/dublinbike_predictive_analytics
#%%
#clf = svm.SVC(kernel='linear', C=1, random_state=42)
#scores = cross_val_score(clf, X, y, cv=5)
#scores
#%%
#

regressions = [
    svm.SVR(),
    linear_model.SGDRegressor(),
    linear_model.BayesianRidge(),
    linear_model.LassoLars(),
    #linear_model.ARDRegression(),
    linear_model.PassiveAggressiveRegressor(),
    linear_model.TheilSenRegressor(),
    linear_model.LinearRegression()]

# fit & score each regression model
for item in regressions:
    print(item)
    reg = item
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred),'\n')
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)),'\n')


#%%

#%%
