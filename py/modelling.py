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
import folium

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
#os.path.join(os.path.curdir, 'selecting_stations.py')
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

ct= pd.crosstab([training_data['NAME'], training_data['BIKE STANDS'], training_data['STATION ID']],training_data['cluster'])
print(ct)
#%%
df = training_data.copy()
df = df.dropna()


#%%

#https://github.com/Panchop10/dublinbike_predictive_analytics
#%%
df = training_data.copy()
df = df.dropna()


#%%
df = df.drop(columns = {'NAME','STATUS','ADDRESS', 'LATITUDE','LONGITUDE', 'LAST UPDATED','AVAILABLE BIKE STANDS',\
                        'time_type', 'hour', 'dayIndex', \
                        'STATION ID','BIKE STANDS', 'AVAILABLE BIKES', 'time_type',\
                        'datetime', 'date_for_merge', 'time', 'TIME', 'day_type', \
                        'year', 'yearWeek'})
#%%
#%%
df = df.drop(columns = {'day_number'})

#%%
# specify the predictors and the response variable

X = df.copy().drop(columns = 'usage')
y = df.usage

dday = df.groupby(['date']).mean()

X = dday.copy().drop(columns = 'usage')
y = dday.usage

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#%%
y_pred = regressor.predict(X_test)
dplot = pd.DataFrame({'actual': y_test, 'predicted': y_pred})

dplot1 = dplot.copy()#.head(40)
dplot1['index'] = dplot1.reset_index().index

#%%

#%%
ax = plt.gca()

dplot1.plot(kind='line',x='index',y='actual', color='green',ax=ax)
dplot1.plot(kind='line',x='index',y='predicted', color='red', ax=ax)
plt.title('Actual vs predicted usage Linear regression')
plt.xlabel('index')
plt.ylabel('bike usage (normalised)')

plt.show(block=True)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#%%
#

regressions = [
    svm.SVR(),  #0.09251714474797995 
    linear_model.SGDRegressor(),   # awful
    linear_model.BayesianRidge(),  #0.030229285858572368
    linear_model.LassoLars(),  # 0.030367853977666812
    linear_model.ARDRegression(),   # out of memory
    linear_model.PassiveAggressiveRegressor(),  # 0.09249226888404853 
    linear_model.TheilSenRegressor(), # out of memory
    linear_model.LinearRegression(),  #0.03022928687137188 
    linear_model.RidgeCV()   ]  #0.030229287363110348
    

# fit & score each regression model
for item in regressions:
    print(item)
    reg = item
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred),'\n')
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)),'\n')


#%%
scores = cross_val_score(reg, X, y, cv=5)
scores
#array([0.21907824, 0.24079283, 0.3658323 , 0.3448517 , 0.20676684])

#%%
# best (lowest) rmse = linear_model.RidgeCV() (and lowest mae)
reg = linear_model.RidgeCV()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(reg)

#%%
dplot = pd.DataFrame({'actual': y_test, 'predicted': y_pred})

dplot1 = dplot.copy()
dplot1['index'] = dplot1.reset_index().index

#%%
ax = plt.gca()

dplot1.plot(kind='line',x='index',y='actual', color='green',ax=ax)
dplot1.plot(kind='line',x='index',y='predicted', color='red', ax=ax)
plt.title('Actual vs predicted usage RidgeCV')
plt.xlabel('index')
plt.ylabel('bike usage (normalised)')

plt.show(block=True)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#%%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(reg, X, y, cv=5)

#%%
sorted(cv_results.keys())
cv_results['test_score']
# array([0.21907824, 0.24079283, 0.3658323 , 0.3448517 , 0.20676684])

#%%
scores = cross_validate(reg, X, y, cv=5,\
                        scoring = ('r2', 'mean_squared_error'), \
                        return_train_score = True)
sorted(scores.keys())

scores['test_r2']
scores['test_mean_squared_error']
#array([-1.18619908e-05, -1.34156808e-05, -1.01247625e-05, -9.78712672e-06, -1.55826859e-05])

#%%

#%%
## Run the model on the full test set and compare to the actual 2020 data
#%%

# train model on all of training data 
reg.fit(X, y)

print(reg)
 

#%%
#final test set - actual 2020 data
test_data = data[data['year'] == 2020]

# check we have the right data
ct= pd.crosstab([test_data['NAME'], test_data['BIKE STANDS'], test_data['STATION ID']],test_data['cluster'])
print(ct)


#%%

df2020 = test_data.copy().dropna()
df2020  = df2020.drop(columns = {'NAME','STATUS','ADDRESS', 'LATITUDE','LONGITUDE', 'LAST UPDATED','AVAILABLE BIKE STANDS',\
                        'time_type', 'hour', 'dayIndex', \
                        'STATION ID','BIKE STANDS', 'AVAILABLE BIKES', 'time_type',\
                        'datetime', 'date_for_merge', 'time', 'TIME', 'day_type', \
                        'year', 'yearWeek'})

#%%
# specify the predictors and the response variable
dday = df2020.groupby(['date']).mean()

X2020  = dday.copy().drop(columns = 'usage')
y2020 = dday.usage

#%%
y_pred = reg.predict(X2020)

print('Mean Absolute Error:', metrics.mean_absolute_error(y2020, y_pred),'\n')
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y2020, y_pred)),'\n')

#%%
ymean = y.mean(0)
print(ymean)

#%%
#y_pred = regressor.predict(X_test)
plot2020 = pd.DataFrame({'actual': y2020, 'predicted': y_pred})
plot2020['mean'] = ymean

df1 = plot2020
df1['index'] = df1.reset_index().index
#print(df1)
#%%
ax = plt.gca()

df1.plot(kind='line',x='index',y='actual', color='green',ax=ax)
df1.plot(kind='line',x='index',y='predicted', color='red', ax=ax)
#df1.plot(kind='line',x='index',y='mean', color='blue', ax=ax)
plt.title('Actual vs predicted usage - 2020')
plt.xlabel('Day of the Year')
plt.ylabel('bike usage (normalised)')

plt.show(block=True)



#%%
len(y2020) - len(y_pred)


#%%
