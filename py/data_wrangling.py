#import necessary libraries and files 
import pandas as pd
import numpy as np
import warnings
import time
import datetime as dt
from datetime import date
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
#import folium
import sklearn
import seaborn as sns

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os
#import glob
#%%
#home = str(os.Path.home())
print(os.getcwd())
old_dir = os.getcwd()
#os.path.join(os.path.curdir, 'selecting_stations.py')
os.chdir(os.path.join(os.path.curdir, 'Documents/GitHub/ML-Project'))
#%%
print(os.getcwd())


#%%
os.chdir(os.path.join(os.path.curdir, '../'))
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
data = data.drop('include', axis = 1)
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
#data['OCCUPANCY_PCT'] =  data['AVAILABLE BIKES'] / data['BIKE STANDS']
#data['FULL'] = np.where(data['OCCUPANCY_PCT'] == 0, 1,0 )
#data['EMPTY'] = np.where(data['OCCUPANCY_PCT'] == 1, 1,0 )

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

print(data.sample(5))
#%%
df_usage = data[['date', 'usage']] 
df_usage.date = dates = [dt.datetime.strptime(d, "%Y-%m-%d") for d in df_usage["date"]]
diff_date = abs(df_usage.date.dt.date.diff())



#%%

m = diff_date.dt.days <=1
data = data[m]


#%%
data.to_csv("data/station_data.csv", index=False)

print(data.columns.values)


#%%
test_data = data[data['year']>= 2020]
test_data.to_csv("data/test.csv", index=False)
print(pd.crosstab(index=test_data['NAME'], columns = data['cluster']))
del test_data
#%%
#training set
training_data = data[data['year'] < 2020]
#len(training_data)
training_data.to_csv("data/training.csv", index=False)#%%


ct= pd.crosstab([training_data['NAME'], training_data['BIKE STANDS'], training_data['STATION ID']],training_data['cluster'])
print(ct)
#%%
df = data.copy()

range_start = date(2019, 12, 15)
range_end = date(2020, 2, 1)

m = (df.date >= range_start) & (df.date <= range_end)
df_short = df[m].sort_values(by='datetime')
plt.plot(df_short.date,df_short['AVAILABLE BIKES'])
#colors = np.array([0, 4.0, 8.0, 12.0, 16.0, 20, 24.0, 28, 32.0, 36.0, 40])
#, c=colors, cmap='viridis'
#plt.scatter(df_short.date,df_short['AVAILABLE BIKES'], s=2, color='hotpink')
plt.show()

#color='hotpink'
#%%


df_short.date.unique()


#%%
d= {'day': [18, 19, 20, 21, 26, 27, 28, 29, 2, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 22, 26], \
    'month': [12, 12, 12, 12, 12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], \
    'year': [2019, 2019, 2019, 2019, 2019, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020, 2020]}


missing = pd.DataFrame(data=d)

missing['date'] = [date(a, b, c) for a, b, c in zip(missing.year, missing.month, missing.day)]
missing.to_csv("data/missing.csv", index = False)
missing

#%%

df_daily=data.groupby('date').agg(['mean']).reset_index()
df_daily.plot(x='date', y='usage',kind="line")
plt.show()


#%%


m = diff_date.dt.days >1
df_usage[m].to_csv('data/discontinuities.csv') 

#%%
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=list("ABCD"))

df = df.cumsum()

plt.figure();

df.plot();