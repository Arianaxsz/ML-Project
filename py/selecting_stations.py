#import necessary libraries and files 
import pandas as pd
import numpy as np
import warnings
import time
import datetime as dt
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
os.path.join(os.path.curdir, 'selecting_stations.py')
os.chdir(os.path.join(os.path.curdir, 'Documents/GitHub/ML-Project'))
print(os.getcwd())


#%%
print(os.getcwd())

#%%

data = pd.read_csv("data/station_data.csv")
#%%

selected_stations = pd.DataFrame(
    {
        "NAME": ['HANOVER QUAY', 'FITZWILLIAM SQUARE EAST', 'YORK STREET EAST', 'NEW CENTRAL BANK', 
'MATER HOSPITAL', 'PARNELL SQUARE NORTH'],
        "cluster": [0, 0, 1, 1, 2, 2]
    
    }
)

#%%
data = pd.merge(data, selected_stations, how='left', on = 'NAME', \
              indicator=False)
#%%

data =  data.dropna()
#m= [not pd.isna(c) for c in data['cluster']]
#m.head()

##
data.to_csv("data/station_data.csv", index=False)
##
ct = pd.crosstab(index=data['NAME'], columns=data['cluster'])
print(ct)

data = data.sort_values(['NAME', 'TIME' ])



#%%
def mapLocations(data) :
    locations  = data[['LATITUDE', 'LONGITUDE', 'cluster']].drop_duplicates()
    #data.sample(50)
    colordict = {0: 'blue', 1: 'red', 2: 'green'}
    dublin_map = folium.Map([53.345, -6.2650], zoom_start=13.0)

    for LATITUDE, LONGITUDE, cluster in zip(locations['LATITUDE'],locations['LONGITUDE'],locations['cluster']):
        folium.CircleMarker(
            [LATITUDE, LONGITUDE],
            color = 'b',
            radius = 8,
            fill_color=colordict[cluster],
            fill=True,
            fill_opacity=0.9
            ).add_to(dublin_map)
    dublin_map  
  
#%%