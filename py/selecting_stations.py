#import necessary libraries and files 
import pandas as pd
import numpy as np
import warnings
import time
import datetime as dt
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import folium
import sklearn
import seaborn as sns

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os
import glob
##
print(os.getcwd())
old_dir = os.getcwd()

#os.chdir('C:/Users/imeld/work/ML_CSP/ML-Project')

##

data = pd.read_csv("data/full_data.csv")
##

data = data.dropn
#m= [not pd.isna(c) for c in data['cluster']]
#m.head()
##
ct = pd.crosstab(index=data['NAME'], columns=data['cluster'])
print(ct)

data = data.sort_values(['NAME', 'TIME' ])

data.to_csv("data/station_data.csv", index=False)

locations  = data[['LATITUDE', 'LONGITUDE', 'cluster']].drop_duplicates()


data.sample(50)
colordict = {0: 'blue', 1: 'red', 2: 'green'}
dublin_map = folium.Map([53.345, -6.2650], zoom_start=13.5)

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
  
