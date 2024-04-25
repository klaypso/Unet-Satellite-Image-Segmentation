
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:43:42 2018

@author: alex
"""
from gdal_utilities import gdal_utils
import pandas as pd
import numpy as np
import os
from shapely import affinity
from shapely.wkt import loads as wkt_loads
from matplotlib.patches import Polygon, Patch

# decartes package makes plotting with holes much easier
from descartes.patch import PolygonPatch
import matplotlib.pyplot as plt


'''
### See what a WTK instance looks like ###

polygonsList = {}
df_image = df[df.ImageId == '6060_2_3']
for cType in CLASSES.keys():
    polygonsList[cType] = wkt_loads(df_image[df_image.ClassType == cType].MultipolygonWKT.values[0])    
    
print(polygonsList[1])    
'''   

'''
#### WORKFLOW ####

1) Define your canvas i.e fig (MATPLOTLIB.figure)
2) Create subplots and get the axes instances i.e ax (MATPLOTLIB.add_subplot)
3) Get a WTK object instance that has sets of co-ordinates
3) Create 2-D Shapely object instance by judging what shape the WTK object is and passing the WTK instance to the correct constructor
4) Create a pacth by passing that 2-D object instance to the PolygonPatch class constructor of descartes
5) Adding that patch to the list of patches of the axis attribute
6) Plot that patch
7) Enjoy !!

    
from shapely.geometry import LineString
fig = plt.figure()
ax = fig.add_subplot(121) # Adding a subplot 
line =  LineString([(0, 0), (1, 1), (0, 2), (2, 2), (3, 1), (1, 0)]) # Creating a line instance by passing a WTK instance
dilated = line.buffer(0.5)

patch1 = PolygonPatch(dilated,facecolor='#99ccff',edgecolor='#6699cc')
ax.add_patch(patch1)

x,y = line.xy
ax.plot(x,y,color = '#999999')
ax.set_xlim(-1,4)
ax.set_ylim(-1,3)

'''



# Give short names, sensible colors and zorders to object types
CLASSES = {
        1 : 'Bldg',
        2 : 'Struct',
        3 : 'Road',
        4 : 'Track',
        5 : 'Trees',
        6 : 'Crops',
        7 : 'Fast H20',
        8 : 'Slow H20',
        9 : 'Truck',
        10 : 'Car',
        }
COLORS = {
        1 : '0.7',
        2 : '0.4',
        3 : '#b35806',
        4 : '#dfc27d',
        5 : '#1b7837',
        6 : '#a6dba0',
        7 : '#74add1',
        8 : '#4575b4',
        9 : '#f46d43',
        10: '#d73027',
        }

# Artists with lower zorder values are drawn first.
# Hence tracks are given the lowest order and cars have been given the highest order

ZORDER = {
        1 : 5,
        2 : 5,
        3 : 4,
        4 : 1,
        5 : 3,
        6 : 2,
        7 : 7,
        8 : 8,