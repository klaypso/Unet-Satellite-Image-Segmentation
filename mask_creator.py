#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:18:33 2018

@author: alex
"""

import numpy as np
import cv2
import os 
import pandas as pd
from shapely.wkt import loads as wkt_loads
from gdal_utilities import gdal_utils

### Generate the masks for ground truth ###
    
class mask_generator :
    
    def __init__(self):
        
        self.CLASSES =  {
                            1 : 'Bldg',
                            2 : 'Struct',
                            3 : 'Road',
                            4 : 'Track',
                            5 : 'Trees',
                            6 : 'Crops',
                            7 : 'Fast_H20',
                            8 : 'Slow_H20',
                            9 : 'Truck',
                            10 : 'Car',
                        }
        
         

    def _convert_coordinates_to_raster(self,coords, img_size, xymax):
        """ Do the transformtions to the co-ordinates of the image - using the formula given.

        Arguments :
            coords --
            img_size --
            xymax --

        Returns :
            coords_int -- 

        """
        Xmax,Ymax = xymax
