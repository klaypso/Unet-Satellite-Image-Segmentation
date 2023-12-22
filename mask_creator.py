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
        H,W = img_size[0:2]
                
        W1 = 1.0*W*W/(W+1)
        H1 = 1.0*H*H/(H+1)
        xf = W1/Xmax
        yf = H1/Ymax
        coords[:,1] *= yf
        coords[:,0] *= xf
        coords_int = np.round(coords).astype(np.int32)
        return coords_int

    def _get_xmax_ymin(self,grid_sizes_panda, imageId):
        """ Returns the xmax and ymin of the photographs.

        Arguments : 
            grid_sizes_panda -- pandas.DataFrame Object
                                The pandas dataframe with all the grid sizes for each image

            imageId          -- str.
                                The id of the image. ex. 6010_0_0.tif

        Returns :
            (xmax,ymin) -- tuple.
                           The maximum x co-ordinate and the minimum y co-ordinate
        """
        xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0,1:].astype(float)
        return (xmax,ymin)


    def _get_polygon_list(self,wkt_list_pandas, imageId, cType):
        """ Gets the list of polygons that were created for class : "cType" in image with id : "imageId"

        Arguments :
            wkt_list_pandas -- pandas.DataFrame Object
                               The pndas DataFrame with all the shape files of the satellite image.
            imageId         -- str.
                               The id of the image. ex. 6010_0_0.tif
            cType           -- str.
                               The name of the class type. ex 'Bldg','Struct'
        Returns :
            polygonList -- 

        """
        df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
        multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
        polygonList = None
        if len(multipoly_def) > 0:
            assert len(multipoly_def) == 1
            polygonList = wkt_loads(multipoly_def.values[0])
        return polygonList


    def _get_and_convert_contours(self,polygonList, raster_img_size, xymax):
        """
        Converts the co-ordindates of the polygons using the transformation rules that were stated.
        It then returns two sets of co-ordinates - the outer contour of the polygons "perim-list", and the inner contour of the polygons "interior_list"
        """
        perim_list = []
        interior_list = []
        if polygonList is None:
            return None
        
        for k in range(len(polygonList)):
                        
            # Get the outer contours of the polygons and add to the perim_list
            poly = polygonList[k]
            perim = np.array(list(poly.exterior.coords))
            perim_c = self._convert_coordinates_to_raster(perim, raster_img_size, xymax)
            perim_list.append(perim_c)
            
            # For each polygon get the interior contours of the po