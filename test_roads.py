
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:07:28 2018

@author: alex
"""
from up_convolution import convolution,trans_convolve
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from gdal_utilities import gdal_utils
from scipy.ndimage import binary_opening,binary_closing
from scipy.signal import medfilt2d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

############################################ UTILITY FUNCTIONS ############################################
def get_masks_list():
    
    '''
    Gets the list of masked tiff files intended for training and testing 
    '''
    
    inDir = os.getcwd()    
    
    files_train_temp = os.listdir(os.path.join(inDir,"Data_masks/Road"))
    files_train_final = []
    
    i = 0 
    for file in files_train_temp : 
        
        extension = os.path.splitext(file) 
    
        if extension[0] == 'Test' :
            files_train_temp.pop(i)
        else:
            files_train_final.append(file[:-9])
    
        i += 1
    
    
    files_test_temp = os.listdir( os.path.join(inDir,"Data_masks/Road/Test") )
    files_test_final = []
    
    for file in files_test_temp:
        files_test_final.append(file[:-9])
    
    return files_train_final,files_test_final


def get_testing_image_pair(files_test,imageId):
    ''' Gets the input,truth for an image.

    Description :
        Get image-ground-truth pair of the image with id "imageId" from the list of
        testing images "files_test"

    Arguments :
        files_test  -- List.
                       The list of names of the tiff images that will be used for testing
        imageId     -- String.
                       The id of the image.
                       ex. '6010_0_0'
    Returns :
        (image_train,truth) -- Tuple.
                               This tuple containing the input image and the ground truth.
    '''
    
    if imageId not in files_test:
        raise ValueError("Invalid value of imageId")
        return None

    # Using gdal to read the input image         
    reader = gdal_utils()
    path = os.path.join(os.getcwd(),"Data/image_stacks/" + imageId + ".tif")
    image_test = reader.gdal_to_nparr(path)
    
    if image_test is None:
        print("Failed to load image")
        return None
    
    path = os.path.join(os.getcwd(),"Data_masks/Road/Test/" + imageId + "_Road.tif")

    # Using gdal to read the ground truth
    truth = reader.gdal_to_nparr(path)
    if truth is None:
        print("Failed to load groung truth")
        return None
    
    return (image_test,truth)


def create_padded_image(image_test) :
    
    '''Pads the input image so that dimensions can be broken into 112x112 patches.

    Arguments :
        image_test -- np.array.
                      The test image ("image_test") to be fed.
    Returns :
        padded_image     -- np.array.
                            The test image ("image_test") padded appropriately.
        num_pics_in_rows -- Int.
                            Number of 112x112 patches that can fit in a row.
        num_pics_in_cols -- Int.
                            Number of 112x112 patches that can fit in a column.                            
    '''

    n_H0,n_W0 = image_test.shape[0],image_test.shape[1]

    # Remainder of length
    n_H0_rem = n_H0%112

    # Remainder of width
    n_W0_rem = n_W0%112
    
    print("Image test dims : {},{}".format(image_test.shape[0],image_test.shape[1]))    
    
    print("n_H0_rem : {}".format(n_H0_rem))
    print("n_W0_rem : {}".format(n_W0_rem))
    
    if (n_W0_rem != 0) and (n_H0_rem != 0) :
                
        # Add padding to the bottom border
        padded_image = np.pad(image_test,pad_width = [[0,112-n_H0_rem],[0,0],[0,0] ],mode="symmetric")
    
        # Add padding to the right border
        padded_image = np.pad(padded_image,pad_width =[[0,0],[0,112-n_W0_rem],[0,0]],mode="symmetric")
        
    # New padded dimensions
    print("Padded image dims : {},{}" .format(padded_image.shape[0],padded_image.shape[1]) )

    if (padded_image.shape[0]%112 != 0) or ( padded_image.shape[1]%112 != 0) :
        raise ValueError("Padding algorithm failed.")

    # Number of 112x112 patches that can fit in a row 
    num_pics_in_rows = padded_image.shape[0]/112

    # Number of 112x112 patches that can fit in a column
    num_pics_in_cols = padded_image.shape[1]/112

    print("Num Pics in rows : {}, Num Pics in cols : {}".format(num_pics_in_rows,num_pics_in_cols))
        
    return padded_image/2047,num_pics_in_rows,num_pics_in_cols

def create_row_input(padded_image,stride,start_row,img_rows = 112,img_cols = 112,img_channels = 9):
    ''' Creates a 

    '''
    
    num_tiles = (int)((padded_image.shape[1] - img_cols)/stride) + 1
    input_matrix = np.zeros(shape = (num_tiles,img_rows,img_cols,img_channels))

    end_row = start_row +  img_rows

    start_col = 0
    end_col = start_col + 112

    count = 0
    while (end_col <= padded_image.shape[1]) :

        #print("a :{} b : {}".format(start_col,end_col))
        
        single_image = padded_image[start_row:end_row,start_col:end_col,:img_channels]