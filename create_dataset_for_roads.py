
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:39:56 2018

@author: alex
"""

from gdal_utilities import gdal_utils
import numpy as np
import os
from imgaug import augmenters as iaa
import pickle

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
            continue
        else:
            files_train_final.append(file[:-9])
    
        i += 1
    
    
    files_test_temp = os.listdir( os.path.join(inDir,"Data_masks/Road/Test") )
    files_test_final = []
    
    for file in files_test_temp:
        files_test_final.append(file[:-9])
    
    return files_train_final,files_test_final


def get_training_image_pair(files_train,imageId):
    ''' Gets the input,truth for an image.

    Description :
        Get image-ground-truth pair of the image with id "imageId" from the list of
        training images "files_train"

    Arguments :
        files_train -- List.
                       The list of names of the tiff images that will be used for training
        imageId     -- String.
                       The id of the image.
                       ex. '6010_0_0'
    Returns :
        (image_train,truth) -- Tuple.
                               This tuple containing the input image and the ground truth.
    '''
    
    if imageId not in files_train:
        raise ValueError("Invalid value of imageId")
        return None

    # Using gdal to read the input image
    reader = gdal_utils()
    path = os.path.join(os.getcwd(),"Data/image_stacks/" + imageId + ".tif")
    image_train = reader.gdal_to_nparr(path)
    
    if image_train is None:
        print("Failed to load image. Wrong path name provided.")
        return None

    # Using gdal to read the ground truth
    path = os.path.join(os.getcwd(),"Data_masks/Road/" + imageId + "_Road.tif")
    truth = reader.gdal_to_nparr(path)
    
    if truth is None:
        print("Failed to load groung truth. Wrong path name provided.")
        return None
    
    return (image_train,truth)

def create_list_of_augmenters(flip_vertical = True, flip_horizontal = True,random_rotations = True):
    '''Creates a list of image augmenters.

    Description :
        Creates a list of image augmenters that can possibly flip an image vertically,
        horizontally and perform clockwise rotations in increments of 90 degrees out
        of [90,180,270,360].

    Arguments :
        flip_vertical    -- Bool.
                            The augmenters created will have the ability to flip
                            images vertically.
        flip_horizontal  -- Bool.
                            The augmenters created will have the ability to flip
                            images horizontally.
        random_rotations -- Bool.
                            The augmenters created will have the ability to rotate
                            images by 90,180,270 and 360 degrees.

    Returns :
        ans -- List.
               The list contains a number of image augmenters that have the capability to perform
               flips and rotations as decided by the input parameters "flip_vertical",
               "flip_horizontal" and "random_rotations". If all three parameters are "false"
               then "None" is returned.
    '''
    
    if flip_vertical and flip_horizontal and random_rotations :
        
        ans = {}

        # flip up down
        flip_ud = 1.0

        # flip left right
        flip_lr = 1.0
