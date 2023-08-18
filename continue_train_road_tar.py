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
import pickle
from sklearn.preprocessing import StandardScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

############################################ NETWORK BUILDING ############################################
def get_bilinear_filter(filter_shape, upscale_factor):
    '''
    Description :  Generates a filter than performs simple bilinear interpolation for a given upsacle_factor
    
    Arguments:
        filter_shape -- [width, height, num_in_channels, num_out_channels] -> num_in_channels = num_out_channels
        upscale_factor -- The number of times you want to scale the image.
  