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
        
    Returns :
        weigths -- The populated bilinear filter
    '''
    
    kernel_size = filter_shape[1]
    
    # Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5
 
    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location)/ upscale_factor)) * (1 - abs((y - centre_location)/ upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    
    for k in range(filter_shape[3]):
        for i in range(filter_shape[2]):
            weights[:, :, i, k] = bilinear
        
    return weights    

def variable_summaries_weights_biases(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    tf.summary.histogram('histogram',var)

def variable_summaries_scalars(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    tf.summary.scalar('value',var)

def create_placeholders(n_H0,n_W0,n_C0):
    """
    Creates the placeholders for the input size and for the number of output classes.
    
    Arguments:
    n_W0 -- scalar, width of an input matrix
    n_C0 -- scalar, number of channels of the input
    n_y  -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    
    with tf.name_scope("Inputs") :
        # Keep the number of examples as a variable (None) and the height of the matrix as variables (None)
        X = tf.placeholder(dtype = tf.float32, shape = (None,n_H0,n_W0,n_C0), name = "X") 
        Y = tf.placeholder(dtype = tf.float32, shape = (None,n_H0,n_W0,1), name = "Y")
    
    
    return X,Y


def initialize_parameters():
    '''
    Description:
        Initialize weight parameters for the weight matrix.

    Returns: 
        weight_parameters - A dictionary containing all the weights of the neural network
    '''
    
    left_1_1_conv = tf.get_variable(name = "Road_tar_left_1_1_conv",shape = (3,3,9,32),dtype = tf.float32,trainable = True)
    left_1_1_conv_bias = tf.get_variable(name = "Road_tar_left_1_1_conv_bias",shape = (32),dtype = tf.float32,trainable = True)
    
    left_1_2_conv = tf.get_variable(name = "Road_tar_left_1_2_conv",shape = (3,3,32,32),dtype = tf.float32,trainable = True)
    left_1_2_conv_bias = tf.get_variable(name = "Road_tar_left_1_2_conv_bias",shape = (32),dtype = tf.float32,trainable = True)

    left_2_1_conv = tf.get_variable(name = "Road_tar_left_2_1_conv",shape = (3,3,32,64),dtype = tf.float32,trainable = True)
    left_2_1_conv_bias = tf.get_variable(name = "Road_tar_left_2_1_conv_bias",shape = (64),dtype = tf.float32,trainable = True)
    
    left_2_2_conv = tf.get_variable(name = "Road_tar_left_2_2_conv",shape = (3,3,64,64),dtype = tf.float32,trainable = True)
    left_2_2_conv_bias = tf.get_variable(name = "Road_tar_left_2_2_conv_bias",shape = (64),dtype = tf.float32,trainable = True)    
    
    left_3_1_conv = tf.get_variable(name = "Road_tar_left_3_1_conv",shape = (3,3,64,128),dtype = tf.float32,trainable = True)
    left_3_1_conv_bias = tf.get_variable(name = "Road_tar_left_3_1_conv_bias",shape = (128),dtype = tf.float32,trainable = True)

    left_3_2_conv = tf.get_variable(name = "Road_tar_left_3_2_conv",shape = (3,3,128,128),dtype = tf.float32,trainable = True)
    left_3_2_conv_bias = tf.get_variable(name = "Road_tar_left_3_2_conv_bias",shape = (128),dtype = tf.float32,trainable = True)
    
    left_4_1_conv = tf.get_variable(name = "Road_tar_left_4_1_conv",shape = (3,3,128,256),dtype = tf.float32,trainable = True)
    left_4_1_conv_bias = tf.get_variable(name = "Road_tar_left_4_1_conv_bias",shape = (256),dtype = tf.float32,trainable = True)    
    
    left_4_2_conv = tf.get_variable(name = "Road_tar_left_4_2_conv",shape = (3,3,256,256),dtype = tf.float32,trainable = True)
    left_4_2_conv_bias = tf.get_variable(name = "Road_tar_left_4_2_conv_bias",shape = (256),dtype = tf.float32,trainable = True)        
    
    centre_5_1_conv = tf.get_variable(name = "Road_tar_centre_5_1_conv",shape = (3,3,256,512),dtype = tf.float32,trainable = True)
    centre_5_1_conv_bias = tf.get_variable(name = "Road_tar_centre_5_1_conv_bias",shape = (512),dtype = tf.float32,trainable = True)    
    
    centre_5_2_conv = tf.get_variable(name = "Road_tar_centre_5_2_conv",shape = (3,3,512,512),dtype = tf.float32,trainable = True)
    centre_5_2_conv_bias = tf.get_variable(name = "Road_tar_centre_5_2_conv_bias",shape = (512),dtype = tf.float32,trainable = True)

    centre_5_3_deconv = tf.get_variable(name = "Road_tar_centre_5_3_deconv",shape = (2,2,128,512),dtype = tf.float32,trainable = False)         

    right_4_1_conv = tf.get_variable(name = "Road_tar_right_4_1_conv",shape = (3,3,128 + 256,256),dtype = tf.float32,trainable = True)
    right_4_1_conv_bias = tf.get_variable(name = "Road_tar_right_4_1_conv_bias",shape = (256),dtype = tf.float32,trainable = True)
    
    right_4_2_conv = tf.get_variable(name = "Road_tar_right_4_2_conv",shape = (3,3,256,256),dtype = tf.float32,trainable = True)
    right_4_2_conv_bias = tf.get_variable(name = "Road_tar_right_4_2_conv_bias",shape = (256),dtype = tf.float32,trainable = True)

    right_4_3_deconv = tf.get_variable(name = "Road_tar_right_4_3_deconv",shape = (2,2,256,256),dtype = tf.float32,trainable = False)         
    
    right_3_1_conv = tf.get_variable(name = "Road_tar_right_3_1_conv",shape = (3,3,128 + 256,128),dtype = tf.float32,trainable = True)
    right_3_1_conv_bias = tf.get_variable(name = "Road_tar_right_3_1_conv_bias",shape = (128),dtype = tf.float32,trainable = True)
    
    right_3_2_conv = tf.get_variable(name = "Road_tar_right_3_2_conv",shape = (3,3,128,128),dtype = tf.float32,trainable = True)
    right_3_2_conv_bias = tf.get_variable(name = "Road_tar_right_3_2_conv_bias",shape = (128),dtype = tf.float32,trainable = True)

    right_3_3_deconv = tf.get_variable(name  = "Road_tar_right_3_3_deconv", shape = (2,2,128,128),dtype = tf.float32,trainable = False)

    right_2_1_conv = tf.get_variable(name = "Road_tar_right_2_1_conv",shape = (3,3,128 + 64,64),dtype = tf.float32,trainable = True)
    right_2_1_conv_bias = tf.get_variable(name = "Road_tar_right_2_1_conv_bias",shape = (64),dtype = tf.float32,trainable = True)
    
    right_2_2_conv = tf.get_variable(name = "Road_tar_right_2_2_conv",shape = (3,3,64,64),dtype = tf.float32,trainable = True)
    right_2_2_conv_bias = tf.get_variable(name = "Road_tar_right_2_2_conv_bias",shape = (64),dtype = tf.float32,trainable = True)

    right_2_3_deconv = tf.get_variable(name = "Road_tar_right_2_3_deconv",shape = (2,2,64,64),dtype = tf.float32,trainable = False)

    right_1_1_conv = tf.get_variable(name = "Road_tar_right_1_1_conv",shape = (9,9,64+32,32),dtype = tf.float32,trainable = True)
    right_1_1_conv_bias = tf.get_variable(name = "Road_tar_right_1_1_conv_bias",shape = (32),dtype = tf.float32,trainable = True)
    
    right_1_2_conv = tf.get_variable(name = "Road_tar_right_1_2_conv",shape = (9,9,32,1),dtype = tf.float32,trainable = True)
    right_1_2_conv_bias = tf.get_variable(name = "Road_tar_right_1_2_conv_bias",shape = (1),dtype = tf.float32,trainable = True)
    
    weight_parameters = {}

    weight_parameters["left_1_1_conv"] = left_1_1_conv
    weight_parameters["left_1_1_conv_bias"] = left_1_1_conv_bias
    
    weight_parameters["left_1_2_conv"] = left_1_2_conv
    weight_parameters["left_1_2_conv_bias"] = left_1_2_conv_bias

    weight_parameters["left_2_1_conv"] = left_2_1_conv
    weight_parameters["left_2_1_conv_bias"] = left_2_1_conv_bias    
    
    weight_parameters["left_2_2_conv"] = left_2_2_conv
    weight_parameters["left_2_2_conv_bias"] = left_2_2_conv_bias    

    weight_parameters["left_3_1_conv"] = left_3_1_conv
    weight_parameters["left_3_1_conv_bias"] = left_3_1_conv_bias        
    
    weight_parameters["left_3_2_conv"] = left_3_2_conv
    weight_parameters["left_3_2_conv_bias"] = left_3_2_conv_bias        

    weight_parameters["left_4_1_conv"] = left_4_1_conv
    weight_parameters["left_4_1_conv_bias"] = left_4_1_conv_bias            
    
    weight_parameters["left_4_2_conv"] = left_4_2_conv
    weight_parameters["left_4_2_conv_bias"] = left_4_2_conv_bias            
        
    weight_parameters["centre_5_1_conv"] = centre_5_1_conv
    weight_parameters["centre_5_1_conv_bias"] = centre_5_1_conv_bias                
    
    weight_parameters["centre_5_2_conv"] = centre_5_2_conv
    weight_parameters["centre_5_2_conv_bias"] = centre_5_2_conv_bias                

    weight_parameters["