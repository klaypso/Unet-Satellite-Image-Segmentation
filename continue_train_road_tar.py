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

    weight_parameters["centre_5_3_deconv"] = centre_5_3_deconv

    weight_parameters["right_4_1_conv"] = right_4_1_conv
    weight_parameters["right_4_1_conv_bias"] = right_4_1_conv_bias            
    
    weight_parameters["right_4_2_conv"] = right_4_2_conv
    weight_parameters["right_4_2_conv_bias"] = right_4_2_conv_bias

    weight_parameters["right_4_3_deconv"] = right_4_3_deconv

    weight_parameters["right_3_1_conv"] = right_3_1_conv
    weight_parameters["right_3_1_conv_bias"] = right_3_1_conv_bias        
    
    weight_parameters["right_3_2_conv"] = right_3_2_conv
    weight_parameters["right_3_2_conv_bias"] = right_3_2_conv_bias
    
    weight_parameters["right_3_3_deconv"] = right_3_3_deconv
    
    weight_parameters["right_2_1_conv"] = right_2_1_conv
    weight_parameters["right_2_1_conv_bias"] = right_2_1_conv_bias
    
    weight_parameters["right_2_2_conv"] = right_2_2_conv
    weight_parameters["right_2_2_conv_bias"] = right_2_2_conv_bias    
    
    weight_parameters["right_2_3_deconv"] = right_2_3_deconv
     
    weight_parameters["right_1_1_conv"] = right_1_1_conv
    weight_parameters["right_1_1_conv_bias"] = right_1_1_conv_bias

    weight_parameters["right_1_2_conv"] = right_1_2_conv
    weight_parameters["right_1_2_conv_bias"] = right_1_2_conv_bias
     
    return weight_parameters


def forward_prop(X,weight_parameters,bool_train = True) : 
    
    '''
    Description :
        Performs the forward propagation in the network.
        
    Arguments :
        X                 -- np.array
                             The input matrix
        weight_parameters -- dict.
                             The initialized weights for the matrix
        bool_train        -- Bool.
                             An argument passed to the batch normalization parameter, to allow the updation of batch mean and variance

    Returns :
        conv18 -- The final feature vector
    '''
    
    left_1_1_conv = weight_parameters["left_1_1_conv"] 
    left_1_2_conv = weight_parameters["left_1_2_conv"]
    
    left_2_1_conv = weight_parameters["left_2_1_conv"]
    left_2_2_conv = weight_parameters["left_2_2_conv"]
    
    left_3_1_conv = weight_parameters["left_3_1_conv"]
    left_3_2_conv = weight_parameters["left_3_2_conv"]
    
    left_4_1_conv = weight_parameters["left_4_1_conv"]
    left_4_2_conv = weight_parameters["left_4_2_conv"]
    
    centre_5_1_conv = weight_parameters["centre_5_1_conv"]
    centre_5_2_conv = weight_parameters["centre_5_2_conv"]

    left_1_1_conv_bias = weight_parameters["left_1_1_conv_bias"] 
    left_1_2_conv_bias = weight_parameters["left_1_2_conv_bias"]
    
    left_2_1_conv_bias = weight_parameters["left_2_1_conv_bias"]
    left_2_2_conv_bias = weight_parameters["left_2_2_conv_bias"]
    
    left_3_1_conv_bias = weight_parameters["left_3_1_conv_bias"]
    left_3_2_conv_bias = weight_parameters["left_3_2_conv_bias"]
    
    left_4_1_conv_bias = weight_parameters["left_4_1_conv_bias"]
    left_4_2_conv_bias = weight_parameters["left_4_2_conv_bias"]
    
    centre_5_1_conv_bias = weight_parameters["centre_5_1_conv_bias"]
    centre_5_2_conv_bias = weight_parameters["centre_5_2_conv_bias"]

    centre_5_3_deconv = weight_parameters["centre_5_3_deconv"]

    right_4_1_conv = weight_parameters["right_4_1_conv"] 
    right_4_1_conv_bias = weight_parameters["right_4_1_conv_bias"]             
    
    right_4_2_conv = weight_parameters["right_4_2_conv"] 
    right_4_2_conv_bias = weight_parameters["right_4_2_conv_bias"] 

    right_4_3_deconv = weight_parameters["right_4_3_deconv"]

    right_3_1_conv = weight_parameters["right_3_1_conv"]
    right_3_1_conv_bias = weight_parameters["right_3_1_conv_bias"]         
    
    right_3_2_conv = weight_parameters["right_3_2_conv"] 
    right_3_2_conv_bias = weight_parameters["right_3_2_conv_bias"]
    
    right_3_3_deconv = weight_parameters["right_3_3_deconv"]
    
    right_2_1_conv = weight_parameters["right_2_1_conv"]
    right_2_1_conv_bias = weight_parameters["right_2_1_conv_bias"]
    
    right_2_2_conv = weight_parameters["right_2_2_conv"] 
    right_2_2_conv_bias = weight_parameters["right_2_2_conv_bias"]   
    
    right_2_3_deconv = weight_parameters["right_2_3_deconv"]
     
    right_1_1_conv = weight_parameters["right_1_1_conv"] 
    right_1_1_conv_bias = weight_parameters["right_1_1_conv_bias"] 

    right_1_2_conv = weight_parameters["right_1_2_conv"] 
    right_1_2_conv_bias = weight_parameters["right_1_2_conv_bias"] 


    ### Left Branch 1st Layer ###
    
    
    ## INTERESTING -- TENSORFLOW DOES A BAD JOB WHEN WE WANT TO PAD AN EVEN INPUT WITH AN ODD KERNEL ##    
    with tf.name_scope("Left_Branch_1st_Layer") :
        
        with tf.name_scope("Conv_1") :
            conv1 = tf.nn.conv2d(tf.pad(X,paddings = [[0,0],[112,112],[112,112],[0,0]],mode = 'SYMMETRIC'),left_1_1_conv,strides = (1,3,3,1),padding = "VALID",name = "convolve")
            conv1 = tf.nn.bias_add(conv1,left_1_1_conv_bias,name = "bias_add")
            conv1 = tf.layers.batch_normalization(conv1,training = bool_train,name = "norm")
            conv1 = tf.nn.leaky_relu (conv1,name = "activation")
            variable_summaries_weights_biases(left_1_1_conv)
            variable_summaries_weights_biases(left_1_1_conv_bias)
    
        with tf.name_scope("Conv_2") :    
            conv2 = tf.nn.conv2d(tf.pad(conv1,paddings = [[0,0],[112,112],[112,112],[0,0]],mode = 'SYMMETRIC'), left_1_2_conv, (1,3,3,1), padding = "VALID",name = "convolve")
            conv2 = tf.nn.bias_add(conv2,left_1_2_conv_bias,name = "bias_add")
            conv2 = tf.layers.batch_normalization(conv2,training = bool_train,name = "norm_2")
            conv2 =  tf.nn.leaky_relu(conv2,name = "activation")
            variable_summaries_weights_biases(left_1_2_conv)
            variable_summaries_weights_biases(left_1_2_conv_bias)
        
        with tf.name_scope("Pool") :
            max_pool_1 = tf.nn.max_pool(tf.pad(conv2,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),ksize = (1,2,2,1), strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    
    ### Left Branch 2nd layer ###
    
    with tf.name_scope("Left_Branch_2nd_Layer") :   

        with tf.name_scope("Conv_1") :
            conv3 = tf.nn.conv2d(tf.pad(max_pool_1,paddings = [[0,0],[64,64],[64,64],[0,0]],mode = 'SYMMETRIC'),left_2_1_conv, (1,3,3,1), padding = "VALID",name = "convolve")
            conv3 = tf.nn.bias_add(conv3,left_2_1_conv_bias,name = "bias_add")
            conv3 = tf.layers.batch_normalization(conv3,training = bool_train,name = "norm_3")
            conv3 =  tf.nn.leaky_relu(conv3,name = "activation")
            variable_summaries_weights_biases(left_2_1_conv)
            variable_summaries_weights_biases(left_2_1_conv_bias)

        with tf.name_scope("Conv_2") :
            conv4 = tf.nn.conv2d(tf.pad(conv3,paddings = [[0,0],[64,64],[64,64],[0,0]],mode = 'SYMMETRIC'),left_2_2_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv4 = tf.nn.bias_add(conv4,left_2_2_conv_bias,name = "bias_add")
            conv4 = tf.layers.batch_normalization(conv4,training = bool_train,name = "norm_4")
            conv4 =  tf.nn.leaky_relu(conv4,name = "activation")
            variable_summaries_weights_biases(left_2_2_conv)
            variable_summaries_weights_biases(left_2_2_conv_bias)

        with tf.name_scope("Pool") :
            max_pool_2 = tf.nn.max_pool(conv4,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")

    
    ### Left Branch 3rd layer ###
    
    with tf.name_scope("Left_Branch_3rd_Layer") :
    
        with tf.name_scope("Conv_1") :
            conv5 = tf.nn.conv2d(tf.pad(max_pool_2,paddings = [[0,0],[32,32],[32,32],[0,0]],mode = 'SYMMETRIC'),left_3_1_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv5 = tf.nn.bias_add(conv5,left_3_1_conv_bias,name = "bias_add")
            conv5 = tf.layers.batch_normalization(conv5,training = bool_train,name = "norm_5")
            conv5 = tf.nn.leaky_relu(conv5,name = "activation")
            variable_summaries_weights_biases(left_3_1_conv)
            variable_summaries_weights_biases(left_3_1_conv_bias)

        with tf.name_scope("Conv_2") :
            conv6 = tf.nn.conv2d(tf.pad(conv5,paddings = [[0,0],[32,32],[32,32],[0,0]],mode = 'SYMMETRIC'),left_3_2_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv6 = tf.nn.bias_add(conv6,left_3_2_conv_bias,name = "bias_add")
            conv6 = tf.layers.batch_normalization(conv6,training = bool_train,name = "norm_6")
            conv6 = tf.nn.leaky_relu(conv6,name = "activation")
            variable_summaries_weights_biases(left_3_2_conv)
            variable_summaries_weights_biases(left_3_2_conv_bias)

        with tf.name_scope("Pool") :
            max_pool_3 = tf.nn.max_pool(conv6,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    ### Left Branch 4th layer ###
    
    with tf.name_scope("Left_Branch_4th_Layer"):
        
        with tf.name_scope("Conv_1") :
            conv7 = tf.nn.conv2d(tf.pad(max_pool_3,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),left_4_1_conv,(1,3,3,1),padding = "VALID",name = "convolve")
            conv7 = tf.nn.bias_add(conv7,left_4_1_conv_bias,name = "bias_add")
            conv7 = tf.layers.batch_normalization(conv7,training = bool_train,name = "norm_7")
            conv7 =  tf.nn.leaky_relu(conv7,name = "activation")
            variable_summaries_weights_biases(left_4_1_conv)
            variable_summaries_weights_biases(left_4_1_conv_bias)
            
        with tf.name_scope("Conv_2") :
            conv8 = tf.nn.conv2d(tf.pad(conv7,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),left_4_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv8 = tf.nn.bias_add(conv8,left_4_2_conv_bias,name = "bias_add")
            conv8 = tf.layers.batch_normalization(conv8,training = bool_train,name = "norm_8")
            conv8 =  tf.nn.leaky_relu(conv8,name = "activation")
            variable_summaries_weights_biases(left_4_2_conv)
            variable_summaries_weights_biases(left_4_2_conv_bias)

        with tf.name_scope("Pool") :
            max_pool_4 = tf.nn.max_pool(conv8,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    
    ### Centre Branch ###
    
    with tf.name_scope("Centre_Branch"):
        
        with tf.name_scope("Conv_1") :
            
            conv9 = tf.nn.conv2d(tf.pad(max_pool_4,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),centre_5_1_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv9 = tf.nn.bias_add(conv9,centre_5_1_conv_bias,name = "bias_add")
            conv9 = tf.layers.batch_normalization(conv9,training = bool_train,name = "norm_9")
            conv9 =  tf.nn.leaky_relu(conv9,name = "activation")
            variable_summaries_weights_biases(centre_5_1_conv) 
            variable_summaries_weights_biases(centre_5_1_conv_bias)
            
        with tf.name_scope("Conv_2") :
        
            conv10 = tf.nn.conv2d(tf.pad(conv9,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),centre_5_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv10 = tf.nn.bias_add(conv10,centre_5_2_conv_bias,name = "bias_add")
            conv10 = tf.layers.batch_normalization(conv10,training = bool_train,name = "norm_10")
            conv10 =  tf.nn.leaky_relu(conv10,name = "activation")
            variable_summaries_weights_biases(centre_5_2_conv)
            variable_summaries_weights_biases(centre_5_2_conv_bias)

            conv10_obj = convolution(conv9.shape[1],conv9.shape[2],conv9.shape[3],centre_5_2_conv.shape[0],centre_5_2_conv.shape[1],centre_5_2_conv.shape[3],3,3,conv9.shape[1],conv9.shape[2])
            de_conv10_obj = trans_convolve(None,True,conv10_obj.output_h,conv10_obj.output_w,conv10_obj.output_d,kernel_h = 2,kernel_w = 2,kernel_d =128,stride_h = 2,stride_w = 2,padding = 'VALID')   
          
        with tf.name_scope("Deconvolve") : 
            de_conv10  = tf.nn.conv2d_transpose(conv10,centre_5_3_deconv, output_shape = (tf.shape(X)[0],de_conv10_obj.output_h,de_conv10_obj.output_w,de_conv10_obj.output_d), strides = (1,2,2,1),padding = 'VALID',name = "deconv")
            variable_summaries_weights_biases(centre_5_3_deconv)   

    ### Right Branch 4th layer ###
    
    with tf.name_scope("Merging") :
    
        merge1 = tf.concat([de_conv10,conv8],axis = 3,name = "merge")   
    
    
    with tf.name_scope("Right_Branch_4th_Layer"):
    
        with tf.name_scope("Conv_1") :
            
            conv11 = tf.nn.conv2d(tf.pad(merge1,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),right_4_1_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv11 = tf.nn.bias_add(conv11,right_4_1_conv_bias,name = "bias_add")
            conv11 = tf.layers.batch_normalization(conv11,training = bool_train,name = "norm_11")
            conv11 =  tf.nn.leaky_relu(conv11,name = "activation")
            variable_summaries_weights_biases(right_4_1_conv)

        with tf.name_scope("Conv_2") :
    
            conv12 = tf.nn.conv2d(tf.pad(conv11,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),right_4_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv12 = tf.nn.bias_add(conv12,right_4_2_conv_bias,name = "bias_add")
            conv12 = tf.layers.batch_normalization(conv12,training = bool_train,name = "norm_12")
            conv12 =  tf.nn.leaky_relu(conv12,name = "activation")
            variable_summaries_weights_biases(right_4_2_conv)
            variable_summaries_weights_biases(right_4_2_conv_bias)

            conv12_obj = convolution(conv11.shape[1],conv11.shape[2],conv11.shape[3],right_4_2_conv.shape[0],right_4_2_conv.shape[1],right_4_2_conv.shape[3],3,3,conv11.shape[1],conv11.shape[2])                
            de_conv12_obj = trans_convolve(None,True,conv12_obj.output_h,conv12_obj.output_w,conv12_obj.output_d,kernel_h = 2,kernel_w = 2,kernel_d = 256,stride_h = 2,stride_w = 2,padding = 'VALID')   
    
        with tf.name_scope("Deconvolve") :  