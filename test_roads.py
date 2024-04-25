
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
        input_matrix[count] = single_image

        start_col += stride
        end_col = start_col + 112

        count += 1

    assert(end_col-stride == padded_image.shape[1])

    return input_matrix

def overlay_ans(final_ans,input_matrix,padded_image,stride,start_row,img_rows = 112,img_cols = 112,img_channels = 9) :

    end_row = start_row +  img_rows

    start_col = 0
    end_col = start_col + 112

    count = 0
    while (end_col <= padded_image.shape[1]) :

        final_ans[start_row:end_row,start_col:end_col,:img_channels] += input_matrix[count]
        count += 1

        start_col += stride
        end_col = start_col + 112

    assert(end_col-stride == padded_image.shape[1])

    return final_ans

def create_count_grid(padded_image,stride) :
    
    height = padded_image.shape[0]
    width  = padded_image.shape[1]
    
    count_matrix = np.zeros(shape = (height,width,1) )
    add_matrix = np.ones(shape = (112,112))
    
    row_start = 0
    row_end = 112
    
    while( row_end <= height ):
    
        col_start  = 0
        col_end    = col_start + 112

        while( col_end <= width ) :
            
            count_matrix[row_start:row_end,col_start:col_end,0] += add_matrix[:,:]
            
            col_start += stride
            col_end = col_start + 112
        
        
        row_start += stride
        row_end = row_start + 112

    return count_matrix

def normalize_image(final_ans,count_matrix) :
    
    assert(final_ans.shape == count_matrix.shape)
    
    average = np.divide(final_ans,count_matrix)

    return average

############################################ UTILITY FUNCTIONS ############################################
    

############################################ NETWORK BUILDING ############################################
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
        
    left_1_1_conv = tf.get_variable(name = "Road_tar_left_1_1_conv",shape = (3,3,9,32),dtype = tf.float32)
    left_1_1_conv_bias = tf.get_variable(name = "Road_tar_left_1_1_conv_bias",shape = (32),dtype = tf.float32)
    
    left_1_2_conv = tf.get_variable(name = "Road_tar_left_1_2_conv",shape = (3,3,32,32),dtype = tf.float32)
    left_1_2_conv_bias = tf.get_variable(name = "Road_tar_left_1_2_conv_bias",shape = (32),dtype = tf.float32)

    left_2_1_conv = tf.get_variable(name = "Road_tar_left_2_1_conv",shape = (3,3,32,64),dtype = tf.float32)
    left_2_1_conv_bias = tf.get_variable(name = "Road_tar_left_2_1_conv_bias",shape = (64),dtype = tf.float32)
    
    left_2_2_conv = tf.get_variable(name = "Road_tar_left_2_2_conv",shape = (3,3,64,64),dtype = tf.float32)
    left_2_2_conv_bias = tf.get_variable(name = "Road_tar_left_2_2_conv_bias",shape = (64),dtype = tf.float32)    
    
    left_3_1_conv = tf.get_variable(name = "Road_tar_left_3_1_conv",shape = (3,3,64,128),dtype = tf.float32)
    left_3_1_conv_bias = tf.get_variable(name = "Road_tar_left_3_1_conv_bias",shape = (128),dtype = tf.float32)

    left_3_2_conv = tf.get_variable(name = "Road_tar_left_3_2_conv",shape = (3,3,128,128),dtype = tf.float32)
    left_3_2_conv_bias = tf.get_variable(name = "Road_tar_left_3_2_conv_bias",shape = (128),dtype = tf.float32)
    
    left_4_1_conv = tf.get_variable(name = "Road_tar_left_4_1_conv",shape = (3,3,128,256),dtype = tf.float32)
    left_4_1_conv_bias = tf.get_variable(name = "Road_tar_left_4_1_conv_bias",shape = (256),dtype = tf.float32)    
    
    left_4_2_conv = tf.get_variable(name = "Road_tar_left_4_2_conv",shape = (3,3,256,256),dtype = tf.float32)
    left_4_2_conv_bias = tf.get_variable(name = "Road_tar_left_4_2_conv_bias",shape = (256),dtype = tf.float32)        
    
    centre_5_1_conv = tf.get_variable(name = "Road_tar_centre_5_1_conv",shape = (3,3,256,512),dtype = tf.float32)
    centre_5_1_conv_bias = tf.get_variable(name = "Road_tar_centre_5_1_conv_bias",shape = (512),dtype = tf.float32)    
    
    centre_5_2_conv = tf.get_variable(name = "Road_tar_centre_5_2_conv",shape = (3,3,512,512),dtype = tf.float32)
    centre_5_2_conv_bias = tf.get_variable(name = "Road_tar_centre_5_2_conv_bias",shape = (512),dtype = tf.float32)

    centre_5_3_deconv = tf.get_variable(name = "Road_tar_centre_5_3_deconv",shape = (2,2,128,512),dtype = tf.float32)         

    right_4_1_conv = tf.get_variable(name = "Road_tar_right_4_1_conv",shape = (3,3,128 + 256,256),dtype = tf.float32)
    right_4_1_conv_bias = tf.get_variable(name = "Road_tar_right_4_1_conv_bias",shape = (256),dtype = tf.float32)
    
    right_4_2_conv = tf.get_variable(name = "Road_tar_right_4_2_conv",shape = (3,3,256,256),dtype = tf.float32)
    right_4_2_conv_bias = tf.get_variable(name = "Road_tar_right_4_2_conv_bias",shape = (256),dtype = tf.float32)

    right_4_3_deconv = tf.get_variable(name = "Road_tar_right_4_3_deconv",shape = (2,2,256,256),dtype = tf.float32)         
    
    right_3_1_conv = tf.get_variable(name = "Road_tar_right_3_1_conv",shape = (3,3,128 + 256,128),dtype = tf.float32)
    right_3_1_conv_bias = tf.get_variable(name = "Road_tar_right_3_1_conv_bias",shape = (128),dtype = tf.float32)
    
    right_3_2_conv = tf.get_variable(name = "Road_tar_right_3_2_conv",shape = (3,3,128,128),dtype = tf.float32)
    right_3_2_conv_bias = tf.get_variable(name = "Road_tar_right_3_2_conv_bias",shape = (128),dtype = tf.float32)

    right_3_3_deconv = tf.get_variable(name  = "Road_tar_right_3_3_deconv", shape = (2,2,128,128),dtype = tf.float32)

    right_2_1_conv = tf.get_variable(name = "Road_tar_right_2_1_conv",shape = (3,3,128 + 64,64),dtype = tf.float32)
    right_2_1_conv_bias = tf.get_variable(name = "Road_tar_right_2_1_conv_bias",shape = (64),dtype = tf.float32)
    
    right_2_2_conv = tf.get_variable(name = "Road_tar_right_2_2_conv",shape = (3,3,64,64),dtype = tf.float32)
    right_2_2_conv_bias = tf.get_variable(name = "Road_tar_right_2_2_conv_bias",shape = (64),dtype = tf.float32)
    
    right_2_3_deconv = tf.get_variable(name = "Road_tar_right_2_3_deconv",shape = (2,2,64,64),dtype = tf.float32)

    right_1_1_conv = tf.get_variable(name = "Road_tar_right_1_1_conv",shape = (9,9,64+32,32),dtype = tf.float32)
    right_1_1_conv_bias = tf.get_variable(name = "Road_tar_right_1_1_conv_bias",shape = (32),dtype = tf.float32)
    
    right_1_2_conv = tf.get_variable(name = "Road_tar_right_1_2_conv",shape = (9,9,32,1),dtype = tf.float32)
    right_1_2_conv_bias = tf.get_variable(name = "Road_tar_right_1_2_conv_bias",shape = (1),dtype = tf.float32)
    
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
        X -- The input matrix
        weight_parameters -- The initialized weights for the matrix
        bool_train -- An argument passed to the batch normalization parameter, to allow the updation of batch mean and variance
        
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
    
    ## INTERESTING -- TENSORFLOW DOES A BAD JOB WHEN WE WANT TO PAD AN EVEN INPUT WITH AN ODD KERNEL ##    
    with tf.name_scope("Left_Branch_1st_Layer") :
        
        with tf.name_scope("Conv_1") :
            conv1 = tf.nn.conv2d(tf.pad(X,paddings = [[0,0],[112,112],[112,112],[0,0]],mode = 'SYMMETRIC'),left_1_1_conv,strides = (1,3,3,1),padding = "VALID",name = "convolve")
            conv1 = tf.nn.bias_add(conv1,left_1_1_conv_bias,name = "bias_add")
            conv1 = tf.layers.batch_normalization(conv1,training = bool_train,name = "norm")
            conv1 = tf.nn.leaky_relu (conv1,name = "activation")
    
        with tf.name_scope("Conv_2") :    
            conv2 = tf.nn.conv2d(tf.pad(conv1,paddings = [[0,0],[112,112],[112,112],[0,0]],mode = 'SYMMETRIC'), left_1_2_conv, (1,3,3,1), padding = "VALID",name = "convolve")
            conv2 = tf.nn.bias_add(conv2,left_1_2_conv_bias,name = "bias_add")
            conv2 = tf.layers.batch_normalization(conv2,training = bool_train,name = "norm_2")
            conv2 =  tf.nn.leaky_relu(conv2,name = "activation")
        
        with tf.name_scope("Pool") :
            max_pool_1 = tf.nn.max_pool(tf.pad(conv2,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),ksize = (1,2,2,1), strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    
    ### Left Branch 2nd layer ###
    
    with tf.name_scope("Left_Branch_2nd_Layer") :   

        with tf.name_scope("Conv_1") :
            conv3 = tf.nn.conv2d(tf.pad(max_pool_1,paddings = [[0,0],[64,64],[64,64],[0,0]],mode = 'SYMMETRIC'),left_2_1_conv, (1,3,3,1), padding = "VALID",name = "convolve")
            conv3 = tf.nn.bias_add(conv3,left_2_1_conv_bias,name = "bias_add")
            conv3 = tf.layers.batch_normalization(conv3,training = bool_train,name = "norm_3")
            conv3 =  tf.nn.leaky_relu(conv3,name = "activation")

        with tf.name_scope("Conv_2") :
            conv4 = tf.nn.conv2d(tf.pad(conv3,paddings = [[0,0],[64,64],[64,64],[0,0]],mode = 'SYMMETRIC'),left_2_2_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv4 = tf.nn.bias_add(conv4,left_2_2_conv_bias,name = "bias_add")
            conv4 = tf.layers.batch_normalization(conv4,training = bool_train,name = "norm_4")
            conv4 =  tf.nn.leaky_relu(conv4,name = "activation")

        with tf.name_scope("Pool") :
            max_pool_2 = tf.nn.max_pool(conv4,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")

    
    ### Left Branch 3rd layer ###
    
    with tf.name_scope("Left_Branch_3rd_Layer") :
    
        with tf.name_scope("Conv_1") :
            conv5 = tf.nn.conv2d(tf.pad(max_pool_2,paddings = [[0,0],[32,32],[32,32],[0,0]],mode = 'SYMMETRIC'),left_3_1_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv5 = tf.nn.bias_add(conv5,left_3_1_conv_bias,name = "bias_add")
            conv5 = tf.layers.batch_normalization(conv5,training = bool_train,name = "norm_5")
            conv5 = tf.nn.leaky_relu(conv5,name = "activation")

        with tf.name_scope("Conv_2") :
            conv6 = tf.nn.conv2d(tf.pad(conv5,paddings = [[0,0],[32,32],[32,32],[0,0]],mode = 'SYMMETRIC'),left_3_2_conv, (1,3,3,1), padding = 'VALID',name = "convolve")
            conv6 = tf.nn.bias_add(conv6,left_3_2_conv_bias,name = "bias_add")
            conv6 = tf.layers.batch_normalization(conv6,training = bool_train,name = "norm_6")
            conv6 = tf.nn.leaky_relu(conv6,name = "activation")

        with tf.name_scope("Pool") :
            max_pool_3 = tf.nn.max_pool(conv6,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    ### Left Branch 4th layer ###
    
    with tf.name_scope("Left_Branch_4th_Layer"):
        
        with tf.name_scope("Conv_1") :
            conv7 = tf.nn.conv2d(tf.pad(max_pool_3,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),left_4_1_conv,(1,3,3,1),padding = "VALID",name = "convolve")
            conv7 = tf.nn.bias_add(conv7,left_4_1_conv_bias,name = "bias_add")
            conv7 = tf.layers.batch_normalization(conv7,training = bool_train,name = "norm_7")
            conv7 =  tf.nn.leaky_relu(conv7,name = "activation")
            
        with tf.name_scope("Conv_2") :
            conv8 = tf.nn.conv2d(tf.pad(conv7,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),left_4_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv8 = tf.nn.bias_add(conv8,left_4_2_conv_bias,name = "bias_add")
            conv8 = tf.layers.batch_normalization(conv8,training = bool_train,name = "norm_8")
            conv8 =  tf.nn.leaky_relu(conv8,name = "activation")

        with tf.name_scope("Pool") :
            max_pool_4 = tf.nn.max_pool(conv8,ksize = (1,2,2,1),strides = (1,2,2,1),padding = "VALID",name = "max_pool")
    
    
    ### Centre Branch ###
    
    with tf.name_scope("Centre_Branch"):
        
        with tf.name_scope("Conv_1") :
            
            conv9 = tf.nn.conv2d(tf.pad(max_pool_4,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),centre_5_1_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv9 = tf.nn.bias_add(conv9,centre_5_1_conv_bias,name = "bias_add")
            conv9 = tf.layers.batch_normalization(conv9,training = bool_train,name = "norm_9")
            conv9 =  tf.nn.leaky_relu(conv9,name = "activation")
            
        with tf.name_scope("Conv_2") :
        
            conv10 = tf.nn.conv2d(tf.pad(conv9,paddings = [[0,0],[8,8],[8,8],[0,0]],mode = 'SYMMETRIC'),centre_5_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv10 = tf.nn.bias_add(conv10,centre_5_2_conv_bias,name = "bias_add")
            conv10 = tf.layers.batch_normalization(conv10,training = bool_train,name = "norm_10")
            conv10 =  tf.nn.leaky_relu(conv10,name = "activation")

            conv10_obj = convolution(conv9.shape[1],conv9.shape[2],conv9.shape[3],centre_5_2_conv.shape[0],centre_5_2_conv.shape[1],centre_5_2_conv.shape[3],3,3,conv9.shape[1],conv9.shape[2])
            de_conv10_obj = trans_convolve(None,True,conv10_obj.output_h,conv10_obj.output_w,conv10_obj.output_d,kernel_h = 2,kernel_w = 2,kernel_d =128,stride_h = 2,stride_w = 2,padding = 'VALID')   
          
        with tf.name_scope("Deconvolve") : 
            de_conv10  = tf.nn.conv2d_transpose(conv10,centre_5_3_deconv, output_shape = (tf.shape(X)[0],de_conv10_obj.output_h,de_conv10_obj.output_w,de_conv10_obj.output_d), strides = (1,2,2,1),padding = 'VALID',name = "deconv")

    ### Right Branch 4th layer ###
    
    with tf.name_scope("Merging") :
    
        merge1 = tf.concat([de_conv10,conv8],axis = 3,name = "merge")   
    
    
    with tf.name_scope("Right_Branch_4th_Layer"):
    
        with tf.name_scope("Conv_1") :
            
            conv11 = tf.nn.conv2d(tf.pad(merge1,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),right_4_1_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv11 = tf.nn.bias_add(conv11,right_4_1_conv_bias,name = "bias_add")
            conv11 = tf.layers.batch_normalization(conv11,training = bool_train,name = "norm_11")
            conv11 =  tf.nn.leaky_relu(conv11,name = "activation")

        with tf.name_scope("Conv_2") :
    
            conv12 = tf.nn.conv2d(tf.pad(conv11,paddings = [[0,0],[16,16],[16,16],[0,0]],mode = 'SYMMETRIC'),right_4_2_conv,(1,3,3,1),padding = 'VALID',name = "convolve")
            conv12 = tf.nn.bias_add(conv12,right_4_2_conv_bias,name = "bias_add")
            conv12 = tf.layers.batch_normalization(conv12,training = bool_train,name = "norm_12")
            conv12 =  tf.nn.leaky_relu(conv12,name = "activation")

            conv12_obj = convolution(conv11.shape[1],conv11.shape[2],conv11.shape[3],right_4_2_conv.shape[0],right_4_2_conv.shape[1],right_4_2_conv.shape[3],3,3,conv11.shape[1],conv11.shape[2])                
            de_conv12_obj = trans_convolve(None,True,conv12_obj.output_h,conv12_obj.output_w,conv12_obj.output_d,kernel_h = 2,kernel_w = 2,kernel_d = 256,stride_h = 2,stride_w = 2,padding = 'VALID')   
    
        with tf.name_scope("Deconvolve") :    
            de_conv12 = tf.nn.conv2d_transpose(conv12,right_4_3_deconv,output_shape = (tf.shape(X)[0],de_conv12_obj.output_h,de_conv12_obj.output_w,de_conv12_obj.output_d), strides = (1,2,2,1),padding = 'VALID',name = "deconv")
    
    ### Right Branch 3rd layer ###
    