
from gdal_utilities import gdal_utils
import numpy as np
import os
from sklearn.externals import joblib
import pickle
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.ndimage import binary_opening,binary_closing

def get_masks_list():
    
    '''
    Gets the list of masked tiff files intended for training and testing 
    '''
    
    inDir = os.getcwd()    
    
    files_train_temp = os.listdir(os.path.join(inDir,"Data_masks/Fast_H20"))
    files_train_final = []
    
    i = 0 
    for file in files_train_temp : 
        
        extension = os.path.splitext(file) 
    
        if extension[1] == '.tif' :
            files_train_final.append(file[:-13])
    
        i += 1
    
    
    files_test_temp = os.listdir( os.path.join(inDir,"Data_masks/Fast_H20/Test") )
    files_test_final = []