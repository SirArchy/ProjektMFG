import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import scipy.signal as sg # type: ignore



def filter_data(data_matrix, window_length, poly_order):
    ''' Glättet mit dem Savitzky-Golay-Filter aus scipy.signal alle Signale in der übergebenen Datenmatrix.
        Datenmatrix muss vorher genormt worden sein. 

        input:  data_matrix (np array), der zuglättenden Spektren.
                window_length (int), positive, ungerade Zahl 
                poly_order (int), muss kleiner sein als window_length 
             
        output: new_data_matrix (np array), enthält die geglätteten Signale, selben Dimensionen wie input Matrix
    '''
    new_data_matrix = sg.savgol_filter(data_matrix[...,i], window_length, poly_order) for i in range(data_matrix.shape[-1]), axis=-1))
      
    return new_data_matrix

