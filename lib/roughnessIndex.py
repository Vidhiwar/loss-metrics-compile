#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import ndimage


def compute_roughness_index(mask):
    """
    This function is used to calculate the roughness index of a 3D  surface.
    The roughness index is the angle that a boundary element in one layer 
    makes with the corresponding boundary element of another layer.
    
    args:
    mask: a 3D numpy vector that is the mask of an object.
    
    Returns:
    roughness_index: a 4D numpy vector countaining the normal angle of each 
    boundary element in one layer to the corresponding neighbouring layers boundary elements.
    """
    mask = mask.astype(bool)
    roughness_index = np.zeros(mask.shape+(2,),np.float)
    
    for i in range(1,mask.shape[0]-1):
        
        #seperating the current, previous and next layer
        layer_prev = mask[i-1,:,:]
        layer_next = mask[i+1,:,:]
        layer = mask[i,:,:]
        
        #Finding the border elements in each layer
        kernel = [[8,4],[2,1]]
        neighbour_code_map_layer_prev = ndimage.filters.correlate(layer_prev.astype(np.uint8), kernel, mode="constant", cval=0)
        neighbour_code_map_layer = ndimage.filters.correlate(layer.astype(np.uint8), kernel, mode="constant", cval=0)
        neighbour_code_map_layer_next = ndimage.filters.correlate(layer_next.astype(np.uint8), kernel, mode="constant", cval=0)

        border_layer_prev = ((neighbour_code_map_layer_prev != 0) & (neighbour_code_map_layer_prev != 15))
        border_layer = ((neighbour_code_map_layer != 0) & (neighbour_code_map_layer != 15))
        border_layer_next = ((neighbour_code_map_layer_next != 0) & (neighbour_code_map_layer_next != 15))
        
        #computing distance eucladian distance transform for each element in a layer
        distmap_layer_prev = np.zeros_like(border_layer_prev)
        distmap_layer_next = np.zeros_like(border_layer_next)

        if border_layer_prev.any():
            distmap_layer_prev = ndimage.morphology.distance_transform_edt(
                ~border_layer_prev, sampling=[1,1])

        if border_layer_next.any():
            distmap_layer_next = ndimage.morphology.distance_transform_edt(
                ~border_layer_next, sampling=[1,1])
            
        idx = np.nonzero(border_layer)[0]
        idy = np.nonzero(border_layer)[1]
        
        roughness_index[i,idx,idy,0] = distmap_layer_prev[idx,idy]
        roughness_index[i,idx,idy,1] = distmap_layer_next[idx,idy]
        
    roughness_index = np.arctan(roughness_index)*180/np.pi
    
    return roughness_index

#  EOF