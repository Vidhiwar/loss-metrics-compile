#!/usr/bin/env python
# coding: utf-8


import torch
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def create_kernel(dim = 2):
    if dim == 2:
        Ky = [[1.0,2.0,1.0],
              [0.0,0.0,0.0],
              [-1.0,-2.0,-1.0]]
        Kx = [[-1.0,0.0,1.0],
              [-2.0,0.0,2.0],
              [-1.0,0.0,1.0]]
        bi_K = [[8, 4], [2,1]]

        K_x_ = [[1.0, 1.0, 1.0],
              [1.0,-8.0,1.0],
              [1.0,1.0,1.0]]
        K_y_ = [[1.0,1.0,1.0],
              [1.0,-8.0,1.0],
              [1.0,1.0,1.0]]
        Kx_ = np.array(K_x)/8.0
        Ky_ = np.array(K_y)/8.0
        return Kx, Ky, bi_K

    elif dim == 3:
        Kx = [[[-1.0,-2.0,-1.0],
               [0.0,0.0,0.0],
               [1.0,2.0,1.0]],
              [[-2.0,-4.0,-2.0],
               [0.0,0.0,0.0],
               [2.0,4.0,2.0]],
              [[-1.0,-2.0,-1.0],
               [0.0,0.0,0.0],
               [1.0,2.0,1.0]]]
        
        Ky = [[[-1.0,-2.0,-1.0],
               [-2.0,-4.0,-2.0],
               [-1.0,-2.0,-1.0]],
              [[0.0,0.0,0.0],
               [0.0,0.0,0.0],
               [0.0,0.0,0.0]],
              [[1.0,2.0,1.0],
               [2.0,4.0,2.0],
               [1.0,2.0,1.0]]]
        
        Kz = [[[-1.0,0.0,1.0],
               [-2.0,0.0,2.0],
               [-1.0,0.0,1.0]],
              [[-2.0,0.0,2.0],
               [-4.0,0.0,4.0],
               [-2.0,0.0,2.0]],
              [[-1.0,0.0,1.0],
               [-2.0,0.0,2.0],
               [-1.0,0.0,1.0]]]
        return Kx,Ky,Kz


def dilation(tensor , d=2 , dim = 2):
    if dim == 2:
        k_shape = 2*d-1
        d_K = torch.ones(k_shape,k_shape).type(torch.float)
        d_K = d_K.repeat(1,1,1,1)
        dil_tensor= torch.nn.functional.conv2d(input=tensor.float(),
                                               weight=d_K, stride=(1,1), padding=(1,1))
        return dil_tensor
    if dim == 3:
        k_shape = 2*d-1
        d_K = torch.ones(k_shape,k_shape,k_shape).type(torch.float)
        d_K = d_K.repeat(1,1,1,1,1)
        dil_tensor= torch.nn.functional.conv3d(input=tensor.float(),
                                               weight=d_K, stride=(1,1,1), padding=(1,1,1))
        return dil_tensor
        

def compute_boundary_loss2D(mask_gt,mask_pred):
    
    t1 =time.time()
    Kx,Ky,bi_K = create_kernel(2)
    
    Kx_weight = torch.tensor([Kx])
    Kx_weight = Kx_weight.repeat(1,1,1,1)
    grad_gt_x = torch.nn.functional.conv2d(input=mask_gt.unsqueeze(0).unsqueeze(0).float(),
                                           weight=Kx_weight.float(), stride=(1,1), padding=(1,1))
    grad_pred_x = torch.nn.functional.conv2d(input=mask_pred.unsqueeze(0).unsqueeze(0).float(),
                                             weight=Kx_weight.float(), stride=(1,1), padding=(1,1))
    
    Ky_weight = torch.tensor([Ky])
    Ky_weight = Ky_weight.repeat(1,1,1,1)
    grad_gt_y = torch.nn.functional.conv2d(input=mask_gt.unsqueeze(0).unsqueeze(0).float(),
                                           weight=Ky_weight.float(), stride=(1,1), padding=(1,1))
    grad_pred_y = torch.nn.functional.conv2d(input=mask_pred.unsqueeze(0).unsqueeze(0).float(),
                                             weight=Ky_weight.float(), stride=(1,1), padding=(1,1))
    
    grad_gt = torch.sqrt(grad_gt_x.pow(2) + grad_gt_y.pow(2))
    grad_pred = torch.sqrt(grad_pred_x.pow(2) + grad_pred_y.pow(2))
    
    grad_gt  = dilation(grad_gt,4)
    grad_pred  = dilation(grad_pred,4)
    
    borders_gt = grad_gt[0,0,:,:].bool()
    borders_pred = grad_pred[0,0,:,:].bool()
    
    if borders_gt.any() & borders_pred.any():
        idx_nonzero_gt = torch.nonzero(borders_gt)
        idx_nonzero_pred = torch.nonzero(borders_pred)

        distances_gt_to_pred = torch.zeros(idx_nonzero_gt.shape[0]).type(torch.float)
        distances_pred_to_gt = torch.zeros(idx_nonzero_pred.shape[0]).type(torch.float)
        
        dist = torch.cdist(idx_nonzero_gt.float(),idx_nonzero_pred.float())
        distances_gt_to_pred = torch.min(dist ,dim = 1)
        distances_pred_to_gt = torch.min(dist ,dim = 0)

    elif borders_gt.any() or borders_pred.any():
        distances_gt_to_pred = torch.tensor([2**20]).type(torch.float)
        distances_pred_to_gt = torch.tensor([2**20]).type(torch.float)

    else:
        distances_gt_to_pred = torch.tensor([0]).type(torch.float)
        distances_pred_to_gt = torch.tensor([0]).type(torch.float)           

    avg_dist_gt_to_pred = torch.sum(distances_gt_to_pred.values)/idx_nonzero_gt.shape[0]
    avg_dist_pred_to_gt = torch.sum(distances_pred_to_gt.values)/idx_nonzero_pred.shape[0]
    
    print(time.time()-t1)
        
    return (avg_dist_gt_to_pred+avg_dist_pred_to_gt)/2


def compute_boundary_loss3D(mask_gt,mask_pred):
    
    t1 =time.time()

    Kx,Ky,Kz = create_kernel(3)
    
    Kx_weight = torch.tensor([Kx])
    Kx_weight = Kx_weight.repeat(1,1,1,1,1)
    grad_gt_x = torch.nn.functional.conv3d(input=mask_gt.unsqueeze(0).unsqueeze(0).float(),weight=Kx_weight, stride=(1,1,1), padding=(1,1,1))
    grad_pred_x = torch.nn.functional.conv3d(input=mask_pred.unsqueeze(0).unsqueeze(0).float(), weight=Kx_weight, stride=(1,1,1), padding=(1,1,1))

    Ky_weight = torch.tensor([Ky])
    Ky_weight = Ky_weight.repeat(1,1,1,1,1)
    grad_gt_y = torch.nn.functional.conv3d(input=mask_gt.unsqueeze(0).unsqueeze(0).float(), weight=Ky_weight, stride=(1,1,1), padding=(1,1,1))
    grad_pred_y = torch.nn.functional.conv3d(input=mask_pred.unsqueeze(0).unsqueeze(0).float(), weight=Ky_weight, stride=(1,1,1), padding=(1,1,1))
 
    Kz_weight = torch.tensor([Kz])
    Kz_weight = Kz_weight.repeat(1,1,1,1,1)
    grad_gt_z = torch.nn.functional.conv3d(input=mask_gt.unsqueeze(0).unsqueeze(0).float(),weight=Kz_weight, stride=(1,1,1), padding=(1,1,1))
    grad_pred_z = torch.nn.functional.conv3d(input=mask_pred.unsqueeze(0).unsqueeze(0).float(), weight=Kz_weight, stride=(1,1,1), padding=(1,1,1))
    
    grad_gt = torch.sqrt(grad_gt_x.pow(2) + grad_gt_y.pow(2) + grad_gt_z.pow(2))
    grad_pred = torch.sqrt(grad_pred_x.pow(2) + grad_pred_y.pow(2) + grad_pred_z.pow(2))
    
    grad_gt  = dilation(grad_gt,3,3)
    grad_pred  = dilation(grad_pred,3,3)
    
    borders_gt = grad_gt[0,0,:,:,:].bool()
    borders_pred = grad_pred[0,0,:,:,:].bool()
    
    if borders_gt.any() & borders_pred.any():
        idx_nonzero_gt = torch.nonzero(borders_gt)
        idx_nonzero_pred = torch.nonzero(borders_pred)
        
        distances_gt_to_pred = torch.zeros(idx_nonzero_gt.shape[0]).type(torch.float)
        distances_pred_to_gt = torch.zeros(idx_nonzero_pred.shape[0]).type(torch.float)
        
        dist = torch.cdist(idx_nonzero_gt.float().cuda(),idx_nonzero_pred.float().cuda())
        
        distances_gt_to_pred = torch.min(dist ,dim = 1)
        distances_pred_to_gt = torch.min(dist ,dim = 0)
        
    elif borders_gt.any() or borders_pred.any():
        distances_gt_to_pred = torch.tensor([2**20]).type(torch.float)
        distances_pred_to_gt = torch.tensor([2**20]).type(torch.float)

    else:
        distances_gt_to_pred = torch.tensor([0]).type(torch.float)
        distances_pred_to_gt = torch.tensor([0]).type(torch.float)    

    avg_dist_gt_to_pred = torch.sum(distances_gt_to_pred.values)/idx_nonzero_gt.shape[0]
    avg_dist_pred_to_gt = torch.sum(distances_pred_to_gt.values)/idx_nonzero_pred.shape[0]
    
    print(time.time()-t1)
        
    return (avg_dist_gt_to_pred+avg_dist_pred_to_gt)/2


def bitmp(img):    
    ary = np.array(img)

    r,g,b = np.split(ary,3,axis=2)
    r=r.reshape(-1)
    g=r.reshape(-1)
    b=r.reshape(-1)

    bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2], 
    zip(r,g,b)))
    bitmap = np.array(bitmap).reshape([ary.shape[0], ary.shape[1]])
    bitmap = np.dot((bitmap > 128).astype(float),255)
    img_b = Image.fromarray(bitmap.astype(np.uint8))
    return img_b

#  EOF
