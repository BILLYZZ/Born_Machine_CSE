# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:47:41 2021

@author: Weishun
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

class RBFMMD2(object):
    '''
    MMD^2 with RBF (Gaussian) kernel.
    
    Args:
        sigma_list (list): a list of bandwidths.
        basis (1darray): defininng space.
      
    Attributes:
        K (2darray): full kernel matrix, notice the Hilbert is countable.
    '''
    def __init__(self, sigma_list, basis):
        self.sigma_list = sigma_list
        self.basis = basis
        self.K = mix_rbf_kernel(basis, basis, self.sigma_list)

    def __call__(self, px, py):
        '''
        Args:
            px (1darray, default=None): probability for data set x, used only when self.is_exact==True.
            py (1darray, default=None): same as px, but for data set y.

        Returns:
            float: loss.
        '''
        pxy = px-py
        return self.kernel_expect(pxy, pxy)

    def kernel_expect(self, px, py):
        '''
        expectation value of kernel function.
        
        Args:
            px (1darray): the first PDF.
            py (1darray): the second PDF.
            
        Returns:
            float: kernel expectation.
        '''
        return px.dot(self.K).dot(py)

def mix_rbf_kernel(x, y, sigma_list):
    '''
    multi-RBF kernel.
    
    Args:
        x (1darray|2darray): the collection of samples A.
        x (1darray|2darray): the collection of samples B.
        sigma_list (list): a list of bandwidths.
        
    Returns:
        2darray: kernel matrix.
    '''
    ndim = x.ndim
    if ndim == 1:
        exponent = np.abs(x[:, None] - y[None, :])**2
    elif ndim == 2:
        exponent = ((x[:, None, :] - y[None, :, :])**2).sum(axis=2)
    else:
        raise
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma)
        K = K + np.exp(-gamma * exponent)
    return K