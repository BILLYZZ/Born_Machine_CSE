import numpy as np
import torch

def digit_basis(geometry): # complete probability space
    num_bit = np.prod(geometry)
    M = 2**num_bit
    x = np.arange(M)
    return x

def binary_basis(geometry): # complete probability space
    num_bit = np.prod(geometry)
    M = 2**num_bit
    x = np.arange(M)
    return unpacknbits(x[:,None], num_bit).reshape((-1,)+geometry)

def unpacknbits(arr, nbit, axis=-1):
    '''unpack numbers to bitstrings.'''
    nd = np.ndim(arr)
    if axis < 0:
        axis = nd + axis
    return (((arr & (1 << np.arange(nbit - 1, -1, -1)).reshape([-1] + [1] * (nd - axis - 1)))) > 0).astype('int8')


def packnbits(arr, axis=-1):
    '''pack bitstrings to numbers.'''
    nd = np.ndim(arr)
    nbit = np.shape(arr)[axis]
    if axis < 0:
        axis = nd + axis
    return (arr * (1 << np.arange(nbit - 1, -1, -1)).reshape([-1] + [1] * (nd - axis - 1))\
           ).sum(axis=axis, keepdims=True).astype('int')

#--------------Calculate the total variance of two probability distributions------------------
def TV(px, py):
    return torch.sum(torch.abs(px-py))

#------------------Gaussian---------------------------------
# Create the gaussian pdf as data probability distribution
def gaussian_pdf(geometry, mu, sigma):
    '''get gaussian distribution function'''
    x = digit_basis(geometry)
    # Probabilities according to gaussian formula:
    pl = 1. / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2. * sigma**2))
    return pl/pl.sum()

# Return the gaussian density value
def gaussian_density(x, mu, sigma):
    pl = 1. / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2. * sigma**2))
    return pl/pl.sum()

#--------------------BAS------------------------------------
# Create the bars and stripe pdf as data probability distribution
def barstripe_pdf(geometry):
    '''get bar and stripes PDF'''
    x = binary_basis(geometry) # All 3 by 3 image patterns
    pl = is_bs(x)
    return pl/pl.sum()

def barstripe_counts(geometry):
    '''get bar and stripes PDF'''
    x = binary_basis(geometry) # All 3 by 3 image patterns
    pl = is_bs(x)
    return pl

def is_bs(samples):
    '''a sample is a bar or a stripe.'''
    return (np.abs(np.diff(samples,axis=-1)).sum(axis=(1,2))==0\
           )|((np.abs(np.diff(samples, axis=1)).sum(axis=(1,2)))==0)
#-------------------Sample from probabilities---------------
def sample_from_prob(x, pl, num_sample): # x may be a torch tensor
    '''
    sample x from probability. Assume pl is a numpy array
    '''
    pl = pl / float(pl.sum())
    indices = np.arange(len(x))
    res = np.random.choice(indices, num_sample, p=pl)
    #print('sum', pl.sum())
    return x[res]


def prob_from_sample(dataset, hndim, packbits):
    '''
    emperical probability from data.
    '''
    if packbits:
        dataset = packnbits(dataset).ravel()
    p_data = np.bincount(dataset, minlength=hndim) # doesn't matter dataset's order, bincount always return the histogram 
    # of numbers in increasing order
    p_data = p_data / float(p_data.sum())
    return p_data

#-------------------Shuffle samples---------------------------
def concat_shuffle(matrix_list, label_list): # assuming matrix_list is a list of matrices of the shape (batch, n_features)
    matrix = torch.cat(matrix_list, axis=0)
    labels = torch.cat(label_list)
    rand_inds = torch.randperm(len(labels))
    return matrix[rand_inds], labels[rand_inds]

#--------------------Kernel related---------------------------
# Other helper functions:
# Calculates the full kernel matrix (tracking full probability space)
def mix_rbf_kernel(x, y, sigma_list): # Kernel matrix
    ndim = x.ndim
    if ndim == 1:
        exponent = (x[:, None] - y[None, :])**2 # Get pair-wise differences organized in a matrix
        print('exponent', exponent)
    elif ndim == 2:
        exponent = ((x[:, None, :] - y[None, :, :])**2).sum(axis=2) #(512, 1, 6) (1, 512, 6)-->(512, 512, 6)
    else:
        raise
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma)
        K = K + torch.exp(-gamma * exponent) # kernel matrix
    return K

def x_space_kernel(x, y):
    ndim = x.ndim
    if ndim == 1:
        result = x[:, None] * y[None, :]
        print('x space kernel', result)
        return result
    elif ndim == 2:
        result = (x[:, None, :] * y[None, :, :]).sum(axis=2)
        print('x space kernel', result)
        return result

def stack_blocks(A, B, C, D): # stack four matrix blocks[[A,B],[C,D]]
        row0 = torch.cat((A,B), dim=1)
        row1 = torch.cat((C,D), dim=1)
        return torch.cat((row0, row1), dim=0)