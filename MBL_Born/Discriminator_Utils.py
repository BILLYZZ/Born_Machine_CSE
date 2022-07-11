# File  that contains differnet discriminator architectures and discriminator loss functions
import torch
import torch.nn as nn
import Utils
# Vanilla Discriminator has one single output
class Vanilla_Discriminator(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Vanilla_Discriminator, self).__init__()
        self.l1 = nn.Sequential(
                      nn.Linear(input_dim, 4),
                      nn.BatchNorm1d(num_features=4),
                      nn.LeakyReLU(0.2, inplace=True),
                      #nn.MaxPool2d(kernel_size=2, stride=2) # dim reduction!
                    )
        self.l2 = nn.Sequential(#second conv layer
                      nn.Linear(4, 4),
                      nn.BatchNorm1d(num_features=4),
                      nn.LeakyReLU(0.2, inplace=True),
                      #nn.MaxPool2d(kernel_size=2, stride=2), # dim reduction!
                    )
        self.l3 = nn.Sequential(# output is 4 by 7 by 7
                      #nn.Flatten(), #[batch-size, 4*7*7]
                      nn.Linear(4, output_dim),
                      nn.Sigmoid()
                    )
                    
    def forward(self, x):
        h1 = self.l1(x) # batch, 1, 14, 14
        h2 = self.l2(h1) 
        #s = h1 + h2
        output = self.l3(h2)
        return output

class MCR_Discriminator(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MCR_Discriminator, self).__init__()
        self.l1 = nn.Sequential(
                      nn.Linear(input_dim, 4),
                      nn.BatchNorm1d(num_features=4),
                      nn.LeakyReLU(0.2, inplace=True),
                      #nn.MaxPool2d(kernel_size=2, stride=2) # dim reduction!
                    )
        self.l2 = nn.Sequential(#second conv layer
                      nn.Linear(4, 4),
                      nn.BatchNorm1d(num_features=4),
                      nn.LeakyReLU(0.2, inplace=True),
                      #nn.MaxPool2d(kernel_size=2, stride=2), # dim reduction!
                    )
        self.l3 = nn.Sequential(# output is 4 by 7 by 7
                      #nn.Flatten(), #[batch-size, 4*7*7]
                      nn.Linear(4, output_dim),
                      nn.BatchNorm1d(num_features=output_dim),
                    )
                    
    def forward(self, x):
        h1 = self.l1(x) # batch, 1, 14, 14
        h2 = self.l2(h1) 
        #s = h1 + h2
        output = self.l3(h2)
        return output

# Firt do dimension reduction (or remapping) to z then do an RBF kernel on the
# feature maps
class MMD_Discriminator(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MMD_Discriminator, self).__init__()
        self.l1 = nn.Sequential(
                      nn.Linear(input_dim, 4),
                      nn.BatchNorm1d(num_features=4),
                      nn.LeakyReLU(0.2, inplace=True),
                      #nn.MaxPool2d(kernel_size=2, stride=2) # dim reduction!
                    )
        self.l2 = nn.Sequential(#second conv layer
                      nn.Linear(4, 4),
                      nn.BatchNorm1d(num_features=4),
                      nn.LeakyReLU(0.2, inplace=True),
                      #nn.MaxPool2d(kernel_size=2, stride=2), # dim reduction!
                    )
        self.l3 = nn.Sequential(# output is 4 by 7 by 7
                      #nn.Flatten(), #[batch-size, 4*7*7]
                      nn.Linear(4, output_dim),
                    )
                    
    def forward(self, x):
        h1 = self.l1(x) # batch, 1, 14, 14
        h2 = self.l2(h1) 
        #s = h1 + h2
        output = self.l3(h2)
        return output


#-----------------------Loss function calculations------------------------------
def mcr_feature_kernel_loss(fX_m_n, fY_m_n, d, m, eps): # d is the feature dimension, same as n
    XXT = torch.matmul( fX_m_n.T, fX_m_n )
    YYT = torch.matmul( fY_m_n.T, fY_m_n )
    
    I_XX_T_YYT = torch.eye(d) + (d/(2*m*eps*eps))*(XXT+YYT)
    I_XX_T = torch.eye(d) + (d/(m*eps*eps))*XXT
    I_YY_T = torch.eye(d) + (d/(m*eps*eps))*YYT
    
    delta_R = (1/2)*torch.logdet(I_XX_T_YYT)-(1/4)*torch.logdet(I_XX_T)-(1/4)*torch.logdet(I_YY_T)
    return -delta_R

def mcr_deep_rbf_kernel_loss(fX_m_n, fY_m_n, d, m, eps, sigma_list): 
    fXT_fX = Utils.mix_rbf_kernel(fX_m_n, fX_m_n, sigma_list) # Returns pair-wise distance in the hilbert space
    fYT_fX = Utils.mix_rbf_kernel(fY_m_n, fX_m_n, sigma_list) # Returns pair-wise distance in the hilbert space
    fXT_fY = fYT_fX.T
    fYT_fY = Utils.mix_rbf_kernel(fY_m_n, fY_m_n, sigma_list) # Returns pair-wise distance in the hilbert space
    sum_x_counts = m
    sum_y_counts = m
    sum_x_y_counts = 2*m

    MT_M = Utils.stack_blocks(fXT_fX, fXT_fY, fYT_fX, fYT_fY)
    log_det_I_plus_alpha_MT_M = torch.logdet(torch.eye(sum_x_y_counts)+(d/((sum_x_y_counts)*eps**2))*MT_M)
    
    log_det_I_plus_alpha_fXT_fX = torch.logdet(torch.eye(sum_x_counts)+(d/((sum_x_counts)*eps**2))*fXT_fX)
    log_det_I_plus_alpha_fYT_fY = torch.logdet(torch.eye(sum_y_counts)+(d/((sum_y_counts)*eps**2))*fYT_fY)

    mcr = log_det_I_plus_alpha_MT_M - (sum_x_counts/sum_x_y_counts)*log_det_I_plus_alpha_fXT_fX - \
    (sum_y_counts/sum_x_y_counts)*log_det_I_plus_alpha_fYT_fY
    return mcr/2

# The MMD loss on features z. Assuming equal numbers of x and y samples
def mmd_deep_rbf_kernel_loss(fX_m_n, fY_m_n, sigma_list):
    m = len(fX_m_n)
    K_XX = Utils.mix_rbf_kernel(fX_m_n, fX_m_n, sigma_list) # Returns pair-wise distance in the hilbert space
    K_XY = Utils.mix_rbf_kernel(fX_m_n, fY_m_n, sigma_list)
    return -( torch.sum(K_XX)/(m*m)-2*torch.sum(K_XY)/(m*m) )

def mmd_feature_kernel_loss(fX_m_n, fY_m_n):
    m = len(fX_m_n)
    K_XX = fX_m_n@(fX_m_n.T)
    K_XY = fX_m_n@(fY_m_n.T)
    return -( torch.sum(K_XX)/(m*m)-2*torch.sum(K_XY)/(m*m) )

def Vanilla_GAN_loss(fX_m_n, fY_m_n):
    return -(torch.mean(torch.log(fY_m_n))+torch.mean(torch.log(1-fX_m_n)))
