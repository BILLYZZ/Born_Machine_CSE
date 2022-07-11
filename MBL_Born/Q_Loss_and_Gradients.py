import torch
import math
import Utils
# These calculation of loss function and gradients are only possible when we have explicit feature mappings
#Use this inner product kernel only when you have full tracking of the complete probability space!
# Good with small probability supports
class Q_gradients_by_density(object): # with the use of an RBF kernel for first and/or second moment matching on the hilbert space
    def __init__(self, Discriminator, x_basis_m_n, mcr_eps=None, sigma_list=None, 
                if_deep_rbf_kernel=False, if_feature_kernel=False): # f_basis is the exact output from the discriminator (m, n_features)
        self.eps = mcr_eps
        self.Discriminator = Discriminator # Referencing the object # This might just be an identity mapping, whatever
        self.x_basis_m_n = x_basis_m_n # A tensor
        self.f_basis_m_n = None # Should be updated whenever the discriminator is updated
        

        self.if_deep_rbf_kernel = if_deep_rbf_kernel
        self.if_feature_kernel = if_feature_kernel
        # For vanilla GAN loss these can remain None.
        self.deep_rbf_K = None # kernel in the deep reproducing hilbert space (RBF on the deep features)
        self.f_K = None # kernel in the feature space, NOT in the reproducing hilbert space
        
        # for memory storage related to the second moment maching in the feature space, NOT in the hilbert space!
        self.eye_Cx_Cy, self.eye_Cx, self.eye_Cy, \
            self.eye_Cx_Cy_inv, self.eye_Cx_inv, self.eye_Cy_inv = None, None, None, None, None, None
        self.sigma_list = sigma_list
        self.update_feature_basis()
        # For the vanilla GAN loss, we don't need an rbf kernel

    def update_feature_basis(self): # call this whenever the discriminator is updated
        self.Discriminator.eval()
        self.f_basis_m_n = self.Discriminator(self.x_basis_m_n) # latent space basis
        self.Discriminator.train()
        # The kernel can now be a mixed RBF applied on the reduced-dimension z space
        #self.K = self.f_basis_m_n@(self.f_basis_m_n.T)
        if self.if_deep_rbf_kernel:
            self.deep_rbf_K = Utils.mix_rbf_kernel(self.f_basis_m_n, self.f_basis_m_n, self.sigma_list) # explicit kernel calculation
        
        if self.if_feature_kernel:
            self.f_K = self.f_basis_m_n@(self.f_basis_m_n.T)

    # Second moment matching in the feature z space
    def mcr_feature_kernel_loss(self, px, py): #f_basis is features of the whole basis, (n_features, m)
        if self.eye_Cx is None:
            self.eye_Cx_Cy, self.eye_Cx, self.eye_Cy, \
                self.eye_Cx_Cy_inv, self.eye_Cx_inv, self.eye_Cy_inv = self.eye_cov_exact(self.f_basis_m_n, px, py)
        ld_eye_Cx_Cy = torch.logdet( self.eye_Cx_Cy )
        ld_eye_Cx = torch.logdet( self.eye_Cx )
        ld_eye_Cy = torch.logdet( self.eye_Cy )
        
        delta_R = (1/2)*ld_eye_Cx_Cy - (1/4)*ld_eye_Cx - (1/4)*ld_eye_Cy
        return delta_R

    # Second moment matching in the rbf hilbert space
    def mcr_deep_rbf_kernel_loss(self, px, py, num_sample):
        x_counts = (num_sample * px).long() # 1d vector
        y_counts = (num_sample * py).long() # 1d vector
        sum_x_counts = x_counts.sum()
        sum_y_counts = y_counts.sum()
        sum_x_y_counts = sum_x_counts + sum_y_counts
        #print('sum x, y counts', [sum_x_counts, sum_y_counts])
        #print('x_counts', x_counts)
        #print('y_counts', y_counts)
        
        fXT_fX = torch.repeat_interleave(torch.repeat_interleave(self.deep_rbf_K, x_counts, axis=0), x_counts, axis=1)
        fYT_fX = torch.repeat_interleave(torch.repeat_interleave(self.deep_rbf_K, y_counts, axis=0), x_counts, axis=1)
        fXT_fY = fYT_fX.T
        fYT_fY = torch.repeat_interleave(torch.repeat_interleave(self.deep_rbf_K, y_counts, axis=0), y_counts, axis=1)
        
        
        MT_M = Utils.stack_blocks(fXT_fX, fXT_fY, fYT_fX, fYT_fY)
        log_det_I_plus_alpha_MT_M = torch.logdet(torch.eye(sum_x_y_counts)+(1/((sum_x_y_counts)*self.eps**2))*MT_M)
        
        log_det_I_plus_alpha_fXT_fX = torch.logdet(torch.eye(sum_x_counts)+(1/((sum_x_counts)*self.eps**2))*fXT_fX)
        log_det_I_plus_alpha_fYT_fY = torch.logdet(torch.eye(sum_y_counts)+(1/((sum_y_counts)*self.eps**2))*fYT_fY)

        mcr = log_det_I_plus_alpha_MT_M - (sum_x_counts/sum_x_y_counts)*log_det_I_plus_alpha_fXT_fX - \
        (sum_y_counts/sum_x_y_counts)*log_det_I_plus_alpha_fYT_fY

        return mcr/2
    
    def mmd_deep_rbf_kernel_loss(self, px, py):
        pxy = px-py
        return self.kernel_expect(self.deep_rbf_K, pxy, pxy) # this directly calculates the mmd loss value itself
    
    def mmd_feature_kernel_loss(self, px, py):
        pxy = px-py
        return self.kernel_expect(self.f_K, pxy, pxy) # this directly calculates the mmd loss value itself

#------------------Gradient calculation functions:-------------------------
    def mmd_deep_rbf_kernel_gradient(self, theta_list, pdf_func, p_data, select_theta_ind=None):
        return self.mmd_gradient(self.deep_rbf_K, theta_list, pdf_func, p_data, select_theta_ind)
    def mmd_feature_kernel_gradient(self, theta_list, pdf_func, p_data, select_theta_ind=None):
        return self.mmd_gradient(self.f_K, theta_list, pdf_func, p_data, select_theta_ind)

    # A helper function
    def mmd_gradient(self, K, theta_list, pdf_func, p_data, select_theta_ind=None):
        if select_theta_ind is None:
            select_range = range(len(theta_list))
        else:
            select_range = select_theta_ind

        self.update_feature_basis() # because the Discriminator has been updated
        prob = pdf_func(theta_list) # Returns a tensor
        grad = []
        for i in select_range:
            # pi/2 phase
            theta_list.data[i] += math.pi/2.
            prob_pos = pdf_func(theta_list)
            # -pi/2 phase
            theta_list.data[i] -= math.pi
            prob_neg = pdf_func(theta_list)
            # recover
            theta_list.data[i] += math.pi/2.

            #p_data = approx_p_data_pn(len(self.mmd_mcr.basis), prob, prob_pos, prob_neg)
            
            grad_pos = self.kernel_expect(K, prob, prob_pos) - self.kernel_expect(K, prob, prob_neg)
            grad_neg = self.kernel_expect(K, p_data, prob_pos) - self.kernel_expect(K, p_data, prob_neg)
            grad.append(grad_pos - grad_neg)
        
        #print('gradients', grad)
        return torch.FloatTensor(grad)




    # For this one we use the rbf kernel so we only have access to the inner products
    def mcr_deep_rbf_kernel_gradient(self, theta_list, pdf_func, p_data, num_sample): # p_data is a tensor
        prob = pdf_func(theta_list) # simulates what we observe from the current circuit
        grad = []
        for i in range(len(theta_list)):
            # pi/2 phase
            theta_list.data[i] += math.pi/2.
            prob_pos = pdf_func(theta_list)
            # -pi/2 phase
            theta_list.data[i] -= math.pi
            prob_neg = pdf_func(theta_list)
            # recover
            theta_list.data[i] += math.pi/2.
            
            
            grad_L_A_cross, grad_L_B_cross, x_sample_ratio = self.phiT_x_M_Dinv_MT_phi_expect(
                prob, p_data, prob_pos, prob_neg, num_sample)
            
            grad_i = ((1-x_sample_ratio)*self.diagonal_expect(prob_pos)-\
                      (1-x_sample_ratio)*self.diagonal_expect(prob_neg))-\
                (grad_L_A_cross) + x_sample_ratio*(grad_L_B_cross)
            grad.append(grad_i)

        return torch.FloatTensor(grad)

    # For this one, we have the explicit form of the feature vectors in f_basis_m_n, so we directly calculate the cov matrix
    def mcr_feature_kernel_gradient(self, theta_list, pdf_func, p_data): 
        self.update_feature_basis() # because the Discriminator has been updated
        prob = pdf_func(theta_list)
        self.eye_Cx_Cy, self.eye_Cx, self.eye_Cy, \
            self.eye_Cx_Cy_inv, self.eye_Cx_inv, self.eye_Cy_inv = self.eye_cov_exact(self.f_basis_m_n, prob, p_data)
        grad = []
        for i in range(len(theta_list)):
            # pi/2 phase
            theta_list.data[i] += math.pi/2.
            prob_pos = pdf_func(theta_list)
            # -pi/2 phase
            theta_list.data[i] -= math.pi
            prob_neg = pdf_func(theta_list)
            # recover
            theta_list.data[i] += math.pi/2.

            grad_LA_plus = self.mcr_expect_inverse_dot_prod_exact(self.eye_Cx_Cy_inv, prob_pos)
            grad_LA_minus = self.mcr_expect_inverse_dot_prod_exact(self.eye_Cx_Cy_inv, prob_neg)
        
            grad_LB_plus = self.mcr_expect_inverse_dot_prod_exact(self.eye_Cx_inv, prob_pos)
            grad_LB_minus = self.mcr_expect_inverse_dot_prod_exact(self.eye_Cx_inv, prob_neg)
            grad_i = (1/4)*(grad_LA_plus-grad_LA_minus)-(1/8)*(grad_LB_plus-grad_LB_minus)
            grad.append(grad_i)
        return torch.FloatTensor(grad)

# GAN gradient should be put in another separate class as it utilizes the minimal amount of resources.

    def gan_gradient(self, theta_list, pdf_func, p_data): 
        # In the GAN case, the f_x_basis_m_n is a two dimensional vector with shape (m,1)
        self.update_feature_basis() # because the Discriminator has been updated
        #prob = pdf_func(theta_list)
        grad = []
        for i in range(len(theta_list)):
            # pi/2 phase
            theta_list.data[i] += math.pi/2.
            prob_pos = pdf_func(theta_list)
            # -pi/2 phase
            theta_list.data[i] -= math.pi
            prob_neg = pdf_func(theta_list)
            # recover
            theta_list.data[i] += math.pi/2.
            
            pos_ln_D = torch.dot(prob_pos, torch.log(self.f_basis_m_n).T[0]) #f_basis_m_n has shape (m, 1)
            neg_ln_D = torch.dot(prob_neg, torch.log(self.f_basis_m_n).T[0])
            
            grad_i = -(pos_ln_D-neg_ln_D)
            grad.append(grad_i)
        return torch.FloatTensor(grad)
    #def mmd_mcr_mix_gradient(s)






# Support functions------------------------------------------------------------------------------
 #About mmd--------------
    def kernel_expect(self, K, px, py): # Calculates px^T K py. K is either deep rbf kernel or feature kernel
        return px@K@(py.T)

    def witness(self, px, py):
        '''witness function of this kernel.'''
        return self.K@((px-py).T)

 #About mcr--------------
    def diagonal_expect(self, px_perturb): 
        '''Calculate E_{px} \phi(x)^T \phi(x)'''
        return torch.diag(self.deep_rbf_K).dot(px_perturb)

    def eye_cov_exact(self, f_basis_m_n, px, py):
        m, n_features = f_basis_m_n.shape
        
        Cx = f_basis_m_n.T@torch.diag(px)@(f_basis_m_n)
        Cy = f_basis_m_n.T@torch.diag(py)@(f_basis_m_n)
        
        eye_Cx_Cy = torch.eye(n_features) + (n_features/(2*self.eps**2)) * (Cx+Cy) 
        eye_Cx = torch.eye(n_features) + (n_features/(self.eps**2)) * Cx 
        eye_Cy = torch.eye(n_features) + (n_features/(self.eps**2)) * Cy 
        
        eye_Cx_Cy_inv = torch.inverse(eye_Cx_Cy)
        eye_Cx_inv = torch.inverse(eye_Cx)
        eye_Cy_inv = torch.inverse(eye_Cy)
        
        return eye_Cx_Cy, eye_Cx, eye_Cy, eye_Cx_Cy_inv, eye_Cx_inv, eye_Cy_inv
        

    def mcr_expect_inverse_dot_prod_exact(self, eye_Cov_inv, px_pm): # px is either the exact wavefunction or sampled
        m = self.f_basis_m_n.shape[0] # number of patterns in the basis 
        
        #eye_Cov_inv = np.linalg.inv(eye_Cov)
        
        eye_Cov_repeat = eye_Cov_inv[None,:].repeat(m, axis=0) #(m, n, n)
        
        f_basis_m_n = self.f_basis_m_n # (m,n)
        f_basis_m_n_1 = f_basis_m_n[:,:,None]
        f_basis_m_1_n = f_basis_m_n[:,None,:]
        
        fx_fxT_m_n_n = f_basis_m_n_1@f_basis_m_1_n #(m, n, n)
        
        temp = torch.sum(fx_fxT_m_n_n*eye_Cov_repeat, axis=(1,2)) #(m)
                      
        assert temp.shape == (m,)
        return torch.sum(temp * px_pm)

    def phiT_x_M_Dinv_MT_phi_expect(self, px, py, px_plus, px_minus, num_sample): # To do: pass in the num_sample value!
        '''Calculate E_{px} phi(x)^T M (I+aMT M)^{-1} M^T phi(x) both in term A and in term B --BZ
            'num_sample' is the number of samples used to estimate the covariance matrix
        '''
        # Sample from px and py, with no perturbation
        x_counts = (num_sample * px).long() # 1d tensor repeat has to be long tensor
        y_counts = (num_sample * py).long() # 1d tesor
        sum_x_counts = x_counts.sum()
        sum_y_counts = y_counts.sum()
        sum_x_y_counts = sum_x_counts + sum_y_counts
        #print('sum x, y counts', [sum_x_counts, sum_y_counts])
        
        fXT_fX = torch.repeat_interleave(torch.repeat_interleave(self.deep_rbf_K, x_counts, axis=0), x_counts, axis=1)
        fYT_fX = torch.repeat_interleave(torch.repeat_interleave(self.deep_rbf_K, y_counts, axis=0), x_counts, axis=1)
        fXT_fY = fYT_fX.T
        fYT_fY = torch.repeat_interleave(torch.repeat_interleave(self.deep_rbf_K, y_counts, axis=0), y_counts, axis=1)
        
        #print([fXT_fX.shape, fXT_fY.shape, fYT_fX.shape, fYT_fY.shape])
        MT_M = Utils.stack_blocks(fXT_fX, fXT_fY, fYT_fX, fYT_fY)
        
            
        #print('lenMT_M', len(MT_M))
        #print('shape MT_M', MT_M.shape)
        I_plus_alpha_MT_M_inv = torch.inverse(torch.eye(sum_x_y_counts)+(1/((sum_x_y_counts)*self.eps**2))*MT_M)
        
        # Not calculate the expected value with perturbed px_plus and px_minus
        # NOTE: when taking the expectation values, directly use the whole probability space
        # without doing the actual sampling (for now)--BZ
        
        fXT_full_fX = torch.repeat_interleave(self.deep_rbf_K, x_counts, axis=1) # Full probability space in axis 0
        fXT_full_fY = torch.repeat_interleave(self.deep_rbf_K, y_counts, axis=1) # Full probability space in axis 0
        
        fXT_full_M = torch.cat([fXT_full_fX, fXT_full_fY], axis=1)
        temp = fXT_full_M@I_plus_alpha_MT_M_inv
        temp_sum = (temp * fXT_full_M).sum(axis=1) # 1d array with len(pl)
        A_pos = temp_sum.dot(px_plus)
        A_neg = temp_sum.dot(px_minus)
        grad_L_A_cross = (1/((sum_x_y_counts)*self.eps**2))*(A_pos-A_neg)
        
        # Now calcualte the similar structure for L_B:
        I_plus_alpha_fXT_fX_inv = torch.inverse(torch.eye(sum_x_counts)+(1/((sum_x_counts)*self.eps**2))*fXT_fX)
        temp = fXT_full_fX@I_plus_alpha_fXT_fX_inv
        temp_sum = (temp * fXT_full_fX).sum(axis=1) # 1d array with len(pl)
        B_pos = temp_sum.dot(px_plus) # scalar
        B_neg = temp_sum.dot(px_minus)
        grad_L_B_cross = (1/((sum_x_counts)*self.eps**2))*(B_pos-B_neg)
        
        return grad_L_A_cross, (sum_x_counts/sum_x_y_counts)*grad_L_B_cross, sum_x_counts/sum_x_y_counts
    



# Ignore the below for now:
#_________________________________    
     
class MMD_sample(object):
    def __call__(self, fX, fY): # fX and fY are both n by m
        fX_T_fX = (fX.T)@fX
        fX_T_fY = (fX.T)@fY
        fY_T_fY = (fY.T)@fY
        return (np.sum(fX_T_fX)-2*np.sum(fX_T_fY)+np.sum(fY_T_fY))/(m*m)
    
    def grad_MMD_sample(self, fX, fY, fX_plus, fX_minus): # fX and fY are both n by m
        fX_plus_T_fX = (fX_plus.T)@fX
        fX_minus_T_fX = (fX_minus.T)@fX
        fX_plus_T_fY = (fX_plus.T)@fY
        fX_minus_T_fY = (fX_minus.T)@fY
        return (fX_plus_T_fX-fX_minus_T_fX-fX_plus_T_fY+fX_minus_T_fY)/(m*m)
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# A class that handles MCR calculation, either with sample approximates or by exact probabilities
class MCR_exact(object):
    def __init__(self, eps, f_basis_m_n, px, py): # These are just numpy float arrays
        self.eps = eps
        self.eye_Cx_Cy, self.eye_Cx, self.eye_Cy, \
            self.eye_Cx_Cy_inv, self.eye_Cx_inv, self.eye_Cy_inv = self.eye_cov_exact(f_basis_m_n, px, py) 
        self.f_basis_m_n = f_basis_m_n
        
    def eye_cov_exact(self, f_basis_m_n, px, py):
        
        m, n_features = f_basis_m_n.shape
        
        Cx = f_basis_m_n.T@np.diag(px)@(f_basis_m_n)
        Cy = f_basis_m_n.T@np.diag(py)@(f_basis_m_n)
        
        eye_Cx_Cy = np.eye(n_features) + (n_features/(2*self.eps**2)) * (Cx+Cy) 
        eye_Cx = np.eye(n_features) + (n_features/(self.eps**2)) * Cx 
        eye_Cy = np.eye(n_features) + (n_features/(self.eps**2)) * Cy 
        
        eye_Cx_Cy_inv = np.linalg.inv(eye_Cx_Cy)
        eye_Cx_inv = np.linalg.inv(eye_Cx)
        eye_Cy_inv = np.linalg.inv(eye_Cy)
        
        return eye_Cx_Cy, eye_Cx, eye_Cy, eye_Cx_Cy_inv, eye_Cx_inv, eye_Cy_inv
        
    def mcr_loss_exact(self): #f_basis is features of the whole basis, (n_features, m)
        ld_eye_Cx_Cy = np.log(np.linalg.det( self.eye_Cx_Cy ))
        ld_eye_Cx = np.log(np.linalg.det( self.eye_Cx ))
        ld_eye_Cy = np.log(np.linalg.det( self.eye_Cy ))
        
        delta_R = (1/2)*ld_eye_Cx_Cy - (1/4)*ld_eye_Cx - (1/4)*ld_eye_Cy
        return delta_R

    def mcr_expect_inverse_dot_prod_exact(self, eye_Cov_inv, px_pm): # px is either the exact wavefunction or sampled
        m = self.f_basis_m_n.shape[0] # number of patterns in the basis 
        
        #eye_Cov_inv = np.linalg.inv(eye_Cov)
        
        eye_Cov_repeat = eye_Cov_inv[None,:].repeat(m, axis=0) #(m, n, n)
        
        f_basis_m_n = self.f_basis_m_n # (m,n)
        f_basis_m_n_1 = f_basis_m_n[:,:,None]
        f_basis_m_1_n = f_basis_m_n[:,None,:]
        
        fx_fxT_m_n_n = f_basis_m_n_1@f_basis_m_1_n #(m, n, n)
        
        temp = np.sum(fx_fxT_m_n_n*eye_Cov_repeat, axis=(1,2)) #(m)
                      
        assert temp.shape == (m,)
        return np.sum(temp * px_pm)
        
    def mcr_grad_exact(self, px_plus, px_minus): 
        
        grad_LA_plus = self.mcr_expect_inverse_dot_prod_exact(self.eye_Cx_Cy_inv, px_plus)
        grad_LA_minus = self.mcr_expect_inverse_dot_prod_exact(self.eye_Cx_Cy_inv, px_minus)
        
        grad_LB_plus = self.mcr_expect_inverse_dot_prod_exact(self.eye_Cx_inv, px_plus)
        grad_LB_minus = self.mcr_expect_inverse_dot_prod_exact(self.eye_Cx_inv, px_minus)
        
        grad = (1/4)*(grad_LA_plus-grad_LA_minus)-(1/8)*(grad_LB_plus-grad_LB_minus)
        return grad
    
    