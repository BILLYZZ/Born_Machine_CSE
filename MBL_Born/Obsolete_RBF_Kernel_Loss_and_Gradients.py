import  math
import torch
import Utils
# This is the MMD calculation by kernel matrix, when the full probability space is tracked (sufficiently small)
# This replaces an NN discriminator
# NOTE: this is only for small proability space! Later I will replace this by a version that calculate
# unbiased estimate ONLY USING SAMPLES.
class RBF_MMD_MCR_by_density(object):
    def __init__(self, mcr_eps, sigma_list, x_basis_m_n, num_sample):
        self.K = None
        self.eps = mcr_eps
        self.x_basis_m_n = x_basis_m_n # torch tensor
        self.sigma_list = sigma_list
        self.num_sample = num_sample # The number of samples used to approximate the covariance matrix
        self.update_kernel()

    def update_kernel(self):
        self.K = Utils.mix_rbf_kernel(self.x_basis_m_n, self.x_basis_m_n, self.sigma_list) # explicit kernel calculation
    #----------------------Helper functions--------------------------------
    def kernel_expect(self, px, py): # Calculates px^T K py (for column vec py px) but note py px here are row vectors
        return px@self.K@(py.T)

    def witness(self, px, py):
        '''witness function of this kernel.'''
        return self.K@((px-py).T)

    def diagonal_expect(self, px_perturb): 
        '''Calculate E_{px} \phi(x)^T \phi(x)'''
        return torch.diag(self.K).dot(px_perturb)
    
    def stack_blocks(self, A, B, C, D): # stack four matrix blocks[[A,B],[C,D]]
        row0 = torch.cat((A,B), dim=1)
        row1 = torch.cat((C,D), dim=1)
        return torch.cat((row0, row1), dim=0)

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
        
        fXT_fX = torch.repeat_interleave(torch.repeat_interleave(self.K, x_counts, axis=0), x_counts, axis=1)
        fYT_fX = torch.repeat_interleave(torch.repeat_interleave(self.K, y_counts, axis=0), x_counts, axis=1)
        fXT_fY = fYT_fX.T
        fYT_fY = torch.repeat_interleave(torch.repeat_interleave(self.K, y_counts, axis=0), y_counts, axis=1)
        
        #print([fXT_fX.shape, fXT_fY.shape, fYT_fX.shape, fYT_fY.shape])
        MT_M = self.stack_blocks(fXT_fX, fXT_fY, fYT_fX, fYT_fY)
        
            
        #print('lenMT_M', len(MT_M))
        #print('shape MT_M', MT_M.shape)
        I_plus_alpha_MT_M_inv = torch.inverse(torch.eye(sum_x_y_counts)+(1/((sum_x_y_counts)*self.eps**2))*MT_M)
        
        # Not calculate the expected value with perturbed px_plus and px_minus
        # NOTE: when taking the expectation values, directly use the whole probability space
        # without doing the actual sampling (for now)--BZ
        
        fXT_full_fX = torch.repeat_interleave(self.K, x_counts, axis=1) # Full probability space in axis 0
        fXT_full_fY = torch.repeat_interleave(self.K, y_counts, axis=1) # Full probability space in axis 0
        
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
    
    #----------------------calculate loss function values------------------------
    def mmd_loss(self, px, py):
        pxy = px-py
        return self.kernel_expect(pxy, pxy) # this directly calculates the mmd loss value itself


    # Calculate the mcr loss using full pl and estimated covariance matrices from large samples:
    # SUBJECT TO CHANGE TO A MORE ARRUCATE evaluation
    def mcr_loss(self, px, py, num_sample):
        x_counts = (num_sample * px).long() # 1d vector
        y_counts = (num_sample * py).long() # 1d vector
        sum_x_counts = x_counts.sum()
        sum_y_counts = y_counts.sum()
        sum_x_y_counts = sum_x_counts + sum_y_counts
        #print('sum x, y counts', [sum_x_counts, sum_y_counts])
        #print('x_counts', x_counts)
        #print('y_counts', y_counts)
        
        fXT_fX = torch.repeat_interleave(torch.repeat_interleave(self.K, x_counts, axis=0), x_counts, axis=1)
        fYT_fX = torch.repeat_interleave(torch.repeat_interleave(self.K, y_counts, axis=0), x_counts, axis=1)
        fXT_fY = fYT_fX.T
        fYT_fY = torch.repeat_interleave(torch.repeat_interleave(self.K, y_counts, axis=0), y_counts, axis=1)
        
        
        MT_M = self.stack_blocks(fXT_fX, fXT_fY, fYT_fX, fYT_fY)
        log_det_I_plus_alpha_MT_M = torch.logdet(torch.eye(sum_x_y_counts)+(1/((sum_x_y_counts)*self.eps**2))*MT_M)
        
        log_det_I_plus_alpha_fXT_fX = torch.logdet(torch.eye(sum_x_counts)+(1/((sum_x_counts)*self.eps**2))*fXT_fX)
        log_det_I_plus_alpha_fYT_fY = torch.logdet(torch.eye(sum_y_counts)+(1/((sum_y_counts)*self.eps**2))*fYT_fY)

        mcr = log_det_I_plus_alpha_MT_M - (sum_x_counts/sum_x_y_counts)*log_det_I_plus_alpha_fXT_fX - \
        (sum_y_counts/sum_x_y_counts)*log_det_I_plus_alpha_fYT_fY

        return mcr/2

    # ----------------------calculate the gradient---------------------
    def mmd_gradient(self, theta_list, pdf_func, p_data): # pdf_func is the pdf generating function from a circuit
        prob = pdf_func(theta_list) # returns a tensor
        grad = []
        for i in range(len(theta_list)):
            # pi/2 phase
            theta_list[i] += math.pi/2.
            prob_pos = pdf_func(theta_list) # returns a tensor
            # -pi/2 phase
            theta_list[i] -= math.pi
            prob_neg = pdf_func(theta_list)
            # recover
            theta_list[i] += math.pi/2.
            
            grad_pos = self.kernel_expect(prob, prob_pos) - self.kernel_expect(prob, prob_neg)
            grad_neg = self.kernel_expect(p_data, prob_pos) - self.kernel_expect(p_data, prob_neg)
            grad.append(grad_pos - grad_neg)
        
        return torch.FloatTensor(grad)

    # Gradient for matching the MCR objective (second order matching) only
    def mcr_gradient(self, theta_list, pdf_func, p_data, num_sample): # p_data is a tensor
        prob = pdf_func(theta_list) # simulates what we observe from the current circuit
        grad = []
        for i in range(len(theta_list)):
            # pi/2 phase
            theta_list[i] += math.pi/2.
            prob_pos = pdf_func(theta_list)
            # -pi/2 phase
            theta_list[i] -= math.pi
            prob_neg = pdf_func(theta_list)
            # recover
            theta_list[i] += math.pi/2.
            
            
            grad_L_A_cross, grad_L_B_cross, x_sample_ratio = self.phiT_x_M_Dinv_MT_phi_expect(
                prob, p_data, prob_pos, prob_neg, self.num_sample)
            
            grad_i = ((1-x_sample_ratio)*self.diagonal_expect(prob_pos)-\
                      (1-x_sample_ratio)*self.diagonal_expect(prob_neg))-\
                (grad_L_A_cross) + x_sample_ratio*(grad_L_B_cross)
            grad.append(grad_i)

        return torch.FloatTensor(grad)

    # def mmd_mcr_mix_gradient(self, theta_list, pdf_func, p_data, x_basis_m_n,
    #     sigma_list, if_update_Kernel, alpha=1.0):
    #     mmd_grad = self.mmd_gradient(theta_list, pdf_func, p_data, x_basis_m_n, sigma_list, 
    #     if_update_Kernel)
    #     mcr_grad = self.mcr_gradient(theta_list, pdf_func, p_data, x_basis_m_n, sigma_list, 
    #     if_update_Kernel)
    #     return mmd_grad + alpha*mcr_grad



        


