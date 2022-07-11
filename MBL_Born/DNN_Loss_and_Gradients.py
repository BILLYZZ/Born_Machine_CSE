# These calculation of loss function and gradients are only possible when we have explicit feature mappings
#Use this inner product kernel only when you have full tracking of the complete probability space!
# Good with small probability supports
class MMD_exact(object):
    def __init__(self, f_basis): # f_basis is the exact output from the discriminator (m, n_features)
        self.basis = f_basis
        self.K = np.matmul(f_basis, f_basis.T) # explicit kernel calculation
        
    def __call__(self, px, py):
        pxy = px-py
        return self.kernel_expect(pxy, pxy) # this directly calculates the mmd loss value itself

    def kernel_expect(self, px, py): # Calculates px^T K py
        return px.dot(self.K.dot(py))

    def witness(self, px, py):
        '''witness function of this kernel.'''
        return self.K.dot(px-py)
    
    def grad_MMD_exact(self, prob, prob_pos, prob_neg, p_data):
        grad_pos = self.kernel_expect(prob, prob_pos) - self.kernel_expect(prob, prob_neg)
        grad_neg = self.kernel_expect(p_data, prob_pos) - self.kernel_expect(p_data, prob_neg)
        return grad_pos - grad_neg
     
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
    
    