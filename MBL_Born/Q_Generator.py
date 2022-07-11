# Quantum Born Machine generator
import torch
import numpy as np
import Utils
# Quantum generator wrapper around the differentiable circuit
# the generator has access to the (either gaussian or deep) kernel, which calculates the circuit gradients
class Q_Generator(torch.nn.Module): 
    def __init__(self, circuit, kernel, loss_func_name): # assume that one has built a quantum circuit
        super(Q_Generator, self).__init__()
       
        self.circuit = circuit
        self.kernel = kernel # For calculating the cricuit parameters' gradients
        self.loss_func_name = loss_func_name
        self.register_parameter(name='theta_list', param=torch.nn.Parameter(torch.rand(circuit.num_param)*2*np.pi))
        print('length of theta list', len(self.theta_list))
    
    def forward(self): # Generate a batch of fake samples
        return self.generate() # Already a detached tensor

    def generate(self):
        pdf_actual = self.circuit.pdf_actual(self.theta_list).detach().numpy().astype('float64') # convert to np for the np.random.choce() operation
        return torch.FloatTensor(Utils.sample_from_prob(self.kernel.x_basis_m_n, pdf_actual, self.circuit.shots)).detach()
    
    def load_gradients(self, p_data, select_theta_ind): # calculate and load gradients to self.theta_list
        if self.loss_func_name == 'MCR_deep_rbf_kernel':
            gradients = self.kernel.mcr_deep_rbf_kernel_gradient(self.theta_list, self.circuit.pdf_sample, p_data, 
                                                                    self.circuit.shots) # numpy array
        elif self.loss_func_name == 'MCR_feature_kernel':
            gradients = self.kernel.mcr_feature_kernel_gradient(self.theta_list, self.circuit.pdf_sample, p_data)
        elif self.loss_func_name == 'MMD_deep_rbf_kernel':
            gradients = self.kernel.mmd_deep_rbf_kernel_gradient(self.theta_list, self.circuit.pdf_sample, p_data, select_theta_ind)
        elif self.loss_func_name == 'MMD_feature_kernel':
            gradients = self.kernel.mmd_feature_kernel_gradient(self.theta_list, self.circuit.pdf_sample, p_data, select_theta_ind)
        elif self.loss_func_name == 'GAN':
            gradients = self.kernel.gan_gradient(self.theta_list, self.circuit.pdf_sample, p_data)
        elif self.loss_func_name == 'mix_deep_rbf_kernel': # mixture between mmd and mcr
            print('to be implemented')
        elif self.loss_func_name == 'mix_kernel_kernel':
            print('to be implemented')
    	
        self.theta_list.grad = gradients 
        


# class Circuit_grad_Function(torch.autograd.Function): # Inherits the autograd.Function class
#     # inputs theta_list and outputs a batch of fake data
#     @staticmethod
#     def forward(ctx, theta_list, circuit, kernel, p_data, batch_size, loss_func_name='Vanilla'): # 
#         ctx.theta_list_numpy = theta_list.numpy() # save parameters (torch)
#         ctx.circuit = circuit # save the raw bm circuit
#         ctx.kernel = kernel
#         ctx.loss_func_name = loss_func_name
#         #ctx.save_for_backward(Y_m_n) # save data tensor Y_m_n this way to prevent memory leak
#         # Return a batch of generated samples
#         X_m_n = circuit.generate(ctx.theta_list_numpy, batch_size)
#         #return torch.tensor(X_m_n).float(), torch.tensor(0.0) # The second output is a scalar dummy
#         return torch.tensor(X_m_n).float()

#     @staticmethod 
#     def _backward(ctx, p_data): # grad_output is the upperstream of the chain rule partial L/partial expect_z
#         """ Backward pass computation """
#         theta_list_numpy = ctx.theta_list_numpy
#         #Y_m_n = ctx.saved_tensors #retrieve the tensors in save_for_backward() expected z not used
#         if ctx.loss_func_name == 'MCR':
#             gradients = ctx.kernel.mcr_gradient(theta_list_numpy, ctx.circuit.pdf_sample, p_data) # numpy array
#         elif ctx.mcr_or_mmd == 'MMD': #'MMD'
#             gradients = ctx.kernel.gradient_mmd(theta_list_numpy, ctx.circuit.pdf_sample, ctx.p_data) # numpy array
#         else:
#             gradients = ctx.bm.gradient_vanilla(theta_list_numpy)
#         return torch.tensor(gradients).float(), None, None
#         #  and it should return as many tensors, as there were inputs to forward().

    