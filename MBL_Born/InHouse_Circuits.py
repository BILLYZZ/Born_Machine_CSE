import numpy as np
import torch
import q_circuit as q
import torch
import Utils
# In house circuit based on numpy
class MLP_Circuit(object):
    def __init__(self, n_qubits, n_layers, connections, shots):
        self.circuit = q.get_diff_circuit(n_qubits, n_layers, connections)
        self.shots = shots
        self.n_qubits = n_qubits
        self.n_patterns = 2**self.n_qubits
        self.num_param = self.circuit.num_param
    @property
    def depth(self):
        return (len(self.circuit)-1)//2

    def pdf_sample(self, theta_list): # run the circuit with given theta_list and get actual prob distribution
        '''get probability distribution function'''
        # Assume receiving a tensor theta_list
        theta_list = theta_list.detach().numpy()
        wf = q.initial_wf(self.circuit.num_bit)
        self.circuit(wf, theta_list)
        pl = np.abs(wf)**2 # probability list of the simulator
        # introducing sampling error to simulate the actual scenario!
        if self.shots is not None:
            pl = Utils.prob_from_sample(Utils.sample_from_prob(np.arange(len(pl)), pl, self.shots),
                    len(pl), False)
        return torch.FloatTensor(pl)

    def pdf_actual(self, theta_list):
        theta_list = theta_list.detach().numpy()
        wf = q.initial_wf(self.circuit.num_bit)
        self.circuit(wf, theta_list)
        pl = np.abs(wf)**2 # probability list of the simulator
        return torch.FloatTensor(pl)
    
    

    # def mmd_loss(self, theta_list):
    #     '''get the loss'''
    #     # get and cahe probability distritbution of Born Machine
    #     self._prob = self.pdf_actual(theta_list)
    #     # use wave function to get mmd loss
    #     return self.Kernel.mmd_loss(self._prob, self.p_data)

    # def mcr_loss(self, theta_list):
    #     prob = self.pdf_actual(theta_list)
    #     return self.Kernel.mcr_loss(px=prob, py=self.p_data, num_sample=500)
    
    # def mmd_gradient(self, theta_list):
    #     '''
    #     This is for MMD only!
    #     '''
    #      if self.y_batch_size is not None:
    #          num_bit = self.circuit.num_bit
    #          p_data = prob_from_sample(sample_from_prob(np.arange(num_bit**2), self.p_data, self.y_batch_size),
    #                      num_bit**2, False)
    #      else:
    #          p_data = self.p_data
    #     return self.Kernel.mmd_gradient(theta_list, self.pdf_sample, p_data,)
    
    # # Gradient for matching the MCR objective (second order matching) only
    # def mcr_gradient(self, theta_list):
    #     '''
    #     Gradient for matching the MCR objective (second order matching) only
    #     '''
    #      if self.y_batch_size is not None:
    #          num_bit = self.circuit.num_bit
    #          p_data = prob_from_sample(sample_from_prob(np.arange(num_bit**2), self.p_data, self.y_batch_size),
    #                      num_bit**2, False)
    #      else:
    #          p_data = self.p_data
        
    #     return self.Kernel.mcr_gradient(theta_list, self.pdf_sample, p_data)

    # def gradient_MMD_MCR(self, theta_list, alpha):
    #     mmd_grad = self.gradient(theta_list)
    #     mcr_grad = self.gradient_MCR_only(theta_list)
    #     return mmd_grad + alpha*mcr_grad
        
    # # IGNORE THIS FUNCTION FOR NOW::::::::::
    # def gradient_numerical(self, theta_list, delta=1e-2):
    #     '''
    #     numerical differenciation.
    #     '''
    #     grad = []
    #     for i in range(len(theta_list)):
    #         theta_list[i] += delta/2.
    #         loss_pos = self.mmd_loss(theta_list)
    #         theta_list[i] -= delta
    #         loss_neg = self.mmd_loss(theta_list)
    #         theta_list[i] += delta/2.

    #         grad_i = (loss_pos - loss_neg)/delta
    #         grad.append(grad_i)
    #     return np.array(grad)


#-----------------Then we can do other circuit architectures as well!!! yeah!!!!--------------------------------
#class XXZ_Circuit(object):
