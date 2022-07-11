from quspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.misc import KL_div
from scipy import integrate,optimize,special,stats,interpolate
import numpy as np # generic math functions
from time import time # timing package
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 25})
#import tensorflow as tf
import cv2
from MMD_loss import RBFMMD2

descp = 'hidden_4+2' #brief description of the simulation for the ploot titles

#define simulation parameters
n_real=500 # number of disorder realisations to search over in each quench
n_M = int(100) #number of quenches/training steps
n_bins=20 #number of bins to use in level statistics
dataset = 'normal' #'MBL'
digit = 0
seed = 42 #24

#define model parameters
L_visible = 4
L_hidden = 2
L = L_visible + L_hidden
basis = spin_basis_1d(L,pauli=False)
size = int(np.sqrt(2**(L_visible)))
N_visible = 2**L_visible

ti=0 #evolving initial time of each quench
tm = 1.0 #evolving final time of each quench
Jxy=1.0 # xy interaction
Jzz_0=1.0 # zz interaction at time t=0
h_d=20 #20.0 #0.1 #3.9 # disorder strength

bins_fixed = np.linspace(0,1,n_bins+1)

#=================set up Heisenberg Hamiltonian========================#

# define operators with PBC using site-coupling lists
J_zz = [[Jzz_0,i,(i+1)%L] for i in range(L)] # PBC
J_xy = [[Jxy/2.0,i,(i+1)%L] for i in range(L)] # PBC

# static and dynamic lists
static = [["+-",J_xy],["-+",J_xy],["zz",J_zz]]
dynamic = []

# compute the time-independent Heisenberg Hamiltonian
H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.float64)

#===============================functions for simulations=============================#

#function generate a MBL Hamiltonian given a disorder realization
def realization(H_XXZ,basis,disorder_m_i,h_disorder):
    #np.random.seed() # the random number needs to be seeded for each parallel process

	# compute disordered Hamiltonian
    unscaled_fields=-1+2*disorder_m_i
    h_z=[[unscaled_fields[i],i] for i in range(basis.L)]
    disorder_field = [["z",h_z]]    
    no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
    Hz=hamiltonian(disorder_field,[],basis=basis,dtype=np.float64,**no_checks)
    H_total=H_XXZ+h_disorder*Hz

    return H_total


def trained_drive(psi_0,disorder_all,basis,tm):
    #np.random.seed()   
    #perform n_M layers of quenches
    for m in range(n_M):
        disorder_m = disorder_all[m]   
        H_total = realization(H_XXZ,basis,disorder_m,h_d)
        psi_f = H_total.evolve(psi_0,ti,tm)

    return psi_f


def hamming_traj(psi_0,disorder_all,basis,tm):
    #np.random.seed()
    hd_traj = np.zeros(n_M)    
    
    #list of sigma_z's  
    z_opt_list = np.zeros([L,basis.Ns,basis.Ns],dtype = 'complex_')
    for i in range(L):
        z_opt_list[i] = Z_opt(i)
    
    z_opt_evolve = np.copy(z_opt_list)
    #perform n_M layers of quenches
    for m in range(n_M):
        t0 = time()
        disorder_m = disorder_all[m]    
        H_total = realization(H_XXZ,basis,disorder_m,h_d)             
        
        #time-evolution operator (num=number of time-steps during the time-evolution)
        U_opt = exp_op(H_total,a=-1j,start=ti,stop=tm,num=10)
        
        #time-evolve each sigma_z
        for i in range(L):
            z_opt_evolve[i] = (U_opt.sandwich(z_opt_evolve[i]))[:,:,-1]
        
        correlator = 0 
        for i in range(L):
            correlator += psi_0.T @ z_opt_evolve[i] @ z_opt_list[i] @ psi_0
        
        hd_traj[m] = 1/2 - 1/(2*L) * correlator.real
        
        tf = time()        
        print('m='+str(m),', hd='+str(np.round(hd_traj[m],4)),', time used = ',np.round(tf-t0,2))

    return hd_traj


#given a state calculate probability distribution by Born's rule
def born_prob(psi_m):
    return np.abs(psi_m)**2/np.sum(np.abs(psi_m)**2)

def reduced_prob(psi_m):
    rho_v = basis.partial_trace(psi_m,sub_sys_A=tuple(range(L_visible)))
    probs = (np.diag(rho_v)/np.sum(np.diag(rho_v))).real
    return probs.astype('float')

def sampling(p_model,n_iter):
    sample_config = np.zeros(N_visible)

    for t in range(n_iter):
        dice = np.random.rand(N_visible)
        dice /= np.sum(dice)
        
        inds = np.where(p_model>dice)[0]
        inds_comp = np.setdiff1d(np.arange(0,64,1),inds)
        sample_config[inds] = 1
        sample_config[inds_comp] =0
        
    return sample_config/np.sum(sample_config)
        
    
#calculate the sigma_z operator at site i
def Z_opt(index,basis):       
    I_2 = np.eye(2)
    sigma_z = np.array([[1,0],[0,-1]])
    
    z_i = np.eye(1)
    for j in range(basis.L):
        if j==index:
            matrix = sigma_z
        else:
            matrix = I_2
        z_i = np.kron(z_i,matrix)
    
    return z_i


def local_magnitization(psi_m,basis):
    L = basis.L
    rho_m = np.outer(psi_m,psi_m)    
    z_expect = np.zeros(L)
    
    for i in range(L):
        z_opt = Z_opt(i,basis)           
        z_expect[i] = np.trace(rho_m @ z_opt)
    
    return z_expect

def IPR(psi_m,basis):
    L = basis.L
    z_expect = np.zeros(L)
    
    for i in range(L):
        z_opt = Z_opt(i,basis)           
        z_expect[i] = psi_m.T @ z_opt @ psi_m
    
    return np.sum(z_expect**4)
 
    
def fidelity(psi_m,q_data):    
    p_m = reduced_prob(psi_m)
    return np.sum(np.sqrt(p_m*q_data))**2


mmd = RBFMMD2(sigma_list=[0.1,0.25,4,10], basis=np.arange(2**L_visible))
def MMD_loss(psi_m,q_data):
    p_model = reduced_prob(psi_m)
    return mmd(p_model,q_data)

#calculate level statistics
def level_stat(E_all):
    DeltaE = E_all[1:] - E_all[:-1]
    rE = DeltaE[1:]/DeltaE[:-1]
    rE_correct = np.where(rE>1,1/rE,rE)
    return rE_correct

#===============================preprocess data=============================#

if dataset == 'uniform':
    #uniform dist.
    n_examples = 10000
    data_random = (1+np.sign(np.random.randn(n_examples,N_visible)))/2
    #calculate emprical probability distribution of data
    data_freq = np.sum(data_random,axis=0)
    q_data = data_freq/np.sum(data_freq)

elif dataset == 'normal':    
    #gaussian dist.
    n_examples = 10000
    w_normal = stats.norm.ppf(np.linspace(1,n_examples,n_examples,endpoint=False)/n_examples, loc=125,scale=20)
    freq_data, bins_data = np.histogram(w_normal,bins=N_visible)
    freq_data += 1
    q_data = freq_data/np.sum(freq_data)
    q_data_test = q_data

elif dataset == 'bimodal':
    #bimodal dist.
    n_examples = 10000
    ecdf = np.linspace(-1/2+1/n_examples,1/2-1/n_examples,n_examples//2) + 1/2
    w1 = stats.norm.ppf(ecdf, 50, 10)
    w2 = stats.norm.ppf(ecdf, 150, 10)
    w_bimodal = np.sort(np.concatenate([w1, w2], axis=0))
    freq_data, bins_data = np.histogram(w_bimodal,bins=N_visible)
    freq_data += 1
    q_data = freq_data/np.sum(freq_data)
    print('p_data', q_data)
elif dataset == 'parity':
    basis_parity = spin_basis_1d(L_visible,pauli=False)
    #bimodal dist.
    even_inds = []
    for i in range(basis_parity.Ns):
        psi_i = np.zeros([basis_parity.Ns],dtype = 'complex_')
        psi_i[i] = 1
        par = np.prod(local_magnitization(psi_i,basis_parity))
        if par == 1:
            even_inds.append(i)
    
    freq_data = np.zeros(basis_parity.Ns)
    freq_data[even_inds] = 1
    q_data = freq_data/np.sum(freq_data)
    q_data_test = q_data

elif dataset == 'MNIST':
    #MNIST dataset
    mnist = tf.keras.datasets.mnist
    (images_mnist, labels_mnist), (images_mnist_test, labels_mnist_test) = mnist.load_data()
    
    data_digit = images_mnist[np.where(labels_mnist==digit)[0]]

    #subsample MNIST to sqrt(2^L_visible) pixels
    data_digit_sub = np.zeros([data_digit.shape[0],size,size])  
    for i in range(data_digit.shape[0]):
        data_digit_sub[i] = cv2.resize(data_digit[i].reshape([28,28]), dsize=(size,size), interpolation=cv2.INTER_CUBIC)
    
    data_digit_sub = np.reshape(data_digit_sub,[-1,size*size])
    data_digit_sub = np.heaviside((data_digit_sub-0.5)*2,1)
    
    freq_data = np.sum(data_digit_sub,axis=0)
    freq_data += np.random.rand(size*size) #1
    q_data = freq_data/np.sum(freq_data)

    
    #repeat the process for test set
    data_digit_test = images_mnist_test[np.where(labels_mnist_test==digit)[0]]
    
    data_digit_test_sub = np.zeros([data_digit_test.shape[0],size,size])  
    for i in range(data_digit_test.shape[0]):
        data_digit_test_sub[i] = cv2.resize(data_digit_test[i].reshape([28,28]), dsize=(size,size), interpolation=cv2.INTER_CUBIC)
    
    data_digit_test_sub = np.reshape(data_digit_test_sub,[-1,size*size])
    data_digit_test_sub = np.heaviside((data_digit_test_sub-0.5)*2,1)
    
    freq_data_test = np.sum(data_digit_test_sub,axis=0)
    freq_data_test += np.random.rand(size*size) #1
    q_data_test = freq_data_test/np.sum(freq_data_test)

    
elif dataset == 'MBL':
    h_data = 3.9
    freq_MBL_data_all = np.zeros([n_real,n_bins])
    #set up initial MBL state
    disorder_data = np.random.rand(L_visible)
    basis_data = spin_basis_1d(L_visible,pauli=False)
    
    # define operators with PBC using site-coupling lists
    J_zz_data = [[Jzz_0,i,(i+1)] for i in range(L_visible-1)] # PBC
    J_xy_data = [[Jxy/2.0,i,(i+1)] for i in range(L_visible-1)] # PBC
    
    # static and dynamic lists
    static_data = [["+-",J_xy_data],["-+",J_xy_data],["zz",J_zz_data]]
    H_XXZ_data = hamiltonian(static_data,dynamic,basis=basis_data,dtype=np.float64)
    H_MBL_data = realization(H_XXZ_data,basis_data,disorder_data,h_data)
    # calculate the energy at infinite temperature for initial MBL Hamiltonian
    eigsh_args={"k":2,"which":"BE","maxiter":1E4,"return_eigenvectors":False}
    Emin,Emax=H_MBL_data.eigsh(**eigsh_args)
    E_inf_temp=(Emax+Emin)/2.0
    # calculate nearest eigenstate to energy at infinite temperature
    E_mid,psi_mid=H_MBL_data.eigsh(k=1,sigma=E_inf_temp,maxiter=1E4)
    psi_mid=psi_mid.reshape((-1,))
    
    E_MBL_data = H_MBL_data.eigvalsh()
    r_MBL_data = level_stat(E_MBL_data)
    freq_MBL_data, bins_MBL_data = np.histogram(r_MBL_data, bins=bins_fixed, density=True) 
    
    freq_data = born_prob(psi_mid)
    q_data = freq_data/np.sum(freq_data)
    q_data_test = np.copy(q_data)

    
#==========================set up simulation================================#

psi_all = np.zeros([n_M,basis.Ns],dtype = 'complex_')
disorder_all = np.zeros([n_M,basis.L])

fidelity_score = np.zeros([n_M])
MMD_score = np.zeros([n_M])

mag_traj = np.zeros([n_M,L])
Ent_traj = np.zeros([n_M])
IPR_traj = np.zeros([n_M])

freq_MBL_train_all = np.zeros([n_real,n_bins])

np.random.seed(seed)
#set up initial product state
ind_0 = basis.state_to_int('111000')
psi_0 = np.zeros([basis.Ns],dtype = 'complex_')
psi_0[ind_0] = 1
psi_m = np.copy(psi_0)


p_model_training = np.zeros([5,N_visible])

#perform n_M layers of quenches
for m in range(n_M):
    t0 = time()
    disorder_m = np.random.rand(n_real,basis.L)
    psi_candidates = np.zeros([n_real,basis.Ns],dtype = 'complex_')
    loss_candidates = np.zeros([n_real])
    
    
    for i in range(n_real):
        H_total = realization(H_XXZ,basis,disorder_m[i],h_d)
        psi_candidates[i] = H_total.evolve(psi_m,ti,tm)

        #optimize for MMD-loss
        loss_candidates[i] = MMD_loss(psi_candidates[i],q_data) 
        
    #find minimum (MMD-loss)
    ind_opt = np.argmin(loss_candidates)

    psi_m = psi_candidates[ind_opt]
    print('p_model best', reduced_prob(psi_m))
    psi_all[m] = psi_m
    disorder_all[m] = disorder_m[ind_opt]
    
    #evaluate loss functions on test set
    fidelity_score[m] = fidelity(psi_m,q_data_test)
    MMD_score[m] = MMD_loss(psi_m,q_data_test)
    
    #evaluate physical quantities
    mag_traj[m] = local_magnitization(psi_m,basis)
    Ent_traj[m] = basis.ent_entropy(psi_m,range(basis.L//2),alpha=1.0)['Sent_A']
    
    tf = time()
    print('iter=',m,', loss=',np.round(loss_candidates[ind_opt],4), ', time used = ', np.round(tf-t0,2))

    if m%int(0.2*n_M) == 0:
        p_model_training[m//int(0.2*n_M)] = reduced_prob(psi_all[m])
    
#final model probability distribution
p_model_trained = reduced_prob(psi_all[-1])

#===================================plotting code====================================
bins_center = (np.arange(0,size*size,1)+np.arange(1,size*size+1,1))/2

#MNIST imshow range
r_min = 0.02
r_max = 0.6*np.max(q_data_test)

fig0, ((ax01, ax02), (ax03, ax04), (ax05, ax06)) = plt.subplots(3, 2,figsize=(20,16))

if dataset == 'MNIST':
    ax01.imshow(np.reshape(p_model_training[0],[size,size]),vmin=r_min,vmax=r_max)
    ax01.set_title('m_quenches='+str(int(0*n_M)))
    
    ax02.imshow(np.reshape(p_model_training[1],[size,size]),vmin=r_min,vmax=r_max)
    ax02.set_title('m_quenches='+str(int(0.2*n_M)))
      
    ax03.imshow(np.reshape(p_model_training[2],[size,size]),vmin=r_min,vmax=r_max)
    ax03.set_title('m_quenches='+str(int(0.4*n_M)))
      
    ax04.imshow(np.reshape(p_model_training[3],[size,size]),vmin=r_min,vmax=r_max)
    ax04.set_title('m_quenches='+str(int(0.6*n_M)))
     
    ax05.imshow(np.reshape(p_model_training[4],[size,size]),vmin=r_min,vmax=r_max) 
    ax05.set_title('m_quenches='+str(int(0.8*n_M)))
       
    ax06.imshow(np.reshape(p_model_trained,[size,size]),vmin=r_min,vmax=r_max)    
    ax06.set_title('m_quenches='+str(int(1*n_M)))


else:
    ax01.bar(bins_center,p_model_training[0])
    ax01.set_title('m_quenches='+str(int(0*n_M)))
    
    ax02.bar(bins_center,p_model_training[1])
    ax02.set_title('m_quenches='+str(int(0.2*n_M)))

    ax03.bar(bins_center,p_model_training[2])
    ax03.set_title('m_quenches='+str(int(0.4*n_M)))

    ax04.bar(bins_center,p_model_training[3])
    ax04.set_title('m_quenches='+str(int(0.6*n_M)))

    ax05.bar(bins_center,p_model_training[4])
    ax05.set_title('m_quenches='+str(int(0.8*n_M)))

    ax06.bar(bins_center,p_model_trained)
    ax06.set_title('m_quenches='+str(int(1*n_M))) 
     
plt.tight_layout()


fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(20,16))

ax1.plot(np.arange(0,n_M,1),np.log10(MMD_score),linewidth=5)
ax1.set_xlabel('m_quenches')
ax1.set_ylabel(r'$\log_{10}$'+'(MMD)')
ax1.set_ylim([-3,1])

ax3.plot(np.arange(0,n_M,1),fidelity_score,linewidth=5)
ax3.set_xlabel('m_quenches')
ax3.set_ylabel('KL div.')
  

if dataset == 'MNIST':
    
    ax2.imshow(np.reshape(q_data_test,[size,size]),vmin=r_min,vmax=r_max)
    ax4.imshow(np.reshape(p_model_trained,[size,size]),vmin=r_min,vmax=r_max)
    
    ax2.set_title('target distribution')
    ax4.set_title('learned distribution')
    
else:
    ax2.bar(bins_center, q_data, align='center',alpha=0.5,label='q_data')
    ax2.bar(bins_center, p_model_trained, align='center',alpha=0.5,label='p_model')
    ax2.legend()
    ax2.set_xlabel('state space')
    ax2.set_ylabel('prob. dist.')
        
    ax4.plot(np.arange(0,n_M,1),fidelity_score,linewidth=5)
    ax4.set_xlabel('m_quenches')
    ax4.set_ylabel('Fidelity')

plt.tight_layout() 

#plot local magnitizations and hamming distance
fig2, ((ax5, ax6), (ax7, ax8)) = plt.subplots(2, 2,figsize=(20,16))

for i in range(L):
    ax5.plot(np.arange(0,n_M,1),mag_traj[:,i],label=str(i)+'th spin')
ax5.set_xlabel('m_quenches')
ax5.set_ylabel(r'$<\sigma_i^z>$')
ax5.legend()

IPR_traj = np.sum(mag_traj**4,axis=-1)
ax6.plot(np.arange(0,n_M,1),IPR_traj)
ax6.set_xlabel('m_quenches')
ax6.set_ylabel('IPR')

ax7.plot(np.arange(0,n_M,1),Ent_traj,linewidth=5)
ax7.set_xlabel('m_quenches')
ax7.set_ylabel(r'$S_{ent}$')

#hd_traj = hamming_traj(psi_0,disorder_all,basis,tm)


fig0.suptitle(descp,fontsize=32)
fig1.suptitle(descp,fontsize=32)    
fig2.suptitle(descp,fontsize=32)    
#fig3.suptitle(descp,fontsize=32)    



plt.show()
