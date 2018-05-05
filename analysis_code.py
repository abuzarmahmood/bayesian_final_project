
# Assumptions:
#1) Gamma distributed firing rates
#2) Neurons in the population encode information by specific patterns of individual firing rates
#3) Neurons have different firing patterns for different tastes
#4) Only firing 1-2s after stimulus delivery is relevant (relevant indices: 3000 - 4000)

# Import libraries
import os
import tables
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.signal as sig
import scipy.optimize as opt
import scipy.stats as stat
from scipy import special as sp
import pdb


## Load dataset
workdir = '/media/sf_shared_folder/jian_you_data/all_tastes_together/file_1'
os.chdir(workdir)
dat = tables.open_file('jy05_20170324_2500mslaser_170324_103823_repacked.h5')

# Concatenate all non-optogenetic trials from all neurons into an array
# laser_duration == 0
stimuli = ['dig_in_0','dig_in_1','dig_in_2','dig_in_3']

# Array structure: [trial, nrn, time]
control_dat = []
for stim in stimuli:
    exec('dig_dat = dat.root.spike_trains.%s' %stim)
    # Firing rate indexed by time according to assumption 4)
    control_dat.append(dig_dat.spike_array[np.where(dig_dat.laser_durations[:]==0)[0],:,3000:5000])
    # Y u break my heart python :(   </3

# List of spikes from every neuron for 4 tastes
# Break dataset into training (7 trials) and testing (7 trials) pieces
# Tossing one trial for indexing symmetry
train_spikes = []
test_spikes = []
#train_inds = np.random.randint(0,15,7)
train_inds = np.random.choice(range(15), 7, replace=False)
test_inds = np.delete(np.arange(15),train_inds)[:-1] 
# Doesn't matter to use same inds for different tastes since trials are not repeated across tastes
for taste in control_dat:
    train_spikes.append(taste[train_inds,:,:])
    test_spikes.append(taste[test_inds,:,:])
    
# Convert spike trains to approximate firing rate by moving mean
train_rate = []
test_rate = []
tot_time = train_spikes[0].shape[2]
window = 200
step = 50
# Welcome to for loop hell!
for taste_ind in range(len(train_spikes)):
    train_rate_array = np.zeros((train_spikes[0].shape[0],train_spikes[0].shape[1],int((tot_time-window)/step)-1))
    test_rate_array = np.zeros(train_rate_array.shape)
    for nrn in range(train_spikes[0].shape[0]):
        for trial in range(train_spikes[0].shape[1]):
            for time_bin in range(train_rate_array.shape[2]):
                #(pdb)
                train_rate_array[nrn,trial,time_bin] = np.sum(train_spikes[taste_ind][nrn,trial,(time_bin*step):(time_bin*step)+window])
                test_rate_array[nrn,trial,time_bin] = np.sum(test_spikes[taste_ind][nrn,trial,(time_bin*step):(time_bin*step)+window])
    
    train_rate.append(train_rate_array)
    test_rate.append(test_rate_array)
     
## Test plots for spikes to firing rate  
#inds = [3,1]
#spikes = train_spikes[0][inds[0],inds[1],:]
#rate = train_rate[0][inds[0],inds[1],:]
#rate2 = sig.resample(rate,spikes.shape[0])
#plt.plot(spikes)
#plt.plot(rate2)

# For training set, for all tastes, for every neuron, calculate mean and standard deviation
# These will be used as starting parameters for optimization
#train_params = []
#for taste_ind in range(len(train_spikes)):
#    train_params_temp = np.zeros((train_spikes[0].shape[0],2))
#    for nrn in range(train_spikes[0].shape[0]):
#        train_params_temp[nrn,0] = np.mean(train_rate[taste_ind][nrn,:,:])
#        train_params_temp[nrn,1] = np.std(train_rate[taste_ind][nrn,:,:])
#
#    train_params.append(train_params_temp)
    
# Now perform, maximum likelihood fit of gamma function for each neuron for each taste
    
temp_dat = np.ndarray.flatten(train_rate[0][:,1,:])

def gamma_dist(x,a,b):
    if a == 0:
        gamma_val = 1
    else:
        gamma_val = sp.gamma(a)
    p = ((b**a)/gamma_val)*(x**(a-1))*(np.exp(-b*x))
    return p

def gamma_lik(x,params): #Actually log lik
    p = sum(np.log(gamma_dist(x[np.nonzero(x)],params[0],params[1])))
    if np.isinf(p) + np.isnan(p):
        p = 0
    return p

def gamma_lik_opti(params):
    p = -gamma_lik(temp_dat,params)
    return p


# Finite difference gradient for gamma disribution
def gamma_lik_grad(params): # Too many nested definitions :p
    p_grad = np.zeros([1,2])[0]
    epsi = 1e-2
    params = np.array(params)
    pert0 = params + np.array([epsi,0])
    pert1 = params + np.array([0,epsi])
    p_grad[0] = (gamma_lik_opti(pert0) - gamma_lik_opti(params))/epsi
    p_grad[1] = (gamma_lik_opti(pert1) - gamma_lik_opti(params))/epsi
    return p_grad


# Test plot for likelihood function
x = np.linspace(0.1,100,100)
y = np.linspace(0.1,100,100)
X,Y = np.meshgrid(x,y)
Z = np.zeros(X.shape)
for i in range(len(x)):
    for j in range(len(y)):
        Z[i,j] = gamma_lik_opti([x[i],y[j]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, np.log(Z), rstride=2, cstride=2)
plt.show()
#
### Shitty gradient vector code
#arrow_x = 8
#arrow_y = 2
#arrow_grad = gamma_lik_grad([arrow_x,arrow_y])
#dx = arrow_grad[0]
#dy = arrow_grad[1]
#plt.arrow(8,2,dx,dy)



train_params = []
all_exit_flags = []
for taste_ind in range(len(train_spikes)):
    train_params_temp = np.zeros((train_spikes[0].shape[1],2))
    exit_temp = np.zeros((train_spikes[0].shape[1],1))
    for nrn in range(train_spikes[0].shape[1]):
        temp_dat = np.ndarray.flatten(train_rate[taste_ind][:,nrn,:])
        result = opt.fmin_tnc(func=gamma_lik_opti, x0=[2,1], fprime=None, approx_grad=True, bounds = [(0,100),(0,100)])
        #result = stat.gamma.fit(temp_dat)
        #params = np.array([result[0],result[2]])
        params = result[0]
        exit_flag = (result[2] == 1) + (result[2] == 2)
        train_params_temp[nrn] = params
        exit_temp[nrn] = exit_flag 

    train_params.append(train_params_temp)
    all_exit_flags.append(exit_temp)
    
## Test plots for firing rate histogram to gamma fit
## *Clapping emoji*
taste_ind = 0
nrn = 0
temp_dat = np.ndarray.flatten(train_rate[taste_ind][:,nrn,:])
plt.hist(temp_dat,density = True)
x = np.linspace(0,10,100)
params = train_params[taste_ind][nrn]
y = gamma_dist(x,params[0],params[1])
plt.plot(x,y)
    
## Use trained likelihood to calculate which taste is being delivered
# 1) Take firing of all neurons from ONE TRIAL and calculate likelihood for EVERY taste
# 2) Find which taste gives maximum likelihood

# So for every trial, there should be four numbers corresponding to the likehlihood 
# of the firing belonging to a particular taste

# Convert test_rate to an array so every trial can be fed through the loop
test_rate_fin = np.array(test_rate) #[taste, nrn, trial, firing_rate]
test_rate_fin = np.reshape(test_rate_fin,(28,14,35))

test_liks = np.zeros((test_rate_fin.shape[0],len(test_rate))) # 4 numbers for every trial
for trial in range(test_rate_fin.shape[0]): # Loop over all test trials
    for taste in range(len(test_rate)): # Loop over every taste model
        all_nrns = np.zeros(test_rate_fin.shape[1]) # Likelihood of every nrn for 1 taste model
        for nrn in range(len(all_nrns)):    # Loop over every neuron for 1 taste models
            nrn_dat = test_rate_fin[trial,nrn,:]
            all_nrns[nrn] = gamma_lik(nrn_dat,train_params[taste][nrn])
        test_liks[trial,taste] = np.sum(all_nrns)
        
decoded_taste = np.argmin(test_liks,axis=1)     
            

# Sum over all nrns in 1 trial to find probability for every trial
all_test_liks = np.sum(np.array(all_test_liks),axis=2)
