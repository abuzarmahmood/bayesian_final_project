
# Assumptions:
#1) Neuron firing distributed as either Gamma, Exponential or Normal
#2) Neurons in the population encode information by specific patterns of individual firing rates
#3) Neurons have different firing patterns for different tastes
#4) Firing 0.5-3s after stimulus delivery is relevant (relevant indices: 2500 - 5000)

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

# Create functions to use for data fitting   
# Probably a good idea to define a class for this but ain't nobody got time for at that :P

def gamma_dist(x,params):
    p = ((params[1]**params[0])/sp.gamma(params[0]))*(x**(params[0]-1))*(np.exp(-params[1]*x))
    return p

def gamma_lik(x,params): #Actually log lik
    #p = sum(np.log(gamma_dist(x[np.nonzero(x)],params))) # Nonzero is imp because taking log
    p = -sum(np.log(gamma_dist(x,params)))
    if np.isinf(p) + np.isnan(p):
        p = 0
    return p


def exp_dist(x,params): # Because the others have 2 parameters
    p = params[1]*np.exp(-params[1]*x)
    return p

def exp_lik(x,params):
    #p = sum(np.log(exp_dist(x[np.nonzero(x)],params)))
    p = -sum(np.log(exp_dist(x,params)))
    return p


def norm_dist(x,params):
    p = (1/(np.sqrt(2*np.pi*params[1])))*np.exp(-((x-params[0])**2)/(2*params[1]))
    return p

def norm_lik(x,params):
    #p = sum(np.log(norm_dist(x[np.nonzero(x)],params)))
    p = -sum(np.log(norm_dist(x,params)))
    return p


## Load dataset
workdir = '/media/sf_shared_folder/jian_you_data/all_tastes_together/file_1'
os.chdir(workdir)
dat = tables.open_file('jy05_20170324_2500mslaser_170324_103823_repacked.h5')

# Concatenate all non-optogenetic trials from all neurons into an array
# laser_duration == 0
stimuli = ['dig_in_0','dig_in_1','dig_in_2','dig_in_3']

# Array structure: [trial, nrn, time]
control_dat = []
all_dat = []
for stim in stimuli:
    exec('dig_dat = dat.root.spike_trains.%s' %stim)
    # Firing rate indexed by time according to assumption 4)
    control_dat.append(dig_dat.spike_array[np.where(dig_dat.laser_durations[:]==0)[0],:,2500:5000])
    all_dat.append(dig_dat.spike_array[:,:,3000:5000])
    # Y u break my heart python :(   </3

# Since data are selected randomly, run multiple times to get distribution of accuracy

def bay_decode(data): # List with array for every taste
    # List of spikes from every neuron for 4 tastes
    # Break dataset into training (7 trials) and testing (7 trials) pieces
    # Tossing one trial for indexing symmetry
    train_spikes = []
    test_spikes = []
    trial_num = data[0].shape[0]
    if ((trial_num%2)==1): # Split trials evenly
        trial_num -=1
    train_inds = np.random.choice(range(trial_num), int(trial_num/2), replace=False)
    test_inds = np.delete(range(trial_num),train_inds) 
    # Doesn't matter to use same inds for different tastes since trials are not repeated across tastes
    for taste in data:
        train_spikes.append(taste[train_inds,:,:])
        test_spikes.append(taste[test_inds,:,:])
        
    # Convert spike trains to approximate firing rate by moving mean
    train_rate = []
    test_rate = []
    tot_time = train_spikes[0].shape[2]
    window = 200
    step = 25
    # Welcome to for loop hell!
    for taste_ind in range(len(train_spikes)):
        train_rate_array = np.zeros((train_spikes[0].shape[0],train_spikes[0].shape[1],int((tot_time-window)/step)-1))
        test_rate_array = np.zeros(train_rate_array.shape)
        for nrn in range(train_spikes[0].shape[1]):
            for trial in range(train_spikes[0].shape[0]):
                for time_bin in range(train_rate_array.shape[2]):
                    # 0.01 added to avoid non-zero values because we're using log-likelihood
                    train_rate_array[trial,nrn,time_bin] = np.sum(train_spikes[taste_ind][trial,nrn,(time_bin*step):(time_bin*step)+window]) + 0.01
                    test_rate_array[trial,nrn,time_bin] = np.sum(test_spikes[taste_ind][trial,nrn,(time_bin*step):(time_bin*step)+window]) + 0.01
        
        train_rate.append(train_rate_array)
        test_rate.append(test_rate_array)
         
    
    
    ## Test plot to visualize log-likelihood function
    #x = np.linspace(0.1,100,100)
    #y = np.linspace(0.1,100,100)
    #X,Y = np.meshgrid(x,y)
    #Z = np.zeros(X.shape)
    #for i in range(len(x)):
    #    for j in range(len(y)):
    #        Z[i,j] = gamma_lik_opti([x[i],y[j]])
    #
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(X, Y, np.log(Z), rstride=2, cstride=2)
    #plt.show()
    
    
    
    # Fit data to distribution
    train_params = []
    for taste_ind in range(len(train_spikes)):
        train_params_temp = np.zeros((train_spikes[0].shape[1],3)) # First 2 for parameters, 3rd for class of PDF
        for nrn in range(train_spikes[0].shape[1]):
            temp_dat = np.ndarray.flatten(train_rate[taste_ind][:,nrn,:])
            temp_dat = temp_dat[np.nonzero(temp_dat)]
            
            # Gamma
            gam_fit = stat.gamma.fit(temp_dat)
            gam_fit = np.array([gam_fit[0],gam_fit[2]])
            # Exponential
            exp_fit = stat.expon.fit(temp_dat)
            # Normal
            nor_fit = stat.norm.fit(temp_dat)
            
            liks = [gamma_lik(temp_dat,gam_fit),exp_lik(temp_dat,exp_fit),norm_lik(temp_dat,nor_fit)]
            fits = [gam_fit, exp_fit, nor_fit]
            final_fit = np.asarray(fits[np.argmin(liks)])
            train_params_temp[nrn,0:2] = final_fit
            train_params_temp[nrn,2] = np.argmin(liks)
    
        train_params.append(train_params_temp)
        
    # Function that takes in x_range, parameters (with type) and spits out corresponding  plot
    def pdf_plot(x,params):
        if params[2] == 0:
            y = gamma_dist(x,params)
        elif params[2] == 1:
            y = exp_dist(x,params)
        elif params[2] == 2:
            y = norm_dist(x,params)
        return y
        
    # Function that calculates likelihood based on type of distribution
    def all_lik(x,params):
        if params[2] == 0:
            y = gamma_lik(x,params)
        elif params[2] == 1:
            y = exp_lik(x,params)
        elif params[2] == 2:
            y = norm_lik(x,params)
        return y
    
    colors = ['r','g','b']
    
    ### Test plots for firing rate histogram to gamma fit
    ### *Clapping emoji*
#    taste_ind = 2
#    nrn = 0
#    temp_dat = np.ndarray.flatten(train_rate[taste_ind][:,nrn,:])
#    plt.hist(temp_dat,density = True)
#    x = np.linspace(0,10,100)
#    params = train_params[taste_ind][nrn,:]
#    y = pdf_plot(x,params)
#    plt.plot(x,y,colors[int(params[2])])
#    plt.xlabel('Firing rate')
#    plt.ylabel('Empirical probability')
#    plt.legend()
#    #
#    ## All da neurons (and plots)
#    rows = len(train_rate)
#    cols = train_rate[0].shape[1]
#    count = 1
#    x = np.linspace(0,10,100)
#    fig, axes = plt.subplots(rows,cols,sharex = 'all',sharey='all')
#    for i in range(rows):
#        for j in range(cols):
#            axes[i,j].hist(np.ndarray.flatten(train_rate[i][:,j,:]),density=1)
#            params = train_params[i][j]
#            #y = pdf_plot(x,params)
#            #axes[i,j].plot(x,y,colors[int(params[2])])
#            count +=1
#            print(count)
#    
#    fig.tight_layout()
        
    ## Use trained likelihood to calculate which taste is being delivered
    # 1) Take firing of all neurons from ONE TRIAL and calculate likelihood for EVERY taste
    # 2) Find which taste gives maximum likelihood
    
    # So for every trial, there should be four numbers corresponding to the likehlihood 
    # of the firing belonging to a particular taste
    
    # Convert test_rate to an array so every trial can be fed through the loop
    test_rate_fin = np.array(test_rate) #[taste, nrn, trial, firing_rate]
    dims = test_rate_fin.shape
    test_rate_fin = np.reshape(test_rate_fin,(dims[0]*dims[1],dims[2],dims[3]))
    
    test_liks = np.zeros((test_rate_fin.shape[0],len(test_rate))) # 4 numbers for every trial
    for trial in range(test_rate_fin.shape[0]): # Loop over all test trials
        for taste in range(len(test_rate)): # Loop over every taste model
            all_nrns = np.zeros(test_rate_fin.shape[1]) # Likelihood of every nrn for 1 taste model
            for nrn in range(len(all_nrns)):    # Loop over every neuron for 1 taste models
                nrn_dat = test_rate_fin[trial,nrn,:]
                all_nrns[nrn] = all_lik(nrn_dat,train_params[taste][nrn])
            test_liks[trial,taste] = np.sum(all_nrns)
            
    decoded_taste = np.argmin(test_liks,axis=1)
    taste_labels = np.ndarray.flatten(np.matlib.repmat(np.array([0,1,2,3]),int(trial_num/2),1),order='F')
    decode_accuracy = sum(taste_labels==decoded_taste)*100/len(taste_labels)
    
    return decode_accuracy

acc = np.zeros((1,10))[0]
for i in range(len(acc)):
    acc[i] = bay_decode(control_dat)
    print(i)