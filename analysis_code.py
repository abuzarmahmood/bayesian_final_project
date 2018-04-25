#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:20:27 2018

@author: abuzarmahmood
"""

# Assumptions:
#1) Gaussian distributed firing rates
#2) Neurons in the population encode information by specific patterns of individual firing rates
#3) Only firing 1-2s after stimulus delivery is relevant (relevant indices: 3000 - 4000)

# Import libraries
import os
import tables
import numpy as np

## Load dataset
workdir = '/media/sf_shared_folder/jian_you_data/all_tastes_together/file_1'
os.chdir(workdir)
dat = tables.open_file('jy05_20170324_2500mslaser_170324_103823_repacked.h5')

# Concatenate all non-optogenetic trials from all neurons into an array
stimuli = ['dig_in_0','dig_in_1','dig_in_2','dig_in_3']

control_dat = []
for stim in stimuli:
    exec('dig_dat = dat.root.spike_trains.%s' %stim)
    # Firing rate indexed by time according to assumption 3)
    control_dat.append(dig_dat.spike_array[np.where(dig_dat.laser_durations[:]>0)[0],:,3000:4000])
    # Y u break my heart python :(   </3

# List of firing rate from every neuron for 4 tastes
# For all tastes, for every neuron, calculate mean and standard deviation