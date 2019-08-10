
############################################
# Code for analysing impulse response imaging experiment data
# Authors: Heather Chang and Minseung Choi
# 2019 July 29
##############################################

from visanalysis.imaging_data import BrukerData
from visanalysis.analysis import utils

import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import os

plt.rcParams.update({'font.size': 20})

class ImpulseStimulusAnalysis():

    def __init__(self, fn='2019-07-09', series_number=1, z_index=0, sample_rate=500):
        # Define the file name and series number you want to use
        self.fn = fn
        self.series_number = series_number

        self.imaging_data = BrukerData.ImagingDataObject(self.fn, self.series_number, load_rois = True, z_index=z_index, sample_rate=sample_rate)
        # Option to plot individual epoch responses on top of the mean response

        some_roi_dict = self.get_roi_dict(self.get_roi_set_names()[0])
      #  self.epoch_response_matrix = self.some_roi_dict['epoch_response']
        self.n_epochs = some_roi_dict['epoch_response'].shape[1] #epoch of stimulus
       # self.n_rois = self.some_roi_dict['epoch_response'].shape[0]

        some_reponse_time_vector = self.get_response_time_vector(self.get_roi_set_names()[0])
        self.imaging_rate = 1/(np.mean(np.diff(some_reponse_time_vector)))
        self.n_sample = len(some_reponse_time_vector)

        #our frame rate should be similar to imaging rate
        #difference of time points, average, 1/that for rate
       # self.imaging_rate = 1/(np.mean(np.diff(some_roi_dict['time_vector'])))
        self.impulse_stimulus = None
        self.idle_color = None
        self.stim_duration = None
        self.pre_duration = None
        self.post_duration = None
        self.epoch_duration = None
        self.stim_time_vector = None

        self.__get_epoch_period_durations()
        self.__get_stim_time_vector()

        self.intensity_indices = None #check these???
        self.size_indices = None

    def __get_epoch_period_durations(self):

        with h5py.File(os.path.join(self.imaging_data.flystim_data_directory, self.imaging_data.file_name) + '.hdf5', 'r') as experiment_file:
            session = experiment_file['/epoch_runs'].get(str(self.imaging_data.series_number))
            self.pre_duration = session.attrs['pre_time']
            self.stim_duration = session.attrs['stim_time']
            self.post_duration = session.attrs['tail_time']
            self.idle_color = session.attrs['idle_color']
        #how long the entire duration
        self.epoch_duration = self.pre_duration + self.stim_duration + self.post_duration #seconds
        return

    def __get_stim_time_vector(self):

       # 1/ imaging rate to get mean of imaging intervals, divide by oversample
       #  rate to represent stimulus well so it doesn't get stretched out
       #  when trying to fit into imaging frames
        stim_time_diff = 1 / self.imaging_rate
        stimulus = np.zeros(self.n_sample) + self.idle_color
        self.stim_time_vector = np.arange(0, stim_time_diff * len(stimulus), stim_time_diff)
        return

    def get_roi_set_names(self):
        return [*self.imaging_data.roi]

    def get_n_roi_in_roi_set(self, roi_set):
        return self.imaging_data.roi[roi_set]['roi_response'].shape[0]

    def get_roi_dict(self, roi_set):
        return self.imaging_data.roi.get(roi_set)
    # Pull out the time_vector and the epoch_response_matrix from this roi set

    #time at which the images were taken.
    def get_response_time_vector(self, roi_set):
        return self.get_roi_dict(roi_set)['time_vector']

    def get_epoch_indices_by_stimulus_type(self, is_bright=True, is_small=True):

        if is_bright:
            self.intensity_indices = [i for i in range(self.n_epochs) if self.imaging_data.epoch_parameters[i]['current_intensity'] == 1]
        else:
            self.intensity_indices = [i for i in range(self.n_epochs) if self.imaging_data.epoch_parameters[i]['current_intensity'] == 0]

        if is_small:
            self.size_indices = [i for i in range(self.n_epochs) if self.imaging_data.epoch_parameters[i]['current_width'] == 20]
        else:
            self.size_indices = [i for i in range(self.n_epochs) if self.imaging_data.epoch_parameters[i]['current_width'] == 120]

        epoch_indices = list(set(self.size_indices) & set(self.intensity_indices))

        return epoch_indices

    def get_epoch_response(self, roi_set, roi_index):
        return self.get_roi_dict(roi_set)['epoch_response'][roi_index,:,:]

    def get_epoch_response_by_stimulus_type(self, roi_set, roi_index, is_bright=True, is_small=True):
        epoch_indices = self.get_epoch_indices_by_stimulus_type(is_bright, is_small)
        epoch_responses = self.get_epoch_response(roi_set, roi_index)

        return epoch_responses[epoch_indices,:]

    def get_epoch_average_response(self, roi_set, roi_index):
        '''
        takes average across all stimulus types!!! Warning!
        '''
        this_roi_epoch_response = self.get_epoch_response(roi_set, roi_index) #go through every roi, every roi has 3 components
        epoch_average = np.mean(this_roi_epoch_response, axis = 0) #mean of this_roi
        return epoch_average

    def get_epoch_average_response_by_stimulus_type(self, roi_set, roi_index, is_bright=True, is_small=True):
        this_roi_average_response = self.get_epoch_response_by_stimulus_type(roi_set, roi_index, is_bright=is_bright, is_small=is_small)
        epoch_average_by_stimulus = np.mean(this_roi_average_response, axis = 0)
        return epoch_average_by_stimulus

    def recover_impulse_stimulus(self, is_bright=True):
        #create some stimulus at this imaging rate, how many frames per second

        #stimulus
        stimulus = np.zeros(self.n_sample) + self.idle_color
        n_sample_pre = int(np.floor(self.pre_duration * self.imaging_rate))
        stimulus[n_sample_pre + 1: n_sample_pre + (int(np.floor(self.stim_duration * self.imaging_rate)) + 1)] = is_bright #1 if bright, 0 if dark

        self.impulse_stimulus = stimulus

        #time at which the images were taken.
        #time_vector = self.get_roi_dict((self.get_roi_set_names()[0])['time_vector'])
        #stim_time_diff = (time_vector[1] - time_vector[0]) / 100 # diff = 1/stimulus_rate
        #length of stim_time_vector must match length of stimulus
        #x = self.stim_time_diff * len(stimulus) + self.stim_time_diff
       # stim_time_vector = np.arange (self.stim_time_diff, x, self.stim_time_diff)

        #sanity check to test that lengths match
        assert (len(stimulus) == len(self.stim_time_vector))

        plt.plot(self.stim_time_vector, stimulus)
        return
    #does NOT matter !!!!! TAKEN ARE OF

    def sec_to_n_frames(self, seconds):
        return int(np.floor(self.imaging_data.sample_rate * seconds))

    def compute_temporal_filter(self, filter_len, roi_set, roi_index, is_bright=True, is_small=True, method = utils.getLinearFilterByFFT):
        '''
        filter_len is defined in seconds
        '''
        #filter_len = int(len(self.impulse_stimulus) / 1.5) #what is 1.5??

        n_filter_frames = self.sec_to_n_frames(filter_len)

        #time points for filter
        filter_time = -np.flip(self.stim_time_vector[0:n_filter_frames])

        # Get the filter!
        filt = method(self.impulse_stimulus, self.get_epoch_average_response_by_stimulus_type(roi_set, roi_index, is_bright, is_small), n_filter_frames)
        filt = np.flip(filt)

        plt.plot(filter_time,filt)
        return filt


#%%
   # def plot_peak_temporal_receptive_field(self, get_roi_set_names, self.n_rois, fn=None):

       # assert get_roi_set_nnames in ??????.keys()
       # assert roi_number in ???????

       # fig = plt.figure()
       # ax_f = fig.add_subplot(2, self.n_rois, self.n_rois+roi_index+1) #add an axis object to the figure
       # ax_f.plot(filter_time, filter, color = ImagingData.colors[roi_index])
        #ax.set_xlabel ("time [s]")
       # epoch_error = epoch_std/math.squrt(self.n_epochs)

        #ax.fill_between(self.time_vector, epoch_average - epoch_error, epoch_average + epoch_error, alpha = 0.5, color = ImagingData.colors[roi_index])

        #Plot individual traces from current_epoch_response matrix vs. current_time vector
        #if plot_individual_traces:
        #    ax.plot(self.time_vector, some_roi_dict['epoch_response'][roi_index,epoch_indices,:].T, color = 'k', alpha = 0.1)
       # plot_individual_traces = False


       # for roi_index in range(number_of_rois): #for-loop over all the rois in this roi set
           # ax = fig.add_subplot(1, number_of_rois, roi_index+1) #add an axis object to the figure

#%%
'''

        fig = plt.figure() #make a matplotlib figure that we'll add axes to as we go

        for roi_index in range(self.n_rois): #for-loop over all the rois in this roi set

            ax = fig.add_subplot(1, self.n_rois, roi_index+1) #add an axis object to the figure

            #Plot individual traces from current_epoch_response matrix vs. cu5rrent_time vector
            if plot_individual_traces:
                ax.plot(self.time_vector,  self.epoch_response_matrix[roi_index,epoch_indices,:].T, color = 'k', alpha = 0.1)

        """
        #TODO: calculate the epoch average responses for each roi
        #   call this epoch_average. It should be 1-D and have length = (number of time points)
        #   hint: use np.mean()
        epoch_average = ...
        """
      #  this_roi = epoch_response_matrix[roi_index,epoch_indices,:] #go through every roi, every roi has 3 components
       # epoch_average = np.mean(this_roi, axis = 0) #mean of every roi

        #Plot time vector vs. epoch_average
        #   here we're using the same colors as the rois so the traces will be color coordinated with the roi map
        ax.plot(self.time_vector, self.epoch_average, color = self.imaging_data.colors[roi_index])

        """
        #TODO: calculate the standard-error of the mean (st-dev/sqrt(n)) as a function of time, call this epoch_error
        #   hint: use np.std() to calculate the st-dev first, then compute the standard error from there
        epoch_std = ...
        epoch_error = ...
        """
        for roi_index in range (self.n_rois):
            this_roi = self.epoch_response_matrix[roi_index,:,:] #go through every roi, every roi has 3 components
            self.epoch_average = np.mean(this_roi, axis = 0) #mean of every roi

            #this_roi = epoch_response_matrix[roi_index,:,:] #every roi
            epoch_std = np.std(this_roi,0) #std of every roi
            epoch_error = (epoch_std/math.sqrt(self.n_epochs))

            #plot the error snake around top of the epoch average
            ax.fill_between(self.time_vector, self.epoch_average - epoch_error, self.epoch_average + epoch_error, alpha = 0.5, color = self.imaging_data.colors[roi_index])

        fig2 = plt.figure()

        for roi_index in range(self.n_rois):
            this_roi_sb = self.epoch_response_matrix[roi_index,small_bright_indices,:] #go through every roi, every roi has 3 components
            epoch_average_sb = np.mean(this_roi_sb, 0) #mean of every roi
            this_roi_sd = self.epoch_response_matrix[roi_index,small_dark_indices,:] #go through every roi, every roi has 3 components
            epoch_average_sb = np.mean(this_roi_sd, 0) #mean of every roi
            this_roi_bb = self.epoch_response_matrix[roi_index,big_bright_indices,:] #go through every roi, every roi has 3 components
            epoch_average_sb = np.mean(this_roi_bb, 0) #mean of every roi
            this_roi_bd = self.epoch_response_matrix[roi_index,big_dark_indices,:] #go through every roi, every roi has 3 components
            epoch_average_sb = np.mean(this_roi_bd, 0) #mean of every roi

            ax = fig2.add_subplot(1, self.n_rois, roi_index+1)
            ax.plot(self.time_vector, epoch_average_sb, "r-", self.time_vector, epoch_average_sd, "b-", self.time_vector, epoch_average_bb, "g-", self.time_vector, epoch_average_bd, "y-")
            #plt.setp(ax, alpha = 0.1)

            #ax.plot(time_vector, epoch_average_large, color = "bo", alpha = 0.1)


        """
        #TODO: find a way to tell from the data what the minimum and maximum dF/F value is
        #   assign these to min_y and max_y rather than hard-code in these values
        min_y = ...
        max_y = ...
        """

        min_y = np.min(self.epoch_response_matrix) #min of any of the values from 3 components
        max_y = np.max(self.epoch_response_matrix)


        min_y = -0.5
        max_y = 3
        ax.set_ylim([min_y,max_y]) #sets the y axis limits for the plot

        #Use the generateRoiMap convenience function to make a roi map for this roi set
        self.imaging_data.generateRoiMap(roi_set_name, scale_bar_length=20)


'''
#%%  COMPARE BIG VS. SMALL IMPULSE SPOTS FOR A SINGLE COLUMN

# Pull out and assign the roi dictionary for the 'column' roi, for each small and large series


"""
#TODO: make a for-loop that loops over each roi in the roi set
#       each iteration should: compute the epoch average response for both the small & large spots and plot them against the time_vector
#      hint: use np.mean() to compute epoch averages
#            use ax = fig.add_subplot() to add new axes to the figure, and ax.plot() to plot to those newly created axes

fig2 = plt.figure()

for roi_index in range(self.n_rois):
    this_roi_sb = self.epoch_response_matrix[roi_index,small_bright_indices,:] #go through every roi, every roi has 3 components
    epoch_average_sb = np.mean(this_roi_sb, 0) #mean of every roi
    this_roi_sd = self.epoch_response_matrix[roi_index,small_dark_indices,:] #go through every roi, every roi has 3 components
    epoch_average_sb = np.mean(this_roi_sd, 0) #mean of every roi
    this_roi_bb = self.epoch_response_matrix[roi_index,big_bright_indices,:] #go through every roi, every roi has 3 components
    epoch_average_sb = np.mean(this_roi_bb, 0) #mean of every roi
    this_roi_bd = self.epoch_response_matrix[roi_index,big_dark_indices,:] #go through every roi, every roi has 3 components
    epoch_average_sb = np.mean(this_roi_bd, 0) #mean of every roi

    ax = fig2.add_subplot(1, self.n_rois, roi_index+1)
    ax.plot(self.time_vector, epoch_average_sb, "r-", self.time_vector, epoch_average_sd, "b-", self.time_vector, epoch_average_bb, "g-", self.time_vector, epoch_average_bd, "y-")
    #plt.setp(ax, alpha = 0.1)

    #ax.plot(time_vector, epoch_average_large, color = "bo", alpha = 0.1)




#Use the generateRoiMap convenience function to make a roi map for this roi set
ImagingData.generateRoiMap('column', scale_bar_length=20)

# %% POPULATION ANALYSIS: COLLECT POPULATION AVERAGE RESPONSE TRACES IN EACH LAYER, FOR EXPT AND CONTROL FLIES





#%% filter

#our frame rate should be similar to imaging rate
#difference of time points, average, 1/that for rate
imaging_rate = 1/(np.mean(np.diff(time_vector)))

#create some stimulus at this imaging rate, how many frames per second
#how long the entire duration
duration = 1.52 #seconds

#how many samples we will create
n_sample = imaging_rate * duration

#stimulus
stimulus = np.zeros(n_sample) + 0.5
n_sample_pre = 0.5 * imaging_rate
stimulus[n_sample_pre + 1: n_sample_pre + (0.02 * imaging_rate + 1)] = 1

plt.plot (stimulus)

def getLinearFilterByFFT(stimulus, response, filter_len):
    filter_fft = np.fft.fft(response) * np.conj(np.fft.fft(stimulus))
    filt = np.real(np.fft.ifft(filter_fft))[0:filter_len]
    return filt


"""
