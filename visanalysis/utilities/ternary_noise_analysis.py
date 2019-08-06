############################################
# Code for analysing ternary noise imaging experiment data
# Authors: Heather Chang and Minseung Choi
# 2019 July 23
##############################################

from visanalysis import imaging_data
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

def getLinearFilterByFFT(stimulus, response, filter_len):
    filter_fft = np.fft.fft(response) * np.conj(np.fft.fft(stimulus))
    filt = np.real(np.fft.ifft(filter_fft))[0:filter_len]
    return filt

class TernaryNoiseAnalysis():
    def __init__(self, fn='2019-06-19', series_number=5, z_index=None):
        self.fn = fn
        self.series_number = series_number

        self.imaging_data = imaging_data.BrukerData.ImagingDataObject(self.fn, self.series_number, load_rois = True, z_index=z_index)

        self.seconds_per_unit_time = np.mean(np.diff(self.imaging_data.response_timing['stack_times']))

        self.degrees_per_unit_phi = self.imaging_data.epoch_parameters[0]['phi_period']
        self.degrees_per_unit_theta = self.imaging_data.epoch_parameters[0]['theta_period']


        self.num_phi = int(180 / self.degrees_per_unit_phi)
        self.num_theta = int(360 /self.degrees_per_unit_theta)


        self.ternary_noise = None
        self.strf = {}
        return

    def get_roi_set_names(self):
        return [*self.imaging_data.roi]

    def get_n_roi_in_roi_set(self, roi_set):
        return self.imaging_data.roi[roi_set]['roi_response'].shape[0]

    def recover_ternary_noise_stimulus(self, max_phi=128, max_theta=256):
        """
        Recovers the ternary noise stimulus shown to the fly using seed and other parameters.
        Returns:
           - 3-dimensional numpy array: (phi, theta, time points)
        """

        ### Generate stimulus based on seed
        output_shape = (max_phi, max_theta)

        num_frames = next(iter(self.imaging_data.roi.values()))['roi_response'].shape[1]

        # Get noise_frames at each imaging acquisition time point
        noise_frames = np.empty(shape=(self.num_phi, self.num_theta, num_frames), dtype=float)
        noise_frames[:] = np.nan

        for stack_ind, stack_time in enumerate(self.imaging_data.response_timing['stack_times']): #sec
            start_inds = np.where(stack_time>self.imaging_data.stimulus_timing['stimulus_start_times'])[0]
            stop_inds = np.where(stack_time<self.imaging_data.stimulus_timing['stimulus_end_times'])[0]
            if start_inds.size==0 or stop_inds.size==0: #before or after first/last epoch
                noise_frames[:,:,stack_ind] = self.imaging_data.run_parameters['idle_color']
            else:
                start_ind = start_inds[-1]
                stop_ind = stop_inds[0]
                if start_ind == stop_ind: #inside an epoch. Get noise grid
                    epoch_ind = int(start_ind)
                    start_seed = self.imaging_data.epoch_parameters[epoch_ind]['start_seed']
                    rand_min = self.imaging_data.epoch_parameters[epoch_ind]['rand_min']
                    rand_max = self.imaging_data.epoch_parameters[epoch_ind]['rand_max']

                    update_rate = self.imaging_data.epoch_parameters[epoch_ind]['update_rate'] #Hz
                    current_time = (stack_time-self.imaging_data.stimulus_timing['stimulus_start_times'][epoch_ind]) #sec

                    seed = int(round(start_seed + current_time*update_rate))
                    np.random.seed(seed)
                    face_colors = np.random.choice([rand_min, (rand_min + rand_max)/2 , rand_max], size=output_shape) # called rand_values in flystim.distribution.Ternary
                    #generates grid using max phi/theta, shader only takes the values that it needs to make the grid
                    face_colors = face_colors[0:self.num_phi,0:self.num_theta]

                    noise_frames[:,:,stack_ind] = face_colors

                else: #between epochs, put up idle color
                    noise_frames[:,:,stack_ind] = self.imaging_data.run_parameters['idle_color']

            self.ternary_noise = noise_frames
        return

    def plot_100th_frame_of_ternary_noise(self):
        plt.imshow(self.ternary_noise[:,:,100])
        return

    def get_roi_set_names(self):
        return self.imaging_data.roi.keys()

    def compute_strf(self, roi_set='column', roi_number=0, filter_len=20, method=getLinearFilterByFFT):
        assert roi_set in self.get_roi_set_names()
        assert self.ternary_noise is not None

        pre_time = self.imaging_data.run_parameters['pre_time'] * 1e3 #msec
        tail_time = self.imaging_data.run_parameters['tail_time'] * 1e3 #msec
        stim_time = self.imaging_data.run_parameters['stim_time'] * 1e3 #msec
        raw_response = self.imaging_data.roi.get(roi_set)['roi_response'].copy()

        n_rois = raw_response.shape[0]

        #do basic baseline first based on first pre_time
        pre_end = self.imaging_data.stimulus_timing['stimulus_start_times'][0] #sec
        pre_start = pre_end - pre_time / 1e3 #sec
        pre_inds = np.where(np.logical_and(\
                    self.imaging_data.response_timing['stack_times'] < \
                    pre_end,self.imaging_data.response_timing['stack_times'] >= pre_start))[0]
        baseline = np.mean(raw_response[:,pre_inds],axis = 1).reshape((n_rois,-1))
        current_response = (raw_response-baseline) / baseline #calculate dF/F

        #Recalcuate baseline for points within epoch based on each epochs pre-time
        #   Accounts for some amount of drift over long recordings (e.g. bleaching)
        for eInd,stimulus_start in enumerate(self.imaging_data.stimulus_timing['stimulus_start_times']):
            epoch_start = stimulus_start - pre_time
            epoch_end = epoch_start + pre_time + stim_time + tail_time
            pre_inds = np.where(np.logical_and(self.imaging_data.response_timing['stack_times'] < stimulus_start,
                                   self.imaging_data.response_timing['stack_times'] >= epoch_start))[0]
            baseline = np.mean(raw_response[:,pre_inds], axis = 1).reshape((n_rois,-1))

            epoch_inds = np.where(np.logical_and(self.imaging_data.response_timing['stack_times'] < epoch_end,
                                   self.imaging_data.response_timing['stack_times'] >= epoch_start))[0]
            current_response[:,epoch_inds] = (raw_response[:,epoch_inds] - baseline) / baseline #calculate dF/F

        ##### Now goal is to come up with SPATIAL FIR (Finite Impulse Response) linear filter

        response = current_response[roi_number,:]

        strf = np.empty((self.num_phi, self.num_theta, filter_len)) # spatiotemporal RF
        for phi in tqdm(range(self.num_phi)):
            for theta in range(self.num_theta):
                stimulus = self.ternary_noise[phi,theta,:]
                strf[phi,theta,:] = getLinearFilterByFFT(stimulus, response, filter_len)

        #if dictionary for roi_set not existed then it will create an empty dictionary.
        if roi_set not in self.strf.keys():
            self.strf[roi_set] = {}
        # a new key roi_number, value strf pair will be added
        self.strf[roi_set][roi_number] = strf

        return

    def plot_avg_spatial_rf(self, roi_set='column', roi_number=0, start_idx=0, end_idx=1, fn=None):
        mean_rf = np.mean(self.strf[roi_set][roi_number][:,:,start_idx:end_idx],axis=2)

        fig = plt.figure(figsize=(10,8))
        plt.imshow(mean_rf, cmap='inferno', extent=[0,360,180,0])
        plt.xlabel("degrees")
        plt.ylabel("degrees")
        plt.colorbar()

        if fn is not None:
            fig.savefig(fn)
        return

    def plot_spatiotemporal_receptive_field(self, roi_set='column', roi_number=0, fn=None):
        print('hello')

    def plot_peak_spatial_receptive_field(self, roi_set='column', roi_number=0, fn=None):

        assert roi_set in self.strf.keys()
        assert roi_number in self.strf[roi_set]

        #find peak
        strf = self.strf[roi_set][roi_number]
        peak_idx = np.unravel_index(strf.argmax(), strf.shape)
        peak_time = peak_idx[2]

        # get average peak from 3 values-ish around peak_time
        avg_start = peak_time-1 if peak_time > 0 else peak_time
        avg_end = peak_time + 2 if peak_time < self.strf[roi_set][roi_number].shape[2]-3 else self.strf[roi_set][roi_number].shape[2] -1

        print ("averaged over " + str(avg_end - avg_start) + " frames of strf.")

        mean_rf = np.mean(self.strf[roi_set][roi_number][:,:,avg_start:avg_end],axis=2)

        fig = plt.figure(figsize=(10,8))
        plt.imshow(mean_rf, cmap='inferno', extent=[0,360,180,0])
        plt.xlabel("degrees")
        plt.ylabel("degrees")
        plt.colorbar()

        if fn is not None:
            fig.savefig(fn)
        return

    def plot_peak_temporal_receptive_field(self, roi_set='column', roi_number=0, fn=None):

        assert roi_set in self.strf.keys()
        assert roi_number in self.strf[roi_set]

        #find peak
        strf = self.strf[roi_set][roi_number]
        peak_idx = np.unravel_index(strf.argmax(), strf.shape)
        peak_y = peak_idx[0]
        peak_x = peak_idx[1]

        # average 9 pixels including and around the peak
        mean_rf = np.flip(np.mean(self.strf[roi_set][roi_number][peak_y-1:peak_y+2, peak_x-1:peak_x+2, :],axis=(0,1)),axis=0)

        filter_time = -np.flip(np.arange(0, len(mean_rf)) * self.seconds_per_unit_time, axis=0)

        fig = plt.figure(figsize=(10,6))
        plt.plot(filter_time, mean_rf)
        plt.xlabel("seconds")
        plt.ylabel("a.u.")

        if fn is not None:
            fig.savefig(fn)
        return
