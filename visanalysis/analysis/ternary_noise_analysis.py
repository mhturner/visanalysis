############################################
# Code for analysing ternary noise imaging experiment data
# Authors: Heather Chang and Minseung Choi
# 2019 July 23
##############################################

from visanalysis import imaging_data
from visanalysis.analysis import utils

import numpy as np
from tqdm import tqdm
from tifffile import imsave
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

class TernaryNoiseAnalysis():
    def __init__(self, fn='2019-06-19', series_number=5, z_index=None, upsample_rate=500, upsample_method='bins'):
        self.fn = fn
        self.series_number = series_number

        self.imaging_data = imaging_data.BrukerData.ImagingDataObject(self.fn, self.series_number, load_rois = True, z_index=z_index, upsample_rate=upsample_rate, upsample_method=upsample_method)

        #self.seconds_per_unit_time = np.mean(np.diff(self.imaging_data.response_timing['stack_times']))
        self.seconds_per_unit_time = 1/self.imaging_data.upsample_rate if self.imaging_data.upsample_rate is not None else self.imaging_data.response_timing['sample_period']

        self.degrees_per_unit_phi = self.imaging_data.epoch_parameters[0]['phi_period']
        self.degrees_per_unit_theta = self.imaging_data.epoch_parameters[0]['theta_period']


        self.num_phi = int(180 / self.degrees_per_unit_phi)
        self.num_theta = int(360 /self.degrees_per_unit_theta)


        self.ternary_noise = None
        self.strf = {}
        return

    def save_stimulus_tiff(self, save_path):
        if self.ternary_noise is None:
            self.recover_ternary_noise_stimulus()
        imsave(save_path, np.moveaxis((self.ternary_noise*255).astype(np.int16), 2, 0))

    def sec_to_n_frames(self, seconds):
        sample_rate = 1/self.seconds_per_unit_time
        return int(np.floor(sample_rate * seconds))

    def n_frames_to_sec(self, n_frames):
        sample_rate = 1/self.seconds_per_unit_time
        return n_frames / sample_rate

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

    def compute_strf(self, filter_len, roi_set='column', roi_number=0, method=utils.getLinearFilterByFFT):
        assert roi_set in self.get_roi_set_names()
        assert self.ternary_noise is not None

        n_filter_frames = self.sec_to_n_frames(filter_len)

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
        baseline = np.nanmean(raw_response[:,pre_inds],axis = 1).reshape((n_rois,-1))
        current_response = (raw_response-baseline) / baseline #calculate dF/F

        #Recalcuate baseline for points within epoch based on each epochs pre-time
        #   Accounts for some amount of drift over long recordings (e.g. bleaching)
        for eInd,stimulus_start in enumerate(self.imaging_data.stimulus_timing['stimulus_start_times']):
            epoch_start = stimulus_start - pre_time
            epoch_end = epoch_start + pre_time + stim_time + tail_time
            pre_inds = np.where(np.logical_and(self.imaging_data.response_timing['stack_times'] < stimulus_start,
                                   self.imaging_data.response_timing['stack_times'] >= epoch_start))[0]
            baseline = np.nanmean(raw_response[:,pre_inds], axis = 1).reshape((n_rois,-1))

            epoch_inds = np.where(np.logical_and(self.imaging_data.response_timing['stack_times'] < epoch_end,
                                   self.imaging_data.response_timing['stack_times'] >= epoch_start))[0]
            current_response[:,epoch_inds] = (raw_response[:,epoch_inds] - baseline) / baseline #calculate dF/F

        ##### Now goal is to come up with SPATIAL FIR (Finite Impulse Response) linear filter

        response = current_response[roi_number,:]

        strf = np.empty((self.num_phi, self.num_theta, n_filter_frames)) # spatiotemporal RF
        for phi in tqdm(range(self.num_phi)):
            for theta in range(self.num_theta):
                stimulus = self.ternary_noise[phi,theta,:]
                trf = np.flip(method(stimulus, response, n_filter_frames))
                baseline = np.mean(trf[0:int(len(trf)/4)]) ## 190812 length div by 4 is arbitrary and only works when the first quarter is too far back in history
                trf = trf-baseline
                strf[phi,theta,:] = trf

        #if dictionary for roi_set not existed then it will create an empty dictionary.
        if roi_set not in self.strf.keys():
            self.strf[roi_set] = {}
        # a new key roi_number, value strf pair will be added
        self.strf[roi_set][roi_number] = strf

        filter_time = -np.flip(np.arange(0, strf.shape[2]) * self.seconds_per_unit_time, axis=0)

        return strf, filter_time

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

    def find_peak_in_rf(self, roi_set='column', roi_number=0):
        strf = self.strf[roi_set][roi_number]
        sigma = 2 #arbitrary
        smoothed_strf = gaussian_filter(strf, sigma)

        #peak_idx = np.unravel_index(np.argmax(smoothed_strf), strf.shape)
        #peak_phi = peak_idx[0]
        #peak_theta = peak_idx[1]
        #peak_time = peak_idx[2]
        #return peak_phi, peak_theta, peak_time

        n_max_clustered = 6 #arbitrary
        n_min_clustered = 5
        distance_thresh = 2 #arbitrary

        peak_1d_idx = np.argsort(-smoothed_strf, axis=None)[:n_max_clustered]

        peak_idx_list = [np.unravel_index(x, strf.shape) for x in peak_1d_idx]

        peak_location_list = [np.array([x[0],x[1]]) for x in peak_idx_list]

        for i in range(len(peak_location_list)):
            peak_loc = peak_location_list[i]
            connected_list = [peak_location_list[0]]
            connected_list_temp = [peak_location_list[0]]

            for j in range(len(peak_location_list)):
                if i == j:
                    continue
                loc = peak_location_list[j]
                connected_list = np.array(connected_list_temp)
                for connected_loc in connected_list:
                    connected = np.all(np.abs(connected_loc - loc) <= distance_thresh)
                    if connected:
                        connected_list_temp.append(loc)
                        continue
            if len(connected_list) >= n_min_clustered:
                print ("Peak cluster found!")
                peak_phi = peak_idx_list[0][0]
                peak_theta = peak_idx_list[0][1]
                peak_time = peak_idx_list[0][2]
                return peak_phi, peak_theta, peak_time

        print( "no peak cluster found")
        return None


    def plot_peak_strf(self, roi_set='column', roi_number=0, spatial_radius=15, fn=None):
        centered_strf = self.get_centered_strf(roi_set=roi_set, roi_number=roi_number, spatial_radius=spatial_radius)
        if centered_strf is None:
            return

        sym_strf = (centered_strf + np.swapaxes(centered_strf, 0, 1))/2
        peak_strf = sym_strf[int(sym_strf.shape[0]/2), :, :]

        max_time = self.seconds_per_unit_time * peak_strf.shape[1]
        max_deg = self.degrees_per_unit_theta * peak_strf.shape[0]

        fig = plt.figure(figsize=(12,4))
        plt.imshow(peak_strf, cmap='inferno', extent=[-max_time,0,max_deg,0], aspect='auto')
        plt.xlabel("Time [sec]")
        plt.ylabel("Degrees")
        plt.colorbar(orientation="horizontal", pad=0.2)

        if fn is not None:
            fig.savefig(fn)
        return

    def get_centered_strf(self, roi_set='column', roi_number=0, spatial_radius=15):
        #find peak
        peak = self.find_peak_in_rf(roi_set, roi_number)
        if peak is None:
            return

        peak_phi = peak[0]
        peak_theta = peak[1]

        phi_radius_pixels = int(np.ceil(spatial_radius/self.degrees_per_unit_phi))
        theta_radius_pixels = int(np.ceil(spatial_radius/self.degrees_per_unit_theta))

        # average 9 pixels including and around the peak
        return self.strf[roi_set][roi_number][peak_phi-phi_radius_pixels:peak_phi+phi_radius_pixels, peak_theta-theta_radius_pixels:peak_theta+theta_radius_pixels, :]


    def get_centered_spatial_rf(self, roi_set='column', roi_number=0, spatial_radius=15):

        assert roi_set in self.strf.keys()
        assert roi_number in self.strf[roi_set]

        centered_strf = self.get_centered_strf(roi_set=roi_set, roi_number=roi_number, spatial_radius=spatial_radius)

        if centered_strf is None:
            return

        peak_time = self.find_peak_in_rf(roi_set, roi_number)[2]

        # get average peak from 3 values-ish around peak_time
        avg_start = peak_time-1 if peak_time > 0 else peak_time
        avg_end = peak_time + 2 if peak_time < centered_strf.shape[2]-3 else centered_strf.shape[2] -1

        print ("Peak time at " + str(self.n_frames_to_sec(self.strf[roi_set][roi_number].shape[2] - peak_time - 1)) + " seconds.")
        print ("averaged over " + str(self.n_frames_to_sec(avg_end - avg_start)) + " frames of strf.")

        spatial_rf = np.mean(centered_strf[:,:,avg_start:avg_end],axis=2)

        return spatial_rf

    def plot_centered_spatial_rf(self, roi_set='column', roi_number=0, spatial_radius=15, fn=None):

        mean_rf = self.get_centered_spatial_rf(roi_set=roi_set, roi_number=roi_number, spatial_radius=spatial_radius)

        if mean_rf is None:
            return

        n_phi = mean_rf.shape[0]
        n_theta = mean_rf.shape[1]
        deg_phi = self.degrees_per_unit_phi * n_phi
        deg_theta = self.degrees_per_unit_theta * n_theta

        fig = plt.figure(figsize=(10,5))
        plt.imshow(mean_rf, cmap='inferno', extent=[0,deg_theta,deg_phi,0])
        plt.xlabel("degrees")
        plt.ylabel("degrees")
        plt.colorbar()

        if fn is not None:
            fig.savefig(fn)
        return


    def plot_spatial_rf(self, roi_set='column', roi_number=0, fn=None):

        assert roi_set in self.strf.keys()
        assert roi_number in self.strf[roi_set]

        #find peak
        peak = self.find_peak_in_rf(roi_set, roi_number)
        if peak is None:
            return
        peak_time = peak[2]

        # get average peak from 3 values-ish around peak_time
        avg_start = peak_time-1 if peak_time > 0 else peak_time
        avg_end = peak_time + 2 if peak_time < self.strf[roi_set][roi_number].shape[2]-3 else self.strf[roi_set][roi_number].shape[2] -1

        print ("Peak time at " + str(self.n_frames_to_sec(self.strf[roi_set][roi_number].shape[2] - peak_time - 1)) + " seconds.")
        print ("averaged over " + str(avg_end - avg_start) + " frames of strf.")

        mean_rf = np.mean(self.strf[roi_set][roi_number][:,:,avg_start:avg_end],axis=2)

        fig = plt.figure(figsize=(10,5))
        plt.imshow(mean_rf, cmap='inferno', extent=[0,360,180,0])
        plt.xlabel("degrees")
        plt.ylabel("degrees")
        plt.colorbar()

        if fn is not None:
            fig.savefig(fn)
        return

    def get_temporal_rf(self, roi_set='column', roi_number=0, spatial_radius=3):

        assert roi_set in self.strf.keys()
        assert roi_number in self.strf[roi_set]

        #find peak
        peak = self.find_peak_in_rf(roi_set, roi_number)
        if peak is None:
            return
        peak_phi = peak[0]
        peak_theta = peak[1]

        # average 9 pixels including and around the peak
        centered_strf = self.get_centered_strf(roi_set, roi_number, spatial_radius=spatial_radius)
        mean_rf = np.mean(centered_strf,axis=(0,1))

        filter_time = -np.flip(np.arange(0, len(mean_rf)) * self.seconds_per_unit_time, axis=0)

        return mean_rf, filter_time


    def plot_temporal_rf(self, roi_set='column', roi_number=0, spatial_radius=3, fn=None):

        mean_rf, filter_time = self.get_temporal_rf(roi_set=roi_set, roi_number=roi_number, spatial_radius=spatial_radius)

        fig = plt.figure(figsize=(10,6))
        plt.plot(filter_time, mean_rf)
        plt.xlabel("seconds")
        plt.ylabel("a.u.")

        if fn is not None:
            fig.savefig(fn)
        return
