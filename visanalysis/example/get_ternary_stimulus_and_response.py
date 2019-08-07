from visanalysis.imaging_data import BrukerData
from visanalysis.analysis import TernaryNoiseAnalysis

file_name = '2019-06-19'
series_number = 5 #index from 1
z_index = None #index from 0

tna = TernaryNoiseAnalysis(file_name, series_number, z_index=z_index)
tna.recover_ternary_noise_stimulus()

tna.get_roi_set_names()
tna.get_n_roi_in_roi_set('column')

roi_set_name = 'column' #column, etc.

responses = tna.imaging_data.roi[roi_set_name]['roi_response']
stimulus = tna.ternary_noise

responses.shape # 4 ROIs, 17xxx time points
stimulus.shape  # n_phi, n_theta, 17xxx time points
