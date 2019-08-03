import numpy as np
from visanalysis.imaging_data.BrukerData import ImagingDataObject


def take_every_other_frame(imaging_data, take_downward=True):
    '''
    MUST be run before registration, AFTER loadImageSeries()
    '''
    is_down = imaging_data.metadata['Zdepths'][1] - imaging_data.metadata['Zdepths'][1] > 0 # whether the first z scan is downwards

    take_even = take_downward == is_down

    start_idx = 0 if take_even else 1
    old_stack_times = imaging_data.response_timing['stack_times']
    new_stack_times = old_stack_times[start_idx::2]
    old_frame_times = imaging_data.response_timing['frame_times']
    new_frame_times = old_frame_times[start_idx::2]
    new_sample_period = np.mean(np.diff(new_stack_times))
    imaging_data.response_timing['stack_times'] = new_stack_times
    imaging_data.response_timing['frame_times'] = new_frame_times
    imaging_data.response_timing['sample_period'] = new_sample_period

    imaging_data.raw_series = imaging_data.raw_series[start_idx::2,]
    imaging_data.current_series = imaging_data.current_series[start_idx::2,]
    imaging_data.roi_image = np.squeeze(np.mean(imaging_data.current_series, axis = 0))
    return imaging_data
