import numpy as np

def take_every_other_frame(imaging_data, take_downward=True):

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

    ''' MOVED INTO BrukerData.ImagingDataObject.loadImageSeries
    if imaging_data.raw_series is not None:
        imaging_data.raw_series = imaging_data.raw_series[start_idx::2,]
    # If registered_series is not None, then current might be already cut in half...
    if imaging_data.registered_series is None and imaging_data.current_series is not None:
        imaging_data.current_series = imaging_data.current_series[start_idx::2,]
        imaging_data.roi_image = np.squeeze(np.mean(imaging_data.current_series, axis = 0))
    '''

    #for roi_set in imaging_data.roi.keys():
    #    imaging_data.roi[roi_set]['time_vector'], imaging_data.roi[roi_set]['epoch_response'] = imaging_data.getEpochResponseMatrix(response_trace = imaging_data.roi[roi_set]['roi_response'])

    return imaging_data