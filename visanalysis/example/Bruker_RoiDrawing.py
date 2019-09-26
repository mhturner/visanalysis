
from visanalysis.imaging_data import BrukerData

file_name = '2019-01-07'
series_number = 19

ImagingData = BrukerData.ImagingDataObject(file_name, series_number, load_rois=False)
ImagingData.loadImageSeries()
# %%
len(ImagingData.stimulus_timing['stimulus_start_times'])
len(ImagingData.epoch_parameters)
# %% Choose Rois
from visanalysis import region
MRS = region.MultiROISelector(ImagingData, roiType = 'freehand', roiRadius = 2)
