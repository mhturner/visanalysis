from visanalysis.imaging_data import BrukerData

file_name = '2019-07-30'
z_index = 0 #index from 0
series_number = 2 #index from 1

ImagingData = BrukerData.ImagingDataObject(file_name, series_number, load_rois=False, z_index=z_index)
ImagingData.loadImageSeries()

# %% Choose Rois
from visanalysis import region
MRS = region.MultiROISelector(ImagingData, roiType = 'freehand', roiRadius = 2)
