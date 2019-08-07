
from visanalysis.imaging_data import BrukerData
from visanalysis.volume_tools import take_every_other_frame

file_name = '2019-07-23'
z_index = 0 #index from 0
series_number = 1 #index from 1

ImagingData = BrukerData.ImagingDataObject(file_name, series_number, load_rois=False, z_index=z_index)
ImagingData.loadImageSeries()
if ImagingData.metadata['bidirectionalZ']:
    ImagingData = take_every_other_frame(ImagingData, take_downward=True)

# %% Choose Rois
from visanalysis import region
MRS = region.MultiROISelector(ImagingData, roiType = 'freehand', roiRadius = 2)
