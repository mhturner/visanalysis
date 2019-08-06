
from visanalysis.imaging_data import BrukerData
from visanalysis.utilities.take_every_other_frame import take_every_other_frame

<<<<<<< HEAD
file_name = '2019-07-23'
z_index = 0 #index from 0
series_number = 1 #index from 1

ImagingData = BrukerData.ImagingDataObject(file_name, series_number, load_rois=False, z_index=z_index)
ImagingData.loadImageSeries()
if ImagingData.metadata['bidirectionalZ']:
    ImagingData = take_every_other_frame(ImagingData, take_downward=True)
=======
file_name = '2019-07-16'
z_index = 0 #index from 0
series_number = 7 #index from 1

ImagingData = BrukerData.ImagingDataObject(file_name, series_number, load_rois=False, z_index=z_index)
ImagingData.loadImageSeries()
ImagingData = take_every_other_frame(ImagingData, take_downward=True)
>>>>>>> ad2161d13d0ca53081f5047dc022f3b3a9ba17ce

# %% Choose Rois
from visanalysis import region
MRS = region.MultiROISelector(ImagingData, roiType = 'freehand', roiRadius = 2)
