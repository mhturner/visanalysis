import os
from visanalysis import imaging_data
import h5py
import xml.etree.ElementTree as ET

def get_n_z_layers(file_name, series_number):
    super = imaging_data.ImagingData.ImagingDataObject(file_name, series_number)
    with h5py.File(os.path.join(super.flystim_data_directory, super.file_name) + '.hdf5','r') as experiment_file:
        roi_group = experiment_file['/epoch_runs'].get(str(series_number)).get('rois')
        n_z_tentative = len([x for x in roi_group.keys() if x[0] == 'z' and x[1:].isdigit()])
        n_roi_groups = len(roi_group.keys())
    if n_z_tentative > 0:
        return n_z_tentative
    elif n_roi_groups > 0:
        return 1

    image_series_name = 'TSeries-' + file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
    metaData = ET.parse(os.path.join(super.image_data_directory, image_series_name) + '.xml')
    root = metaData.getroot()
    if len(root.findall('Sequence')) > 1: # Multiple z planes
        n_z_layers = len(root.findall('Sequence')[0].findall('Frame'))
    else:
        n_z_layers = 0

    return n_z_layers
