import os
from visanalysis import imaging_data
import xml.etree.ElementTree as ET

def get_n_z_layers(file_name, series_number):
    super = imaging_data.ImagingData.ImagingDataObject(file_name, series_number)
    image_series_name = 'TSeries-' + file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
    metaData = ET.parse(os.path.join(super.image_data_directory, image_series_name) + '.xml')
    root = metaData.getroot()
    n_z_layers = len(root.findall('Sequence')[0].findall('Frame'))

    return n_z_layers
