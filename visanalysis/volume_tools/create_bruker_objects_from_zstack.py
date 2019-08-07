import os
from visanalysis import imaging_data
import xml.etree.ElementTree as ET

def create_bruker_objects_from_zstack(file_name, series_number, load_rois = True, z_index=None):
    super = imaging_data.ImagingData.ImagingDataObject(file_name, series_number)
    image_series_name = 'TSeries-' + file_name.replace('-','') + '-' + ('00' + str(series_number))[-3:]
    if z_index is None: # If z_index is not provided, check whether it is a volume and split up the z stacks.
        metaData = ET.parse(os.path.join(super.image_data_directory, image_series_name) + '.xml')
        root = metaData.getroot()

        n_zstacks = len(root.findall('Sequence')[0].findall('Frame'))
        if n_zstacks > 1: #i.e. multiple z stacks
            return [imaging_data.BrukerData.ImagingDataObject(file_name, series_number, load_rois, z_index=i) for i in range(n_zstacks)]
    return imaging_data.BrukerData.ImagingDataObject(file_name, series_number, load_rois, z_index=z_index)
