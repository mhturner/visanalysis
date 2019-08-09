from visanalysis import imaging_data
from visanalysis.volume_tools import get_n_z_layers

def create_bruker_objects_from_zstack(file_name, series_number, load_rois = True, z_index=None):
    if z_index is None: # If z_index is not provided, check whether it is a volume and split up the z stacks.
        n_zstacks = get_n_z_layers(file_name, series_number)
        if n_zstacks > 1: #i.e. multiple z stacks
            return [imaging_data.BrukerData.ImagingDataObject(file_name, series_number, load_rois, z_index=i) for i in range(n_zstacks)]
    return imaging_data.BrukerData.ImagingDataObject(file_name, series_number, load_rois, z_index=z_index)
