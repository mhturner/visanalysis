#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:44:51 2018

WARNING: this script will delete image series data so make sure you know what
it's doing and back your raw data up on the server before using this

@author: mhturner
"""
import os
import fnmatch
import re
from tifffile import imsave

from visanalysis import imaging_data
from visanalysis.volume_tools import create_bruker_objects_from_zstack
from visanalysis.volume_tools import take_every_other_frame

#dates = ['20190710','20190712','20190715','20190716','20190722','20190723']
#            x           x          x       x   ?? (epochs not matched)  x
dates = ['20190730']
take_downward=True


for date in dates:
    file_directory = '/home/clandinin/Desktop/heather_data/' + date

    # get files of unregistered time series in current working directory
    file_names = sorted(fnmatch.filter(os.listdir(file_directory),'TSeries-*[0-9].tif'))

    for file_name in file_names:
        print(file_name)
        series_number = int(re.split('-|\.',file_name)[-2])
        tmp_str = re.split('-', file_name)[1]
        fn = ''.join([tmp_str[0:4],'-',tmp_str[4:6],'-',tmp_str[6:8]])
        ImagingData = create_bruker_objects_from_zstack(fn, series_number, load_rois = False)

        if not type(ImagingData) == list:
            ImagingData = [ImagingData]
        for idata in ImagingData:
            idata.image_series_name = 'TSeries-' + fn.replace('-','') + '-' + ('00' + str(series_number))[-3:]
            idata.loadImageSeries()
            print ('bidirectionalZ = ' + str(idata.metadata['bidirectionalZ']))
            if idata.metadata['bidirectionalZ'] and take_downward is not None:
                down_or_up = "downward" if take_downward else "upward"
                print ('Taking ' + down_or_up + ' frames only.')
                idata = take_every_other_frame(idata, take_downward=take_downward)
            idata.registerStack()
            if len(ImagingData) > 1: #multiple planes
                save_path = os.path.join(file_directory, file_name.split('.')[0] + '_z' + str(idata.z_index) + '_reg' + '.tif')
            else:
                save_path = os.path.join(file_directory, file_name.split('.')[0] + '_reg' + '.tif')
            imsave(save_path, idata.registered_series)
            print('Saved: ' + save_path)

        #os.remove(os.path.join(file_directory, file_name))
