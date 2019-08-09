import xml.etree.ElementTree as ET
import numpy as np
import skimage.io as io


from visanalysis import imaging_data









raw_file_name = '/Users/minseung/Google Drive/School/Stanford/Clandinin Lab/Data/liveimaging/Heather/20190712/TSeries-20190712-002.tif'
raw_series = io.imread(raw_file_name)



metaData = ET.parse('/Users/minseung/Google Drive/School/Stanford/Clandinin Lab/Data/liveimaging/Heather/20190712/TSeries-20190712-002.xml')

root = metaData.getroot()


root.getchildren()


root.find('PVStateShard').items()
root.find('PVStateShard').getchildren()[3].getchildren()[0].items() #X-axis
root.find('PVStateShard').getchildren()[3].getchildren()[1].items() #Y-axis


root.findall('Sequence')[0].items()
root.findall('Sequence')[1].items()
root.findall('Sequence')[2].items()
root.findall('Sequence')[3].items()




root.findall('Sequence')[0].getchildren()
root.findall('Sequence')[1].getchildren()
root.findall('Sequence')[2].getchildren()



root.findall('Sequence')[0].getchildren()[1].items()


bool(root.find('Sequence').get('bidirectionalZ'))


root.findall('Sequence')[0].findall('Frame')[0].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].items()


pvss = root.findall('Sequence')[0].findall('Frame')[0].find('PVStateShard')


[x.findall('SubindexedValues')[2].findall('SubindexedValue')[1].get('value') for x in pvss.findall('PVStateValue') if x.get('key') == 'positionCurrent']




[pvsv.findall('SubindexedValues')[2] for pvsv in root.find('PVStateShard').findall('PVStateValue') if ('key', 'positionCurrent') in pvsv.items()][0].findall('SubindexedValue')[1].get('value')




positionCurrent

root.findall('Sequence')[0].findall('Frame')[0].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].get('value')
root.findall('Sequence')[0].findall('Frame')[1].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[0].findall('Frame')[2].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[0].findall('Frame')[3].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[0].findall('Frame')[4].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[0].findall('Frame')[5].find('PVStateShard').findall('PVStateValue')
[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()





root.findall('Sequence')[1].findall('Frame')[0].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[1].findall('Frame')[1].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[1].findall('Frame')[2].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[1].findall('Frame')[3].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[1].findall('Frame')[4].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()
root.findall('Sequence')[1].findall('Frame')[5].find('PVStateShard').findall('PVStateValue')[2].findall('SubindexedValues')[2].findall('SubindexedValue')[1].items()





root.findall('Sequence')[0].findall('Frame')[1].items()

root.findall('Sequence')[0].findall('Frame')[2].items()

root.findall('Sequence')[0].findall('Frame')[5].items()


root.findall('Sequence')[1].findall('Frame')[0].items()

root.findall('Sequence')[1].findall('Frame')[1].items()

root.findall('Sequence')[1].findall('Frame')[2].items()









is_volume = len(root.findall('Sequence')) > 1


root.findall('Sequence')


tframes = root.findall('Sequence')
tframes[2].items()
tframes[2].getchildren()
root.findall('Sequence')[0].get('bidirectionalZ')

############## Getting time stamps for each z stack
tframes = root.findall('Sequence')
n_tframes = len(tframes)
n_zstacks = len(tframes[0].findall('Frame'))

# Multi-plane, xy time series for each plane
stack_times = [[] for _ in range(n_zstacks)]
for tframe in tframes:
    tz_stack = tframe.findall('Frame')
    assert (len(tz_stack) == n_zstacks)
    for z in range(n_zstacks):
        frTime = tz_stack[z].get('relativeTime')
        assert (frTime is not None)
        stack_times[z].append(float(frTime))
stack_times = np.array(stack_times) #sec
frame_times = stack_times.copy()    #sec
sample_period = np.mean(np.diff(stack_times)) # sec
#######################

########## loadImageSeries
current_series = io.imread('/Users/minseung/Google Drive/School/Stanford/Clandinin Lab/Data/liveimaging/Heather/20190712/TSeries-20190712-002.tif')

current_series.shape

roi_image = np.squeeze(np.mean(current_series, axis = 0))
roi_response = []
roi_mask = []
roi_path = []

roi_image.shape

is_volume = len(roi_image.shape) == 3
############


########## registerStack
reference_time_frame = 1 #sec, first frames to use as reference for registration
reference_frame = [np.where(stack_times[i,:] > reference_time_frame)[0][0] for i in range(n_zstacks)]


reference_image = np.squeeze(np.mean(self.raw_series[0:reference_frame,:,:], axis = 0))
register = CrossCorr()
model = register.fit(self.raw_series, reference=reference_image)

self.registered_series = model.transform(self.raw_series)
if len(self.registered_series.shape) == 3: #xyt
    self.registered_series = self.registered_series.toseries().toarray().transpose(2,0,1) # shape t, y, x
elif len(self.registered_series.shape) == 4: #xyzt
    self.registered_series = self.registered_series.toseries().toarray().transpose(3,0,1,2) # shape t, z, y, x

self.current_series = self.registered_series






############




###############

for child in root.find('Sequence').getchildren():
    print (child.items())



child = next(iter(root.find('Sequence').getchildren()))

child.text

#stack_times.shape
stack_times = stack_times[1:] #trim extra 0 at start
frame_times = stack_times


stack_times = stack_times # sec
frame_times = frame_times # sec
sample_period = np.mean(np.diff(stack_times)) # sec
response_timing = {'stack_times':stack_times, 'frame_times':frame_times, 'sample_period':sample_period }

###############

root = metaData.getroot()

print ([x.get('key') for x in list(root.find('PVStateShard'))])

metadata = {}
for child in list(root.find('PVStateShard')):
    if child.get('value') is None:
        for subchild in list(child):
            new_key = child.get('key') + '_' + subchild.get('index')
            new_value = subchild.get('value')
            metadata[new_key] = new_value

    else:
        new_key = child.get('key')
        new_value = child.get('value')
        metadata[new_key] = new_value


metadata['version'] = root.get('version')
metadata['date'] = root.get('date')
metadata['notes'] = root.get('notes')

#metadata
##############

reference_time_frame = 1 #sec, first frames to use as reference for registration
reference_frame = np.where(response_timing['stack_times'] > reference_time_frame)[0][0]

len(response_timing['stack_times'])


reference_image = np.squeeze(np.mean(self.raw_series[0:reference_frame,:,:], axis = 0))
register = CrossCorr()
model = register.fit(self.raw_series, reference=reference_image)

self.registered_series = model.transform(self.raw_series)
if len(self.registered_series.shape) == 3: #xyt
    self.registered_series = self.registered_series.toseries().toarray().transpose(2,0,1) # shape t, y, x
elif len(self.registered_series.shape) == 4: #xyzt
    self.registered_series = self.registered_series.toseries().toarray().transpose(3,0,1,2) # shape t, z, y, x

self.current_series = self.registered_series
