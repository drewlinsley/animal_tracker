from __future__ import division, print_function
import os
import numpy as np
import csv
from scipy import misc, ndimage
from matplotlib import pyplot as plt
from monkey_localization.mirc_fun import *
from monkey_localization import run_vgg19
import re
import glob
from tqdm import tqdm

#Settings
main_dir = '/home/drew/Documents/neural_decoding/monkey_skeletons/template_matching'
image_dir = '/home/drew/Documents/neural_decoding/monkey_skeletons/treadmill5'
project_name = 'monkey_localization'
output_dir = main_dir + '/' + 'cropped_images'
model_dir = main_dir + '/' + project_name + '/' + 'models/'
model_name = 'vgg19_weights.h5'
model_type = 'keras'

#Start fun
model = initialize_model(model_dir + model_name, model_type, []) #Add a prototxt if using a caffe

#Prepare model targets
csv_file = model_dir + 'monkey.csv'
target_categories = np.where(np.asarray(read_file(csv_file,',')).astype(np.uint8)==1)[0] 
downsample = 4
target_size = (224,224,3)
stride = [5,5]
threshold = 'median'
produce_crops = True

#List images
ims = sorted(glob.glob(image_dir + '/*.jpeg'), key = os.path.getmtime)

#Loop through images
bb_array = np.zeros((len(ims),4))
for i in tqdm(range(0,len(ims))):
	ti = misc.imread(ims[i])
	ri, hm, bb = find_location(model,ti,target_categories,model_type,downsample,target_size,stride,threshold)
	bb_array[i,:] = bb
	if produce_crops:
		cropped_im = crop_im(ri,bb)
		im_name = output_dir + '/bb_' + re.split('/',ims[i])[-1]
		save_image(cropped_im,im_name)
np.save(output_dir + '/bb_array.npy',bb_array)

