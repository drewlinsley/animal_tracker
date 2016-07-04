from __future__ import division, print_function
import os 
import numpy as np
import csv
from scipy import misc, ndimage
from matplotlib import pyplot as plt
import re
from skimage.measure import label, regionprops

def find_location(model,image,categories,model_type,downsample,target_size,stride,threshold):
	res_target = np.divide(image.shape,downsample)
	res_image = resize_im(image,[res_target[0],res_target[1]])
	im_size = res_image.shape
	x_vec = range(0,im_size[0] - target_size[0],stride[0])
	y_vec = range(0,im_size[1] - target_size[1],stride[1])
	#localization_map = np.zeros(len(x_vec),len(y_vec))
        localization_map = np.zeros((im_size[0],im_size[1]))
	for x in x_vec:
#		print('X coordinate: ' + str(x))
		for y in y_vec:
			pred_image = np.zeros((im_size[0],im_size[1]))
			bb = [x,y,target_size[0],target_size[1]]
			image_crop = crop_im(res_image,bb)
			prediction = model_predict(model,adjust_im(image_crop,target_size),model_type)
                        test_map = test_criterion(prediction,categories,'binary')[0]
			pred_image[x:x+target_size[0],y:y+target_size[1]] = test_map
			localization_map = np.amax(np.dstack((localization_map,pred_image)),axis=2)
			#localization_map[x,y] = int(test_criterion(prediction,categories,'binary')[0] == 0)
	if threshold == 'mean':
		localization_map = localization_map > np.mean(localization_map)	
	elif threshold == 'median':	
                localization_map = localization_map > np.median(localization_map)		
	else:
		localization_map = int(localization_map > threshold)
	labeled_map = label(localization_map)
	props = regionprops(labeled_map)
	bb_array = biggest_box(props)	

	return res_image, localization_map, bb_array

def biggest_box(props):
	areas = []
	for b in props:
		areas.append(b.area)
	return np.asarray(props[np.argmax(np.asarray(areas))].bbox)

def save_image(image,outfile):
	plt.figure()
	plt.imshow(image)
	plt.savefig(outfile)
	plt.close('all')

def crop_im(image,bb):
	return image[bb[0]:bb[0] + bb[2],bb[1]:bb[1] + bb[3],:]

def read_file(name, deli):

	with open(name,'r') as f:
		reader=csv.reader(f,delimiter=deli,quoting=csv.QUOTE_NONE)
		out = list(reader)
	return out

def show_im(im):
	if 'numpy' in str(type(im)):
		plt.figure()
		plt.imshow(im)
	else: #expect list
		f, axarr = plt.subplots(1, 5)
		axarr[0].imshow(im[0])
		axarr[0].set_title('top left crop')
		axarr[1].imshow(im[1])
		axarr[1].set_title('top right crop')
		axarr[2].imshow(im[2])
		axarr[2].set_title('bottom right crop')
		axarr[3].imshow(im[3])
		axarr[3].set_title('bottom left crop')
		axarr[4].imshow(im[4])
		axarr[4].set_title('Blur')
	plt.show()

def rgb2gray(rgb):
	return np.dot(rgb[...,:3],[0.299,0.587,0.114])

def resize_im(im,target_size):
	im = misc.imresize(im,target_size)
	return im

def prepare_im(im,patch_params):
	#Crop as necessary and resize back to target_size
        #im = resize_im(im[patch_params[0]:patch_params[2],patch_params[1]:patch_params[2],:],target_size)
	im = im[patch_params[0]:patch_params[0] + patch_params[2],patch_params[1]:patch_params[1] + patch_params[3],:]
	if patch_params[-1] > 0:
		#If necessary, blur
		im = blur_descendent(im, patch_params[-1]) #In case the parent needs a blurring
	return im

def adjust_im(im,target_size):
	im = resize_im(im,target_size)
	im = im.astype(np.float32)
	#im[:,:,0] -= 103.939
	#im[:,:,1] -= 116.779
	#im[:,:,2] -= 123.68
	im[:,:,0] -= 122.6789143406786 #R
	im[:,:,1] -= 116.66876761696767 #G 
	im[:,:,2] -= 104.0069879317889 #B
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	return im

def revert_im(im):
	im = im[0,:,:,:].transpose((1,2,0))
	#im[:,:,0] += 103.939
	#im[:,:,1] += 116.779
	#im[:,:,2] += 123.68
	im[:,:,0] += 122.6789143406786 #R
	im[:,:,1] += 116.66876761696767 #G
	im[:,:,2] += 104.0069879317889 #B
	return im

def blur_descendent(im,factor):
	target_size = im.shape
        resize_target = np.round(np.multiply(target_size[0:2],factor)).astype(int) #might have got the resize factor backwards...
        im = resize_im(resize_im(im,(resize_target[0],resize_target[1],target_size[-1])),target_size)
        return im

def gather_descendents(idx,target_size,factor): #Old function version
	im = idx[0]
	params = idx[1]
	im_shape = im.shape
	#Produce crops of 20% at each corner
	coor_modifier = int(np.round(params[2] * factor))
	coor_x = params[0] + coor_modifier
	coor_y = params[1] + coor_modifier
	coor_diameter = params[2] - coor_modifier
	patch_modifier = int(np.round(params[2] * factor))
	x_diameter = im_shape[0] - patch_modifier
	y_diameter = x_diameter

	tli = [0,0,x_diameter,y_diameter,0] ; tl = [params[0],params[1],x_diameter,y_diameter,0] #First is image-patch based coordinates--second is global image-based coordinates
	tri = [patch_modifier,0,x_diameter,y_diameter,0] ; tr = [coor_x,params[1],coor_diameter,coor_diameter,0]
	bli = [0,patch_modifier,x_diameter,y_diameter,0] ; bl = [params[0],coor_y,coor_diameter,coor_diameter,0]
	bri = [patch_modifier,patch_modifier,x_diameter,y_diameter,0]; br = [coor_x,coor_y,coor_diameter,coor_diameter,0]
	bii = [0,0,im_shape[0],im_shape[1],params[4] + 1]; 
	bi = [params[0],params[1],params[2],params[3],params[4] + 1] #iterate the blur parameter
	descendent_ims = map(lambda x: prepare_im(im,x), [tli,tri,bli,bri,bii])
	descendent_list = [[descendent_ims[0],tl],[descendent_ims[1],tr],[descendent_ims[2],bl],\
		[descendent_ims[3],br],[descendent_ims[4],bi]]
	return descendent_list, descendent_ims

def check_ml(MIRC_list,patch_candidate):
	#Implement a search algorithm to accellerate search of MIRCs
	add_mirc = 1
	p_coors = []
	for p in MIRC_list:
		p_coors = p[1]
		p_xy = p_coors[0:1]
		p_sb = p_coors[2:-1]
		if patch_candidate[1][0:1] == p_xy and patch_candidate[1][2:-1] >= p_sb:
			add_mirc = 0 #do not add this MIRC!
			break
	return add_mirc

def check_params(patch_list,descendent):
	#For adding patches to the search list
	add_patch = 1
	for p in patch_list:
		if p == descendent:
			add_patch = 0
			break
	return add_patch

def ismember(a,b):
	bind = {}
	for i, elt in enumerate(b):
		if elt not in bind:
			bind[elt] = 1

	out_1 = np.asarray([bind.get(itm, 0) for itm in a])
	out_2 = np.asarray([i for i, x in enumerate(out_1) if x == 1])
	return out_1, out_2

def test_criterion(im_pred,target_category,criterion):
	#Determine if we have reached a MIRC
	mirc_status = 0;
	if criterion == 'binary': #Continue until chosen category is no longer the max
		mirc_status = np.max(im_pred[0][target_category])
		top_prob = 0;
		#candidates = np.argsort(im_pred[0])[::-1][0:5] #[0] = top-1, [0:5] = top-5
		#top_prob = np.sort(im_pred[0][target_category])[::-1][0]
		#membership,ids = ismember(candidates[0],target_category)
		#if ids.size>0: #Correct recognition
		#	0
		#else: #potential mirc -- incorrect categorization
		#	mirc_status = 1
	elif criterion == 'median': #Ullman's cutoff for descendants
		top_prob = np.max(im_pred[0][target_category])
		if top_prob > .5: #Correct recognition
			0
		else: #potential mirc -- incorrect categorization
			mirc_status = 1
	return mirc_status, top_prob

def multi_test_criterions(descendents,target_category,criterion):
	MIRC_candidates = []
	MIRC_scores = []
	for d in descendents:
		c,s = test_criterion(d,target_category,criterion)
		MIRC_candidates.append(c)
		MIRC_scores.append(s)
	return MIRC_candidates, MIRC_scores

def update_patch_array(patch_array,remove_these):
	updated_patches = []
	for i in range(0,len(patch_array)):
		if i in remove_these:
			0
		else:
			updated_patches.append(patch_array[i])
	return updated_patches

def initialize_model(model_pointer,model_type,prototxt):
	if model_type == 'keras':
		from keras.optimizers import SGD
		if re.findall('16',model_pointer):
			from run_vgg16 import VGG_16
			model = VGG_16(model_pointer)
		elif re.findall('19',model_pointer):
			from run_vgg19 import VGG_19
			model = VGG_19(model_pointer)
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd, loss='categorical_crossentropy')
	elif model_type == 'caffe':
		import caffe
		caffe.set_mode_gpu()
		model = caffe.Net(prototxt, model_pointer, caffe.TEST)
	return model

def model_predict(model,im,model_type):
	if model_type == 'caffe':
		pdic = model.forward_all(data=im)
		pdic = pdic['prob']
		#model.blobs['data'].data[...] = im
		#output = model.forward()
		#pdic = output['prob']  # the output probability vector for the first image in the batch
	elif model_type == 'keras':
		pdic = model.predict(im)
	return pdic

def manual_conv(im,size,model,im_size,target_size,factor,target_category,criterion,output_dir,short_run,min_size,model_type):
	#For each scale and x/y position, perform a MIRC search
	remove_these = []
	MIRC_list = []
	MIRC_count = 0
	#Track a mask for x/y/size so that these can be ignored as we find MIRCs. if a mirc was found at a position
	#then if x/y == that x/y + scale/blur is >= that position, don't search here.
	for start_size in size:
		print('Working size ' + str(start_size))
		for x in range(0,im_size[0]):
			if x + start_size > im_size[0]: #If we're beyond the image boundary, move to the next scale
				print('Beyond x coordinate extent')
				break
			for y in range(0,im_size[1]):
				if y + start_size > im_size[1]: #If we're beyond the image boundary, move to the next scale
					print('Beyond x coordinate extent')
					break
				init_transformation = [x,y,start_size,start_size,0]
				init_parent = prepare_im(im,init_transformation)
				init_check, init_prob = test_criterion(model_predict(model,adjust_im(init_parent,target_size),model_type),target_category,criterion)				
				if init_check == 1: #incorrect categorization
					#cant recognize
					print('skipping cause I cant recognize')
				else: #we can recognize this image, pursue MIRCs
					patch_array = [[init_parent,[x,y,start_size,start_size,0],init_prob]] #im/x/y/diameter/blurs
					pal = len(patch_array)
					while pal > 0 and pal < 1e5:
						#After each pass through the loop remove elements of patch_array
						patch_array = update_patch_array(patch_array,remove_these)
						remove_these = []
						for idx in range(0,len(patch_array)): #For each patch in the list
							descendent_idx,dis = gather_descendents(patch_array[idx],target_size,factor) #Gather descendents
							cnn_dis = map(lambda x: adjust_im(x,target_size), dis)
							descendent_predictions = map(lambda x: model_predict(model,x,model_type), cnn_dis) #Predict object category from each
							MIRC_candidates, MIRC_scores = multi_test_criterions(descendent_predictions,target_category,criterion) #see which pass the threshold
							if np.sum(MIRC_candidates) == len(dis): #This patch is a MIRC
								add_mirc = check_ml(MIRC_list,patch_array[idx])
								if add_mirc:
									print(str(patch_array[idx][1]) + ' -- Found a MIRC!')
									MIRC_list.append(patch_array[idx]) #add idx along with patch_info -- we are done looking at this position
									#The followoing line is stupid... need to figure out a better way to package the MIRC probabilities in the patch_array
									_, patch_prob = test_criterion(model_predict(model,adjust_im(patch_array[idx][0],target_size),model_type),target_category,criterion)				
									MIRC_package = [patch_array[idx],descendent_idx,patch_prob,MIRC_scores]
									np.save(output_dir + '/' + str(MIRC_count) + '.npy',MIRC_package) #save the mirc image
									MIRC_count += 1
									patch_array.pop(idx)
								if short_run == 'on': #Don't search here anymore
									patch_array = []
									break
							else: 
								print(str(patch_array[idx][1]) + ' -- Candidate results: ' + str(MIRC_candidates) + ' : Array size = ' + str(len(patch_array)))
								remove_these.append(idx)
								for ix in range(0,len(MIRC_candidates)):
									if MIRC_candidates[ix] == 0 and descendent_idx[ix][1][2] > min_size and check_params(patch_array,descendent_idx[ix][1]) and check_ml(MIRC_list,descendent_idx[ix]):
										#patch_array.append([descendent_idx[ix],MIRC_scores]) #Find patches to proceed with and add them to the list
										patch_array.append(descendent_idx[ix]) #Find patches to proceed with and add them to the list
						pal = len(patch_array) #Recount patch_array

	print('!!!!!!Finished!!!!!!')
	return MIRC_list
