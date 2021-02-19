# Install lungmask from https://github.com/amrane99/lungmask using pip install git+https://github.com/amrane99/lungmask
from lungmask import mask
import SimpleITK as sitk
import os
import numpy as np
from mp.utils.load_restore import pkl_dump
from mp.paths import storage_data_path
import mp.data.datasets.dataset_utils as du

def LungSegmentation(input_path, target_path, gpu=False, cuda=0):

	# Load ct scan and create segmentation
	input_image = sitk.ReadImage(input_path)
	segmentation = mask.apply(image=input_image, gpu=gpu, cuda=cuda)  # default model is U-net(R231)

	# load alternative models
	# model = mask.get_model('unet','LTRCLobes')
	# segmentation = mask.apply(input_image, model)

	file_name = input_path.split('/')[-1].split('.nii')[0]
	sitk.WriteImage(sitk.GetImageFromArray(segmentation), os.path.join(target_path, file_name+"_lung_seg.nii.gz"))
	sitk.WriteImage(input_image, os.path.join(target_path, file_name+".nii.gz"))

	return segmentation

def calculateSegmentationVolume(original_file_path, scan_np):
	# If a pixel is segmented, the volume is voxel space (x * y * z) --> Remember scaling if scl_slope field in original image is nonzero
	reader = sitk.ImageFileReader()
	reader.SetFileName(original_file_path)
	reader.LoadPrivateTagsOn()
	reader.ReadImageInformation()

	voxel_x = float(reader.GetMetaData('pixdim[1]'))
	voxel_y = float(reader.GetMetaData('pixdim[2]'))
	voxel_z = float(reader.GetMetaData('pixdim[3]'))
	try:
		# If 10, this indicated mm and seconds
		voxel_unit = int(reader.GetMetaData('xyzt_units')) 
	except:
		# If error occurs, field is empty, so set to mm
		voxel_unit = 10

	scl_slope = float(reader.GetMetaData('scl_slope'))
	scl_inter = float(reader.GetMetaData('scl_inter'))
	
	if scl_slope != 0:
		voxel_vol = (scl_slope * voxel_x + scl_inter)\
				  * (scl_slope * voxel_y + scl_inter)\
				  * (scl_slope * voxel_z + scl_inter)
	else:
		voxel_vol = voxel_x * voxel_y * voxel_z
 
	# Calculate segmentation volume based on voxel_vol
    # Determine start index and end index of segmentation (0-based)
    # --> Start and end point of Lung
	start_seg = True
	end_seg = True
	discard = False
	start_seg_idx = None
	end_seg_idx = None
	for idx, ct_slice in enumerate(scan_np):
		transp = np.transpose(np.nonzero(ct_slice))
		if len(transp) != 0 and idx > 0:
			start_seg = False
			start_seg_idx = idx
			break
		if len(transp) != 0 and idx == 0:
			discard = True
			break

	reversed_scan = scan_np[::-1]
	#if not discard:
	for idx, ct_slice in enumerate(reversed_scan):
		transp = np.transpose(np.nonzero(ct_slice))
		if len(transp) != 0 and idx > 0:
			end_seg = False
			end_seg_idx = len(reversed_scan) - idx - 1 # to get 0-based
			break
		if len(transp) != 0 and idx == 0:
			discard = True
			break
	
	# Calculate segmentation volume based on voxel_vol
	segmentation_volume = 0
	#if not discard:
	for ct_slice in scan_np:
		transp = np.transpose(np.nonzero(ct_slice))
		# Calculate volume of segmented 2D slice
		segmentation_volume += len(transp)*voxel_vol
	
	if voxel_unit == 10: # segmentation_volume is in mm^3
		segmentation_volume /= 1000 # to ml
	
	segmentation_volume = int(segmentation_volume)
	print("The segmentation has a volume of {} ml.".format(segmentation_volume))

	return discard, segmentation_volume, start_seg_idx, end_seg_idx

def CheckWholeLungCaptured(input_path, target_path, gpu=False, cuda=0):
	scan_np = LungSegmentation(input_path, target_path, gpu, cuda)
	discard, segmentation_volume, start_seg_idx, end_seg_idx = calculateSegmentationVolume(input_path, scan_np)

	return discard, segmentation_volume, start_seg_idx, end_seg_idx


"""Grand Chalenge Data"""
def GC(source_path, target_path, gpu=True, cuda=7):
	r"""Extracts MRI images and saves the modified images."""
	# Filenames have the form 'volume-covid19-A-XXXX_ct.nii'
	filenames = [x for x in os.listdir(source_path) if 'covid19' in x
					and '_seg' not in x and '._' not in x]

	# Create directories if not existing
	if not os.path.isdir(target_path):
		os.makedirs(target_path)

	result = dict()
	for num, filename in enumerate(filenames):
		discard, tlc, start_seg_idx, end_seg_idx = CheckWholeLungCaptured(os.path.join(source_path, filename), target_path, gpu, cuda)
		
		if not discard:
			print("Based on start index of the segmentation {} and the end index of the segmentation {}, the whole lung should be captured.".format(start_seg_idx, end_seg_idx))
			print("Total Lung Capacity: {} ml.".format(tlc))
			# Thresholds might be adapted
			if 4000 < tlc and tlc < 4400:
				print("\n Based on the total lung capacity of {} ml, the CT scan might be from a woman, since it fits the average total lung capacity of a women (approx. 4200 ml).".format(tlc))
			if 5800 < tlc and tlc < 6200:
				print("\n Based on the total lung capacity of {} ml, the CT scan might be from a man, since it fits the average total lung capacity of a man (approx. 6000 ml).".format(tlc))
		else:
			print("Based on start index of the segmentation {} and the end index of the segmentation {}, the whole lung is not captured.".format(start_seg_idx, end_seg_idx))


		result[filename] = [discard, tlc, start_seg_idx, end_seg_idx]
	
	# Save dict
	pkl_dump(result, 'GC', path=target_path)


"""Decathlon Lung Data"""
def Decathlon(source_path, target_path, gpu=True, cuda=7):
	r"""Extracts MRI images and saves the modified images."""
	images_path = os.path.join(source_path, 'imagesTr')

	# Filenames have the form 'lung_XXX.nii.gz'
	filenames = [x for x in os.listdir(images_path) if x[:4] == 'lung']

	# Create directories if not existing
	if not os.path.isdir(target_path):
		os.makedirs(target_path)

	result = dict()
	for num, filename in enumerate(filenames):		
		discard, tlc, start_seg_idx, end_seg_idx = CheckWholeLungCaptured(os.path.join(images_path, filename), target_path, gpu, cuda)
		
		if not discard:
			print("Based on start index of the segmentation {} and the end index of the segmentation {}, the whole lung should be captured.".format(start_seg_idx, end_seg_idx))
			print("Total Lung Capacity: {} ml.".format(tlc))
			# Thresholds might be adapted
			if 4000 < tlc and tlc < 4400:
				print("\n Based on the total lung capacity of {} ml, the CT scan might be from a woman, since it fits the average total lung capacity of a women (approx. 4200 ml).".format(tlc))
			if 5800 < tlc and tlc < 6200:
				print("\n Based on the total lung capacity of {} ml, the CT scan might be from a man, since it fits the average total lung capacity of a man (approx. 6000 ml).".format(tlc))
		else:
			print("Based on start index of the segmentation {} and the end index of the segmentation {}, the whole lung is not captured.".format(start_seg_idx, end_seg_idx))


		result[filename] = [discard, tlc, start_seg_idx, end_seg_idx]

	# Save dict
	pkl_dump(result, 'Decathlon', path=target_path)


"""Frankfurt Uniklinik Data"""
def FRA(source_path, target_path, gpu=True, cuda=7):
	r"""Extracts MRI images and saves the modified images."""
	images_path = source_path

	# Filenames are provided in foldernames: patient_id/images.nii.gz
	filenames = set(file_name for file_name in os.listdir(images_path)
					if file_name[:1] != '.')
					
	# Create directories if not existing
	if not os.path.isdir(target_path):
		os.makedirs(target_path)

	result = dict()
	for num, filename in enumerate(filenames):
		discard, tlc, start_seg_idx, end_seg_idx = CheckWholeLungCaptured(os.path.join(images_path, filename, 'image.nii.gz'), target_path, gpu, cuda)
		
		if not discard:
			print("Based on start index of the segmentation {} and the end index of the segmentation {}, the whole lung should be captured.".format(start_seg_idx, end_seg_idx))
			print("Total Lung Capacity: {} ml.".format(tlc))
			# Thresholds might be adapted
			if 4000 < tlc and tlc < 4400:
				print("\n Based on the total lung capacity of {} ml, the CT scan might be from a woman, since it fits the average total lung capacity of a women (approx. 4200 ml).".format(tlc))
			if 5800 < tlc and tlc < 6200:
				print("\n Based on the total lung capacity of {} ml, the CT scan might be from a man, since it fits the average total lung capacity of a man (approx. 6000 ml).".format(tlc))
		else:
			print("Based on start index of the segmentation {} and the end index of the segmentation {}, the whole lung is not captured.".format(start_seg_idx, end_seg_idx))


		result[filename] = [discard, tlc, start_seg_idx, end_seg_idx]

	# Save dict
	pkl_dump(result, 'FRA', path=target_path)


if __name__ == '__main__':
	# Extract necessary paths GC   
	global_name = 'GC_Corona'
	dataset_path = os.path.join(storage_data_path, global_name, 'Train')
	original_data_path = os.path.join(du.get_original_data_path(global_name), 'Train')
	print("Start with Grand Challenge train dataset.")
	GC(original_data_path, dataset_path)

	dataset_path = os.path.join(storage_data_path, global_name, 'Validation')
	original_data_path = os.path.join(du.get_original_data_path(global_name), 'Validation')
	print("Start with Grand Challenge validation dataset.")
	GC(original_data_path, dataset_path)

	# Extract necessary paths Decathlon
	global_name = 'DecathlonLung'
	dataset_path = os.path.join(storage_data_path, global_name)
	original_data_path = du.get_original_data_path(global_name)
	print("Start with Decathlon Lung dataset.")
	Decathlon(original_data_path, dataset_path)

	# Extract necessary paths FRA
	global_name = 'FRACorona'
	dataset_path = os.path.join(storage_data_path, global_name, 'train')
	original_data_path = os.path.join(du.get_original_data_path(global_name), 'train')
	print("Start with train dataset of FRA UK.")
	FRA(original_data_path, dataset_path)

	dataset_path = os.path.join(storage_data_path, global_name, 'test')
	original_data_path = os.path.join(du.get_original_data_path(global_name), 'test')
	print("Start with test dataset of FRA UK.")
	FRA(original_data_path, dataset_path)