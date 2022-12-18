'''
@Author     : Ali Mustofa HALOTEC
@Module     : Container Number Iso Code Detection
@Created on : 10 Oct 2022
'''
#!/usr/bin/env python3
# Path: src/apps/detection.py
import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional
from pydantic import BaseModel
from src.utils.utils import download_and_unzip_model

import torch
from torch import nn

class InputDetection(BaseModel):
	image: 			list
	min_confidence: Optional[float] = 0.5
	get_one: 		Optional[bool] 	= True
	boxes_ori: 		Optional[bool] 	= True
	resized_size: 	Optional[int] 	= 640
	image_shape:	Optional[tuple] = (1280, 720, 3)
	
class Detection:
	def __init__(self, root_path:str, model_config:dict) -> None:
		'''
		Load model 
		@params:
			- root_path:str ->  root of path model
			- model_config:dict -> config of model {filename, classes, url, file_size}
		'''
		self.root_path :str          = root_path
		self.model_config :dict      = model_config
		self.model_name :str         = f'{root_path}/{model_config["filename"]}'
		self.image_size :int         = model_config['image_size']
		self.classes :list           = model_config['classes']
		self.filter_classes :list    = model_config['filter_classes']
		self.device :torch           = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.model :nn.Module        = self.__load_model()

	@staticmethod
	def __check_model(root_path:str, model_config:dict) -> None:
		if not os.path.isfile(f'{root_path}/{model_config["filename"]}'):
			download_and_unzip_model(
				root_dir    = root_path,
				name        = model_config['filename'],
				url         = model_config['url'],
				file_size   = model_config['file_size'],
				unzip       = False
			)
		else: print('Load model char detection')

	def __load_model(self) -> torch.nn.Module:
		self.__check_model(self.root_path, self.model_config)
		try:
			model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_name)
		except:
			model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_name, force_reload=True)
		return model
	
	def extract_results(self, results, min_confidence:float = 0.5, get_one=False, \
		boxes_ori:bool = True, resized_size:int = 0, image_shape:tuple = (1080, 720)):
		'''
		Format result([tensor([[151.13147, 407.76913, 245.91382, 454.27802,   0.89075,   0.00000]])])
		Filter min confidence prediction and classes id/name
		Cropped image and get index max value confidence lavel
		Args:
			result(models.common.Detections): result detection YoloV5
			min_confidence(float): minimal confidence detection in range 0-1
			boxes_ori(bool): if true, calculate boxes to original resolution
			resized_size(int): value of resized image detection
			image_shape(tuple): height, width image original
		Return:
			result(dict): {
				casess:[{
					confidence(float): confidence,
					bbox(list) : [x_min, y_min, x_max, y_max]
				}]
			}
		'''
		results_format  = results.xyxy
		results_filter =  dict({i:list() for i in self.classes})
		
		if len(results_format[0]) >= 1:
			for i in range(len(results_format[0])):
				classes_name    = self.classes[int(results_format[0][i][-1])]
				confidence      = float(results_format[0][i][-2])
				if classes_name in self.filter_classes and confidence >= min_confidence:

					x_min, y_min = int(results_format[0][i][0]), int(results_format[0][i][1])
					x_max, y_max = int(results_format[0][i][2]), int(results_format[0][i][3])
					# change coordinate to original w n h
					if boxes_ori:
						resized_size = self.image_size if resized_size == 0 else resized_size
						height, width = image_shape[:2]
						x_min = int((x_min/resized_size)*width)
						y_min = int((y_min/resized_size)*height)
						x_max = int((x_max/resized_size)*width)
						y_max = int((y_max/resized_size)*height)

					if get_one:
						if results_filter[classes_name]:
							if results_filter[classes_name][0]['confidence'] < confidence:
								results_filter[classes_name][0] = \
									{'confidence': round(confidence,2), 
									'bbox':[x_min, y_min, x_max, y_max]}
							# else: pass
						else:
							results_filter[classes_name].append({
								'confidence': round(confidence,2), 
								'bbox':[x_min, y_min, x_max, y_max]
							})
					else:
						results_filter[classes_name].append({
							'confidence': round(confidence,2), 
							'bbox':[x_min, y_min, x_max, y_max]
						})
		# Delete key if detection null
		for i in self.classes:results_filter.__delitem__(i) if not results_filter[i] else None

		return results_filter
	
	@staticmethod
	def visualize_result(image:np.array, results_filter:dict) -> np.array:
		'''
		Draw bounding box result
		Args:
			image(numpy.ndarray) : image/frame
			results_filter(dict) : {
				casess:[{
					confidence(float): confidence,
					bbox(list) : [x_min, y_min, x_max, y_max]
				}]
			}
		Return:
			image(numpy.ndarray) : image drawed
		'''
		for result in results_filter.items():
			label = result[0]
			for i in result[1]:
				conf = int(i['confidence']*100)
				bbox = i['bbox']
				x_min, y_min, x_max, y_max = \
				bbox[0], bbox[1], bbox[2], bbox[3]

				cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (86, 71, 255), 2)
				# Draw text
				label_conf = f'{label.upper()}:{conf}%'
				cv2.rectangle(image, (x_min, y_min-20), (x_min+(len(label_conf)*13), y_min), (86, 71, 255), cv2.FILLED)
				cv2.putText(image, label_conf, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
		return image

	def __call__(self, image:np.array, size:int = None):
		'''
		Prediction image object detectionn YoloV5
		Args:
			image(numpy.ndarray) : image/frame
		Return:
			results_prediction(models.common.Detections) : results -> convert to (results.xyxy/resultsxywh)
		'''
		if size: results = self.model(image, size=size)
		else: results = self.model(image)
		return results

if __name__ == '__main__':
	root_path = os.path.expanduser('/Users/alimustofa/Downloads/')
	detection_config = {
		'filename'          : 'avori_detection.pt',
		'image_size'        : 640,
		'classes'           : ['avocado'],
		'filter_classes'    : ['avocado'],
		'url'               : 'https://github.com/Alimustoofaa/ContainerNumber-Dev/releases/download/detection_v1/container_iso_maxgross.pt',
		'file_size'         : 14749585,
		
	}
	detection = Detection(root_path, detection_config)

	image = cv2.imread('12022051823052897.jpg')
	image_resize = cv2.resize(
					image.copy(),
					(detection_config['image_size'], 
					detection_config['image_size']), 
					interpolation = cv2.INTER_AREA)

	results = detection(image_resize, size=detection_config['image_size'])
	filtered = detection.extract_results(
					results, 0.5,
					get_one=True, 
					boxes_ori=True, 
					resized_size=detection_config['image_size'], 
					image_shape=image.shape)
	img_drawed = detection.visualize_result(image.copy(), filtered)
	cv2.imwrite('img_drawed.jpg', img_drawed)
	Image.fromarray(cv2.cvtColor(img_drawed, cv2.COLOR_BGR2RGB))