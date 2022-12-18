import cv2
import numpy as np
from .apps.detection import Detection
from .apps.classification import Classification

from .configs.models import *
from .utils.colors import recognize_color

from typing import List
from typing import Optional
from pydantic import BaseModel

detection_config = DETECTION_CONFIG['avocado']
classification_config = CLASSIFICATION_CONFIG['avocado']
detection       = Detection(ROOT_PATH, detection_config)
classification  = Classification(ROOT_PATH, classification_config)


def get_avg_color(image):
    img_temp = image.copy()
    img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = np.average(image, axis=(0,1))
    img_temp = cv2.resize(img_temp, (25,25))
    return img_temp

def get_texture_image(image):
    h, w = image.shape[:2]
    # Calculate centroid
    cx, cy= int(w/2), int(h/2)
    # radius crop
    radius_cx , radius_cy = int(0.5*cx), int(0.5*cy)
    x_min, y_min, x_max, y_max = cx-radius_cx, cy-radius_cy, cx+radius_cx, cy+radius_cy
    return image[y_min:y_max, x_min:x_max]

def image_classification(image):
    results = list()
    image_resize = cv2.resize(
                image.copy(),
                (classification_config['image_size'], 
                classification_config['image_size']), 
                interpolation = cv2.INTER_AREA)
    results_class = classification(image_resize)
    for i in results_class:
        results.append([i[0], int(i[1]*100)])
        print(f'Classification : {i[0]} | confidecne : {int(i[1]* 100)} %')
    return results

class Detection(BaseModel):
    label : str
    conf  : float
    bbox  : list

class ResultFruit(BaseModel):
    detection	: Optional[Detection] = {}
    classification_texture: Optional[list] = []
    classification_color: Optional[str] = ''


def main_process(image):
    # results_fruit = {
    #     'detection': None,
    #     'classification_texture': list(),
    #     'classification_color': None
    # }
    image_resize = cv2.resize(
                image.copy(),
                (detection_config['image_size'], 
                detection_config['image_size']), 
                interpolation = cv2.INTER_AREA)
    cv2.imwrite('image.jpg', image)
    # Detection
    results = detection(image_resize, size=detection_config['image_size'])
    result_detection = detection.extract_results(
                results, 0.0,
                get_one=True, 
                boxes_ori=True, 
                resized_size=detection_config['image_size'])
    print(result_detection)
    # results['detection'] = result_detection
    try:
        confidence_c = result_detection['avocado'][0]['confidence']
        x_min_a, y_min_a, x_max_a, y_max_a = result_detection['avocado'][0]['bbox']
        img_avocade = image[y_min_a:y_max_a, x_min_a:x_max_a]
        # get texture image 
        img_texture = get_texture_image(img_avocade)
        # classification
        result_classification = image_classification(img_texture)
        # Color classification
        avg_color = get_avg_color(img_texture)
        b,g,r = avg_color[1,1]
        b,g,r =int(b),int(g),int(r) 
        result_color = recognize_color(b,g,r)
        result_fruit = ResultFruit(
            detection= {'label': 'avocado', 'conf': int(confidence_c*100), 'bbox': [x_min_a, y_min_a, x_max_a, y_max_a]},
            classification_texture = result_classification,
            classification_color = result_color
        )
    except KeyError:
        result_fruit = ResultFruit()
    # print(result_fruit.json())
    # print(type(result_fruit.json()))
    return result_fruit
