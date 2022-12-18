import cv2
import numpy as np
from .apps.detection import Detection
from .apps.classification import Classification

from .configs.models import *
from .utils.colors import recognize_color

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
    results_conf = list()
    results_class = list()
    image_resize = cv2.resize(
                image.copy(),
                (classification_config['image_size'], 
                classification_config['image_size']), 
                interpolation = cv2.INTER_AREA)
    results_classification = classification(image_resize)
    for i in results_classification:
        results_conf.append(i[1])
        results_class.append(i[0])
        print(f'Classification : {i[0]} | confidecne : {i[1]} %')
    return results_class, results_conf


def main_process(image):
    image_resize = cv2.resize(
                image.copy(),
                (detection_config['image_size'], 
                detection_config['image_size']), 
                interpolation = cv2.INTER_AREA)
    # Detection
    results = detection(image_resize, size=detection_config['image_size'])
    result_detection = detection.extract_results(
                results, 0.0,
                get_one=True, 
                boxes_ori=True, 
                resized_size=detection_config['image_size'],
                image_shape=image.shape)
    image_drawed = detection.visualize_result(
        image.copy(), result_detection
    )

    # results['detection'] = result_detection
    try:
        confidence_c = result_detection['avocado'][0]['confidence']
        x_min_a, y_min_a, x_max_a, y_max_a = result_detection['avocado'][0]['bbox']
        img_avocade = image[y_min_a:y_max_a, x_min_a:x_max_a]
        # get texture image 
        img_texture = get_texture_image(img_avocade)
        # classification
        results_class, results_conf = image_classification(img_texture)
        # Color classification
        avg_color = get_avg_color(img_texture)
        b,g,r = avg_color[1,1]
        b,g,r =int(b),int(g),int(r) 
        result_color = recognize_color(b,g,r)
        # result_fruit = ResultFruit(
        #     detection= {'label': 'avocado', 'conf': int(confidence_c*100), 'bbox': [x_min_a, y_min_a, x_max_a, y_max_a]},
        #     classification_texture = result_classification,
        #     classification_color = result_color
        # )
    except KeyError:
        return image, dict(zip(classification_config['filter_classes'], map(float, [0,0,0,0])))
    # print(result_fruit.json())
    # print(type(result_fruit.json()))
    print(dict(zip(results_class, map(float, results_conf))))
    return image_drawed, dict(zip(results_class, map(float, results_conf)))
