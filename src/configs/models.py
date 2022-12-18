import os

ROOT_PATH = os.path.expanduser('/Users/alimustofa/Downloads')

DETECTION_CONFIG = {
    'avocado': {
        'filename'          : 'avori_detection.pt',
        'image_size'        : 640,
        'classes'           : ['avocado'],
        'filter_classes'    : ['avocado'],
        'url'               : 'https://github.com/Alimustoofaa/ContainerNumber-Dev/releases/download/detection_v1/container_iso_maxgross.pt',
        'file_size'         : 14749585,
    }
}

CLASSIFICATION_CONFIG = {
    'avocado': {
        'filename'          : 'resnet34_model_avorit.pt',
        'image_size'        : 480,
        'classes'           :['almost_ripe', 'not_ripe', 'overripe', 'ripe'],
        'filter_classes'    : ['almost_ripe', 'not_ripe', 'overripe', 'ripe'],
        'url'               : 'https://github.com/Alimustoofaa/ContainerNumber-Dev/releases/download/detection_v1/container_iso_maxgross.pt',
        'file_size'         : 14749585,
    }
}

