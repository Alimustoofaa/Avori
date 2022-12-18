import os
import torch
from PIL import Image
import numpy as np
from torch import cuda, device
from torchvision import transforms
from src.utils.utils import download_and_unzip_model

class Classification:
    def __init__(self, root_path, model_config):
        self.root_path = root_path
        self.model_config = model_config
        self.model_name = f'{root_path}/{model_config["filename"]}'
        self.image_size = model_config['image_size']
        self.classes = model_config['classes']
        self.filter_classes = model_config['filter_classes']
        self.device = device("cuda:0" if cuda.is_available() else "cpu")
        self.transform_image = self.transfrom_img()
        self.model = self.__load_model()
    
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
        else: print('Load model Classification')

    def __load_model(self):
        self.__check_model(self.root_path, self.model_config)
        print(self.model_name)
        return torch.load(self.model_name, map_location=self.device)
        
    def transfrom_img(self):
        return transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            ])
    
    @classmethod
    def tensor_image_to_numpy(cls, tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    def image_processing(self, image):
        image = self.transform_image(image)
        image = image.unsqueeze(0)
        return image.to(self.device)
    
    def __call__(self, image):
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        image = image.resize((self.image_size, self.image_size))
        image = self.image_processing(image)
        if torch.cuda.is_available():
            image_tensor = image.view(1, 3, 224, 224).cuda()
        else:
            image_tensor = image.view(1, 3, 224, 224)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        probs = torch.exp(output).to(self.device)
        topk, topclass = probs.topk(len(self.filter_classes), dim=1)
        result = list()
        for i in range(len(probs[0])):
            classes =  self.classes[topclass.cpu().numpy()[0][i]]
            conf = round(topk.cpu().numpy()[0][i], 2)
            if classes in self.filter_classes:
                result.append([classes, conf])
        return result

if __name__ == '__main__':
    root_path = os.path.expanduser('/Users/alimustofa/Downloads/')
    classification_config = {
        'filename'          : 'resnet34_model_400.pt',
        'image_size'        : 480,
        'classes'           :['almost_ripe', 'not_ripe', 'overripe', 'ripe'],
        'filter_classes'    : ['almost_ripe', 'not_ripe', 'overripe', 'ripe'],
        'url'               : 'https://github.com/Alimustoofaa/ContainerNumber-Dev/releases/download/detection_v1/container_iso_maxgross.pt',
        'file_size'         : 14749585,
    }

    # model_name = '/Users/alimustofa/Downloads/resnet34_model_avorit.pt'
    avori_classification = Classification(root_path,model_config=classification_config)