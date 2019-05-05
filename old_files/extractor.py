# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import torch
import torch.nn.functional as F

import torchvision
from torchvision import models, transforms

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
class Extractor():
    def __init__(self):

        self.my_embedding = torch.zeros(1,2048,8,8)

        self.model = models.inception_v3(pretrained=True).to(device)
            
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.layer = self.model._modules.get('Mixed_7c')
        self.model.eval()
                    
                        
    def extract(self, image_path):
        imsize = 299
        loader = transforms.Compose([transforms.Scale((imsize,imsize)), transforms.ToTensor()])
        
        from PIL import Image
        def image_loader(image_path):
            image = Image.open(image_path)
            image = loader(image).float()            
            image = image.unsqueeze(0)
            return image.to(device)
        
        def copy_data(m, i, o):
            self.my_embedding.copy_(o.data)
        
        # 5. Attach that function to our selected layer
        h = self.layer.register_forward_hook(copy_data)
        
        image = image_loader(image_path)
        # 6. Run the model on our transformed image
        self.model(image)                    
        # 7. Detach our copy function from the layer
        h.remove()
        
        # Adaptive average pooling
        features = F.adaptive_avg_pool2d(self.my_embedding, (1, 1))
        # N x 2048 x 1 x 1
        features = features.view(features.size(0), -1)
        # N x 2048

        return features

##image_path = './val/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01-0002.jpg'
#image_path = './val/ApplyLipstick/v_ApplyLipstick_g01_c01-0002.jpg'
#f = FeatureExtractor()
#ftr = f.extract(image_path)
#
#image_path = './val/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01-0002.jpg'
#f = FeatureExtractor()
#ftr2 = f.extract(image_path)
#
#image_path = './val/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01-0001.jpg'
#f = FeatureExtractor()
#ftr3 = f.extract(image_path)
#
#
#import numpy as np
#v = (ftr3-ftr2)**2
#v.sum()











