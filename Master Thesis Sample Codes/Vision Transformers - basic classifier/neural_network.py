# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 19:50:03 2021

@author: Amirhossein
"""

import torch
import timm
import torch.nn as nn
import torchvision
from torchsummary import summary as torch_summary

"""
        Args:
            input_shape: the dimension of the input image.

            num_classes: number of output classes.

            convs: it's 'AlexNet' or a list of tuples with these info:
                (out_channels, kernel_size, stride, padding, has_pooling), the first four are
                
            fcs: a list of integers representing number of Linear neurons in each layer

            conv_drop_rate: float(0-1), drop rate used for Conv2d layers

            fc_drop_rate: float(0-1), drop rate used for Linear layers
"""

class VisionTransormer(nn.Module):
    def __init__(self, backbone_name, num_classes, last_layers_number):
        super(VisionTransormer, self).__init__()
        
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes = num_classes)
        model_parameters = list(self.model.parameters())
        for param in model_parameters[:-1 * last_layers_number]:
            param.requires_grad = False
        
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x):
        output = self.model(x)
        return output

    def get_loss(self, outputs, targets):
        loss = self.cross_entropy(outputs, targets)
        return loss
    
    def summary(self, input_size):
        torch_summary(self, input_size)


def main():
    
    num_classes = 5
    backbone_name = 'vit_base_patch16_224'
    last_layers_number = 4
    
    model = VisionTransormer(backbone_name, num_classes, last_layers_number)
    device = torch.device("cpu")

    model.to(device)
    
    torch_summary(model, (3, 224, 224), device = 'cpu')
    print(model)



if __name__ == "__main__":
    main()