# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 10:55:25 2021

@author: Amirhossein
"""

import torch
import torchvision
from PIL import Image
from auto_encoder import AutoEncoder
import numpy as np
import cv2
import os
from tqdm import tqdm



def get_predict(model, device, image):
    model.eval()
    tensor_img =torch.unsqueeze(image, 0).to(device)
    predict = torch.squeeze(model(tensor_img), 0)
    predict = np.moveaxis(predict.to(torch.device('cpu')).detach().numpy(), 0, -1)
    predict[predict < 0] = 0
    predict[predict > 1] = 1
    return predict

        
def create_model_name(encoder_convs, fc_layers, decoder_convs, prefix):
    model_name = prefix + "_"
    for conv in encoder_convs:
        model_name += "{}_".format(conv[0])
    model_name += "_"
    for fc in fc_layers:
        model_name += "{}_".format(fc)
    for conv in decoder_convs:
        model_name += "_{}".format(conv[0])
    model_name += ".pt"
    return model_name


def read_image_as_tensor(image_path, image_shape):
    transforms = []
    transforms.append(torchvision.transforms.Resize(image_shape))
    transforms.append(torchvision.transforms.ToTensor())
    transforms = torchvision.transforms.Compose(transforms)
    
    img = transforms(Image.open(image_path))
    return img



input_path = "inputs"
input_shape = (1, 256, 256)

encoder_convs = [(4, (3,3), True),
                 (8, (3, 3), True),
                 (8, (3, 3), True),
                 (8, (3, 3), True),
                 (8, (3, 3), True)
                 ]

decoder_convs = [(8, (3, 3), True),
                 (8, (3, 3), True),
                 (8, (3, 3), True),
                 (8, (3, 3), True),
                 (1, (3, 3), True)
                 ]

fc_layers = [16]

model_name_prefix = 'trunk_real_pretrained_with_occlusion'
model_name = create_model_name(encoder_convs, fc_layers, decoder_convs, model_name_prefix)
save_model_path = os.path.join('../neural network/results/{}'.format(model_name[:-3]),
                               model_name)

auto_encoder = AutoEncoder(encoder_convs, fc_layers, decoder_convs, input_shape)
auto_encoder.load_state_dict(torch.load(save_model_path))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
auto_encoder.to(device)


input_image_names = os.listdir(input_path)

for image_name in tqdm(input_image_names):
    image_path = os.path.join(input_path, image_name)
    img = read_image_as_tensor(image_path, (input_shape[1], input_shape[2]))
    
    prediction = get_predict(auto_encoder, device, img)
    cv2.imwrite("predictions/{}".format(image_name), prediction * 255)







