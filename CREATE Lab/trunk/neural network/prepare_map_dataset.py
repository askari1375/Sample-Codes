# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:58:57 2021

@author: Amirhossein
"""

from PIL import Image
import os
from tqdm import tqdm


image_path = "../../../../../../skills/deep learning/Datasets/maps/train"
image_names = list(os.walk(image_path))[0][2]

for image_name in tqdm(image_names):
    img = Image.open(os.path.join(image_path, image_name))
    
    x = img.crop((0, 0, img.size[0]/2, img.size[1]))
    y = img.crop((img.size[0]/2, 0, img.size[0], img.size[1]))
    
    x.save("data/train/input/{}.png".format(image_name[:-4]))
    y.save("data/train/output/{}.png".format(image_name[:-4]))



image_path = "../../../../../../skills/deep learning/Datasets/maps/val"
image_names = list(os.walk(image_path))[0][2]

for image_name in tqdm(image_names):
    img = Image.open(os.path.join(image_path, image_name))
    
    x = img.crop((0, 0, img.size[0]/2, img.size[1]))
    y = img.crop((img.size[0]/2, 0, img.size[0], img.size[1]))
    
    x.save("data/validation/input/{}.png".format(image_name[:-4]))
    y.save("data/validation/output/{}.png".format(image_name[:-4]))