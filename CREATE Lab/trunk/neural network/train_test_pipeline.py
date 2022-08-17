# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:02:51 2021

@author: Amirhossein
"""

import torch
from torch.utils.data import DataLoader
from picture_picture_dataset import ImageDataset
from auto_encoder import AutoEncoder
from train_test_functions import train, test
import os


""" ------------------------------- Config ------------------------------- """

input_shape = (1, 256, 256)

BATCH_SIZE = 16
learning_rate = 1e-3

DATA_LOADER_KWARGS = {
    'batch_size': BATCH_SIZE,
    'shuffle': True,
    'pin_memory': True
}

key_word = "with"
train_input_path = "data/real trunk {} occlusion/train/input".format(key_word)
train_output_path = "data/real trunk {} occlusion/train/output".format(key_word)

val_input_path = "data/real trunk {} occlusion/validation/input".format(key_word)
val_output_path = "data/real trunk {} occlusion/validation/output".format(key_word)

save_results_folder = "results/"

model_name_prefix = 'trunk_real_pretrained_{}_occlusion'.format(key_word)

pre_trained_model_path = "results/trunk_3D_without_occlusion_4_8_8_8_8__16__8_8_8_8_1/trunk_3D_without_occlusion_4_8_8_8_8__16__8_8_8_8_1.pt"
#pre_trained_model_path = None

""" ----------------------------- data loader ----------------------------- """


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


""" ----------------------------- data loader ----------------------------- """


train_dataset = ImageDataset(train_input_path, train_output_path, size = -1)
val_dataset = ImageDataset(val_input_path, val_output_path, size = -1)

train_loader = DataLoader(train_dataset, **DATA_LOADER_KWARGS)
val_loader = DataLoader(val_dataset, **DATA_LOADER_KWARGS)


""" --------------------------- Neural Network --------------------------- """

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

model_name = create_model_name(encoder_convs, fc_layers, decoder_convs, model_name_prefix)

save_folder = save_results_folder + model_name[:-3]
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
    os.mkdir(save_folder + "/progress")
    os.mkdir(save_folder + "/validation prediction")





device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

auto_encoder = AutoEncoder(encoder_convs, fc_layers, decoder_convs, input_shape)

if pre_trained_model_path is not None:
    auto_encoder.load_state_dict(torch.load(pre_trained_model_path))


auto_encoder.to(device)

auto_encoder.summary(input_shape)


""" ---------------------------- Train Process ---------------------------- """


optimizer = torch.optim.Adam(auto_encoder.parameters(), lr = learning_rate)

train(auto_encoder,
      train_loader,
      val_loader,
      optimizer,
      num_epochs = 100,
      device = device,
      save_folder = save_folder,
      model_name = model_name)


""" ---------------------------- Test Process ---------------------------- """

save_model_path = os.path.join(save_folder, model_name)
auto_encoder.load_state_dict(torch.load(save_model_path))

test(auto_encoder, val_loader, save_folder + "/validation prediction", device)




