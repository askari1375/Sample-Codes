# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 11:28:12 2021

@author: Amirhossein
"""

import torch
import torch.nn as nn
from torchsummary import summary as torch_summary


class AutoEncoder(nn.Module):
    
    """
        Args:
            encoder_convs : list of tuples with this informations:
                (output_channels, kernel_size, has_pool)
            decoder_convs : list of tuples with this informations:
                (output_channels, kernel_size, has_upsample)
            input_shape : (depth, height, width)
            conv_drop_rate : None or float number between 0 and 1
    """
    
    def __init__(self, encoder_convs, fcs, decoder_convs, input_shape, conv_drop_rate = None, fc_drop_rate = None):
        super(AutoEncoder, self).__init__()
        
        self.mse_loss = nn.MSELoss()
        
        encoder_layers = []
        last_layer_channels_number = input_shape[0]
        
        for conv in encoder_convs:
            
            """
                Layers:
                    1. Conv2d
                    2. BatchNorm2d
                    3. ReLU
                    4. Dropout if 'conv_drop_rate' is not None
                    5. MaxPooling(2,2) is has_pool is true            
            """
            
            encoder_layers.append(nn.Conv2d(in_channels = last_layer_channels_number,
                                         out_channels = conv[0],
                                         kernel_size = conv[1],
                                         padding = tuple([int((x - 1) / 2) for x in conv[1]])))
            last_layer_channels_number = conv[0]
            
            encoder_layers.append(nn.BatchNorm2d(last_layer_channels_number))
            encoder_layers.append(nn.ReLU())
            if conv_drop_rate is not None:
                encoder_layers.append(nn.Dropout2d(conv_drop_rate))
            if conv[2]:
                encoder_layers.append(nn.MaxPool2d(2, 2))
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        
        encoder_output_shape = self.encoder_layers(torch.randn(1, *input_shape)).shape[1:]
        
        """
            Layers:
                1. Linear
                2. BatchNorm1d
                3. ReLU
                4. Dropout if 'fc_drop_rate' is not None
        """
        
        fc_layers = []
        
        fc_layers.append(nn.Flatten())
        encoder_output_nodes_number = encoder_output_shape[0] * encoder_output_shape[1] * encoder_output_shape[2]
        last_nodes_number = encoder_output_nodes_number
        
        for nodes_number in fcs:
            fc_layers.append(nn.Linear(last_nodes_number, nodes_number))
            last_nodes_number = nodes_number
            fc_layers.append(nn.BatchNorm1d(last_nodes_number))
            fc_layers.append(nn.ReLU())
            if fc_drop_rate is not None:
                fc_layers.append(nn.Dropout(fc_drop_rate))
        
        fc_layers.append(nn.Linear(last_nodes_number, encoder_output_nodes_number))
        fc_layers.append(nn.ReLU())
        fc_layers.append(Reshape(encoder_output_shape))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        
        
        decoder_layers = []
        
        for conv in decoder_convs:
            """
                Layers:
                    1. Upsample if has_upsample is true
                    2. Conv2d
                    3. BatchNorm2d
                    4. ReLU
                    5. Dropout if 'conv_drop_rate' is not None
            """
            if conv[2]:
                decoder_layers.append(nn.Upsample(scale_factor = 2))
            decoder_layers.append(nn.Conv2d(in_channels = last_layer_channels_number,
                                            out_channels = conv[0],
                                            kernel_size = conv[1],
                                            padding = tuple([int((x - 1) / 2) for x in conv[1]])))
            last_layer_channels_number = conv[0]
            decoder_layers.append(nn.BatchNorm2d(last_layer_channels_number))
            decoder_layers.append(nn.ReLU())
            if conv_drop_rate is not None:
                decoder_layers.append(nn.Dropout2d(conv_drop_rate()))
            
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        
        encoded_data = self.encoder_layers(x)
        processed_encoded_data = self.fc_layers(encoded_data)
        decoded_data = self.decoder_layers(processed_encoded_data)
        return decoded_data
    
    def get_loss(self, outputs, targets):
        loss = self.mse_loss(outputs, targets)
        return loss
    
    def summary(self, input_shape):
        torch_summary(self, input_shape)
        


class Reshape(nn.Module):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape
    
    def forward(self, x):        
        return x.view(-1, *self.new_shape)



def main():

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
    
    input_shape = (1, 256, 256)
    
    auto_encoder = AutoEncoder(encoder_convs, fc_layers, decoder_convs, input_shape)
    
    auto_encoder.to(torch.device('cuda:0'))

    auto_encoder.summary(input_shape)


if __name__ == "__main__":
    main()          
            