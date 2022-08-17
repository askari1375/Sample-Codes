# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 18:51:00 2021

@author: Amirhossein
"""


def create_model_name(num_classes,
                      lstm_nodes,
                      final_fcs,
                      fc_drop_rate):
    
    
    model_name = "alexnet_backbone_"
    
    model_name += "_{}__{}_".format(num_classes, lstm_nodes)
    
    for k in final_fcs:
        model_name += "_{}".format(k)
    model_name += "_"
    
    
    if fc_drop_rate is None:
        model_name += "_0"
    else:
        model_name += "_{}".format(str(fc_drop_rate)[2:])
        
    
    model_name += ".pt"
    return model_name


def main():
    
    
    num_classes = 2
    lstm_nodes = 256
    final_fcs = [64, 125, 50]
    fc_drop_rate = 0.54321
    
    
    model_name = create_model_name(num_classes,
                                   lstm_nodes,
                                   final_fcs,
                                   fc_drop_rate)
    
    print(model_name)



if __name__ == "__main__":
    main()






    