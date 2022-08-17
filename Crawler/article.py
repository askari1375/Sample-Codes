# -*- coding: utf-8 -*-
"""
Created on Mon May 10 08:24:57 2021

@author: Amirhossein
"""

class ScrapyArticle:
    
    def __init__(self):
        self.text_list = None
        self.h1_list = None
        self.h2_list = None
        self.h3_list = None
        self.h4_list = None
        self.h5_list = None
        
        self.url = None
        self.title = None
        self.category = None
        
    
    def load_from_dict(self, data):
        for key, value in self.__dict__.items():
            setattr(self, key, data[key])