# -*- coding: utf-8 -*-
"""
Created on Sun May  9 23:58:55 2021

@author: Amirhossein
"""

import scrapy
import json
from article import ScrapyArticle


class Myspider(scrapy.Spider):
    
    name = 'tuserkani'
    
    def __init__(self):
        super().__init__()
        
        self.content_urls = None   
        
        self.counter = 0
        
        
        with open("temporary files/content links.txt") as f:
            self.content_urls = f.readlines()
        
        with open("temporary files/content links categories.txt") as f:
            raw_content_categories = f.read()
        self.content_categories = raw_content_categories.split("\n")
        
        
        
        
        #self.content_urls = self.content_urls[:3] #########################
        
        
    
    
    def start_requests(self):
        
        for url in self.content_urls:
            yield scrapy.Request(url=url, callback=self.my_parse)
            
    def my_parse(self, response):
        
        article = ScrapyArticle()
        
        x_path_title = "//h1/text()"
        
        x_path_text = "//article//section[position()>1 and position()<4]//text()"
        x_path_h1 = "//article//section[position()>1 and position()<4]//h1//text()"
        x_path_h2 = "//article//section[position()>1 and position()<4]//h2//text()"
        x_path_h3 = "//article//section[position()>1 and position()<4]//h3//text()"
        x_path_h4 = "//article//section[position()>1 and position()<4]//h4//text()"
        x_path_h5 = "//article//section[position()>1 and position()<4]//h5//text()"
        
        article.text_list = response.xpath(x_path_text).getall()
        article.h1_list = response.xpath(x_path_h1).getall()
        article.h2_list = response.xpath(x_path_h2).getall()
        article.h3_list = response.xpath(x_path_h3).getall()
        article.h4_list = response.xpath(x_path_h4).getall()
        article.h5_list = response.xpath(x_path_h5).getall()
        
        article.title = response.xpath(x_path_title).get()
        article.url = response.url
        article.category = self.content_categories[self.counter]
        
        json_article = json.dumps(article.__dict__)
        
        print(" ++++++++++++++++++++++++++++ {} ++++++++++++++++++++++++++".format(self.counter + 1))
        with open("temporary files/scrapy results/{}.txt".format(self.counter + 1), "a", encoding = 'utf-8') as f:
            f.write(json_article)
        self.counter += 1    
            







