# -*- coding: utf-8 -*-
"""
Created on Mon May 10 05:08:31 2021

@author: Amirhossein
"""

from requests import get
from lxml import html
from urllib.parse import urljoin


def download_url(url):
    try:
        response = get(url)
        return response.text
    except:
        print("An exception occurred")
        

def extract_xpath(data, x_path):        
    tree = html.fromstring(data)
    return tree.xpath(x_path)


def extract_category_list_urls(start_url, x_path):
    main_page_data = download_url(start_url)
    res = extract_xpath(main_page_data, x_path)
    return res
        
        
def extract_content_list_page_url(category_list_urls, category_names, x_path):
    res = []
    category = []
        
    for idx in range(len(category_list_urls)):
        url = category_list_urls[idx]
        page_data = download_url(url)
        last_page_number = int(extract_xpath(page_data, x_path)[0])
            
        for k in range(last_page_number):
            new_url = urljoin(url, "page/{}/".format(k + 1))            
            res.append(new_url)
            category.append(category_names[idx])
    
    return res, category


def extract_content_urls(content_list_page_url, page_categories, x_path):
    res = []
    categories = []
    n = len(content_list_page_url)
    for idx in range(n):
        url = content_list_page_url[idx]
        page_data = download_url(url)
        url_list = extract_xpath(page_data, x_path)
        category_list = [page_categories[idx]] * len(url_list)
        res += url_list
        categories += category_list
        print("step {} / {}".format(idx, n))
    return res, categories



category_list_xpath = "//*[@id='site-navigation']/ul/li[position()>1 and position()<6]/a/@href"
last_page_number_xpath = "(//li/a[@class='page-numbers'])[last()]/node()"
content_url_xpath = "//article//h3/a/@href"
        
start_url = 'https://searchingfortruth.ir/'


category_list_urls = None
content_list_page_url = None
content_urls = None

category_names = ["quran", "defaa", "ravankavi", "sayer"]
category_list_urls = extract_category_list_urls(start_url,
                                                category_list_xpath)

content_list_page_url,  category_content_list_page_url= extract_content_list_page_url(category_list_urls,
                                                                                      category_names,
                                                                                      last_page_number_xpath)
content_urls, content_categories = extract_content_urls(content_list_page_url,
                                                        category_content_list_page_url,
                                                        content_url_xpath)
            


with open("searchingfortruth/temporary files/content links.txt", "w") as f:
    for url in content_urls:
        f.write(url + "\n")

with open("searchingfortruth/temporary files/content links categories.txt", "w") as f:
    for name in content_categories:
        f.write(name + "\n")

print("\ntotal links : {}".format(len(content_urls)))


