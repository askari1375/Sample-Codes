# -*- coding: utf-8 -*-
"""
Created on Wed May 12 06:30:07 2021

@author: Amirhossein
"""

import os
import shutil
import subprocess


os.system("scrapy startproject searchingfortruth")

print("scrapy project created")

if not os.path.exists("searchingfortruth/temporary files"):
    os.mkdir("searchingfortruth/temporary files")

if not os.path.exists("searchingfortruth/temporary files/scrapy results"):
    os.mkdir("searchingfortruth/temporary files/scrapy results")

print("extracting content links ...")

os.system("python link_extractor.py")


shutil.copy("crawler.py", "searchingfortruth/searchingfortruth/spiders")
shutil.copy("article.py", "searchingfortruth")



with open("searchingfortruth/temporary files/content links.txt") as f:
    links = f.readlines()
total_links = len(links)


print("content links extracted")
print("total links : {}".format(total_links))

print("Please wait ...")
print("you can check searchingfortruth/temporary files/scrapy results to find out that how many file from {} files created".format(total_links))

p = subprocess.Popen(["scrapy", "crawl", "tuserkani"], cwd = "searchingfortruth/")
p.wait()

print("all pages are crawled")

if not os.path.exists("results"):
    os.mkdir("results")

os.system("python create_tex_file.py")

print("Latex file created")
