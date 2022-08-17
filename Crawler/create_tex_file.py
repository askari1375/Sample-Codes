# -*- coding: utf-8 -*-
"""
Created on Tue May 11 02:11:46 2021

@author: Amirhossein
"""

from article import ScrapyArticle
import json


def load_article(k):

    with open("searchingfortruth/temporary files/scrapy results/{}.txt".format(k), "r") as f:
        json_data = f.read()
    

    article_dict = json.loads(json_data)
    article = ScrapyArticle()
    article.load_from_dict(article_dict)
    
    return article


def remove_empty_elements_of_list(elements_list):
    not_empty_elements_list = []
    for e in elements_list:
        if len(e) > 0:
            not_empty_elements_list.append(e)
    elements_list = not_empty_elements_list


def remove_empty_elements(article):
    remove_empty_elements_of_list(article.h1_list)
    remove_empty_elements_of_list(article.h2_list)
    remove_empty_elements_of_list(article.h3_list)
    remove_empty_elements_of_list(article.h4_list)
    remove_empty_elements_of_list(article.h5_list)
    remove_empty_elements_of_list(article.text_list)


"""annotation_keys = 0 : text
                    1 : h1
                    1 : h2
                    1 : h3
                    1 : h4
                    1 : h5"""


def find_and_assign_annotation(text, header, annotations, label):
    
    text_pointer = 0
    header_pointer = 0
    
    while header_pointer < len(header) and text_pointer < len(text):
        if text[text_pointer] == header[header_pointer]:
            annotations[text_pointer] = label
            text_pointer += 1
            header_pointer += 1
        else:
            text_pointer += 1

def create_annotaions(article):
    annotations = []
    for k in range(len(article.text_list)):
        annotations.append(0)
    
    find_and_assign_annotation(article.text_list,
                               article.h1_list,
                               annotations, 2)
    
    find_and_assign_annotation(article.text_list,
                               article.h2_list,
                               annotations, 3)
    
    find_and_assign_annotation(article.text_list,
                               article.h3_list,
                               annotations, 4)
    
    find_and_assign_annotation(article.text_list,
                               article.h4_list,
                               annotations, 5)
    
    find_and_assign_annotation(article.text_list,
                               article.h5_list,
                               annotations, 6)
    
    return annotations


def ctrate_annotated_text_list(article):
    
    if type(article.title) is not list:
        title_list = [article.title]
    else:
        title_list = article.title
    
    annotated_text = []
    for title_element in title_list:        
        annotated_text.append((1, title_element))
    
    annotations = create_annotaions(article)
    
    for k, t in enumerate(article.text_list):
        annotated_text.append((annotations[k], t))
    
    return annotated_text
    

def remove_just_t_n(annotated_list):
    res = []
    for annotation, e in annotated_list:
        list_e = list(e)
        t_num = 0
        n_num = 0
        other_num = 0
        for c in list_e:
            if c == "\t":
                t_num += 1
            elif c == "\n":
                n_num += 1
            else:
                other_num += 1
        
        if other_num == 0:
            if n_num > 0:
                res.append((annotation, "\n"))
            else:
                res.append((annotation, "\t"))
        else:
            res.append((annotation, e))
    
    return res


def merge_elements(annotated_list):
    res = []
    previous_label = -1
    for label, e in annotated_list:
        
        if label == 0:
            if e[-1] != "\n":
                e += "\n"            
        
        if label != previous_label:
            res.append([label, e])
            previous_label = label
        else:
            previous_e = res[-1][1]
            if e == "\n":
                if previous_e[-1] != "\n":
                    res[-1] = [label, previous_e + e]
            elif e == "\t":
                if previous_e[-1] != "\t":
                    res[-1] = [label, previous_e + e]
            else:
                res[-1] = [label, previous_e + e]
                if label == 0:
                    res[-1] = [label, res[-1][1] + "\n"]
    
    for k in range(len(res)):
        res[k] = (res[k][0], res[k][1])
    
    return res

def normalaize_headers(annotated_list):
    
    res = []
    annotation_set = set()
    for e in annotated_list:
        annotation_set.add(e[0])
    
    annotation_map = [-1] * (max(annotation_set) + 1)
    
    pointer = 0
    for k in range(max(annotation_set) + 1):
        if k in annotation_set:
            annotation_map[k] = pointer
            pointer += 1
    
    for label, e in annotated_list:
        new_element = (annotation_map[label], e)
        res.append(new_element)
        
    return res

def remove_numbers_from_headers(annotated_list):
    
    res = []
    for label, e in annotated_list:
        if label > 1:
            k = 0
            c = e[k]
            while c.isdigit() or c == "-" or c ==" ":                
                if k < len(e) - 1:
                    k += 1
                    c = e[k]
                else:
                    break
            e = e[k:]
                
        res.append((label, e))
    
    return res

def preprocess_loaded_article(article):
    annotated_text = ctrate_annotated_text_list(article)
    annotated_text_removed = remove_just_t_n(annotated_text)
    annotated_merged = merge_elements(annotated_text_removed)
    annotated_normalized = normalaize_headers(annotated_merged)
    annotated_removed_numbers_in_header = remove_numbers_from_headers(annotated_normalized)
    return annotated_removed_numbers_in_header

class Article:
    
    def __init__(self, article_number):
        article = load_article(article_number)
        self.annotated_data = preprocess_loaded_article(article)
        self.category = article.category
        self.url = article.url
        self.title = article.title
        
        
        self.sorting_key = self.find_sorting_key()


        
    def find_sorting_key(self):
        
        len_part_number = 3
        len_name = 10
        
        
        order = ("quran", "defaa", "ravankavi", "sayer")
        order_dict = dict(zip(order, list(range(len(order)))))
        
        digits_list = [s for s in list(str(self.url)) if s.isdigit()]
        _3_digits = ["0"] * len_part_number
        for k in range(min(len_part_number, len(digits_list))):
            _3_digits[len_part_number - 1 - k] = str(digits_list[-1 * (k + 1)])
        part_number_str = "".join(_3_digits)
        if int(part_number_str) > 100:
            part_number_str = "0" * len_part_number
        
        name = ((self.url.split("/")[3]).split("-")[0] + "0" * len_name)[:len_name]
        
        
        priority_list = []
        priority_list.append(str(order_dict[self.category]))
        priority_list.append(name)
        priority_list.append(part_number_str)
        priority_list.append(str(self.url))
        
        sort_key = " ".join(priority_list)
        
        return sort_key


def find_total_articles_number():
    with open("searchingfortruth/temporary files/content links categories.txt", "r") as f:
        data = f.readlines()
    return len(data)


def load_articles():
    articles = []

    articles_total_number = find_total_articles_number()

    for k in range(articles_total_number):
        articles.append(Article(k + 1))
    
    return articles


def sort_articles(articles):    
    
    articles.sort(key = lambda x: x.sorting_key)


def single_article_latex_code(article):
    
    annotation_dict = {1 : "\\section{",
                       2 : "\\subsection{",
                       3 : "\\subsubsection{",
                       4 : "\\paragraph{",
                       5 : "\\paragraph{"}
    
    annotated_data = article.annotated_data
    res = ""    
    
    for label, data in annotated_data:
        if label in annotation_dict:
            res += annotation_dict[label] + data + "}\n"
        else:
            res += data + "\n\n"
    
    return res

def modify_extra_chars(text):
    
    text = text.replace("_", " ")
    text = text.replace("&", " and ")
    return text
            


def create_latex(articles):
    
    with open("start_part.txt", "r", encoding="utf-8") as f:
        start_part = f.read()
    with open("end_part.txt", "r") as f:
        end_part = f.read()
        
    previous_chapter = ""
    chapter_pointer = 0
    chapters_list = ["قرآن", "دفاع عقلانی از دین", "روانکاوی و فرهنگ", "سایر موضوعات"]
    
    
    latex_code = ""
    for article in articles:
        
        if article.category != previous_chapter:
            latex_code += "\\chapter{" + chapters_list[chapter_pointer] + "}\n\\newpage\n"
            chapter_pointer += 1
            previous_chapter = article.category
        
        article_tex = single_article_latex_code(article)
        latex_code += article_tex + "\n"
    
    latex_code = start_part + latex_code + end_part
    
    latex_code = modify_extra_chars(latex_code)
    
    return latex_code    
        
    

articles = load_articles()
sort_articles(articles)
latex_code = create_latex(articles)


with open("results/DrTuserkani.tex", "w", encoding="utf-8") as f:
    f.write(latex_code)








