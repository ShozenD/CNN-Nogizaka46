import httplib2
import json
import os 
import urllib.request
import glob
import shutil
from urllib.parse import quote

# Basic Setup
API_KEY='YOUR_API_KEY'
CUSTOM_SEARCH_ENGINE='YOUR_CUSTOM_SEARCH_ENGINE'
KEYWORDS=["生田絵梨花","齋藤飛鳥","白石麻衣","西野七瀬","橋本奈々未"]
NUM_OF_IMAGES=100 # Will Error if more than 100 

# Function: Obtain Image Url via Google Custom Search API
def getImageUrl(search_item: list, total_num: int):
    img_list = []
    i = 0
    while i < total_num:
        query_img = "https://www.googleapis.com/customsearch/v1?key=" + API_KEY + "&cx=" + CUSTOM_SEARCH_ENGINE + "&num=" + str(10 if(total_num-i)>10 else (total_num-i)) + "&start=" + str(i+1) + "&q=" + quote(search_item) + "&searchType=image"
        res = urllib.request.urlopen(query_img)
        data = json.loads(res.read().decode('utf-8'))
        for j in range(len(data['items'])):
            img_list.append(data['items'][j]['link'])
        i=i+10
    return img_list

# Function: Obtain Images 
def getImage(search_item: list, img_list: list, base_dir_name='Images'):
    os.mkdir(base_dir_name) # create base dir
    item_dir = os.path.join(base_dir_name, search_item)
    os.mkdir(item_dir) # directory to house the images
    http = httplib2.Http(".cache") # Initiate http request object instance 
    for i in range(len(img_list)):
        try:
            response, content = http.request(img_list[i])
            filename = os.path.join(item_dir, search_item + '.' + str(i) + '.jpg')
            with open(filename, 'wb') as f:
                f.write(content)
        except:
            print('Error: failed to download image')
            continue

# Obtain Images
for j in range(len(KEYWORDS)):
    print('=== downloading images for {} ==='.format(KEYWORDS[j]))
    img_list=getImageUrl(KEYWORDS[j],NUM_OF_IMAGES)
    getImage(KEYWORDS[j], img_list)