import pandas as pd
import os
from ftfy import fix_encoding


def get_position_image(path_image):
    img = path_image.split('/')[-1]
    set_name_image = list(os.listdir('D:\Study\DoAnTruyVan\static\data_img\image'))
    return set_name_image.index(img)


def get_in4_from_positon(i):
    data = pd.read_csv("static/data_img/data.csv")
    name_product = list(data['name_product'])
    link = list(data['link'])
    price = list(data['price'])
    try:
        fix_name = fix_encoding(name_product[i])
        name_product[i] = fix_name
    except:
        None
    return name_product[i], link[i], str(f'{int(price[i]):n}') + ' Đồng'


# i = get_locate_image('static/data_img/image/1588_00002.jpg')
# data = pd.read_csv("static/data_img/data.csv")
# name_img = list(data['name_img'])

# a, b, c = get_in4_from_positon(0)
# print(a, b, c)