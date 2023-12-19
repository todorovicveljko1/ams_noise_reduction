# Create csv with pandas from data in data/{name} folder
# Output: data/{name}.csv
# data/{name} folder has two subfolders: images and masks
# images {img}.png and masks {img}_mask.png
# csv should have file name, width, height, sample range,
# sample range is obtained from the mask we take the min and max x,y values 
# also we need to take into account the padding we add to the images mininum is 112 from the edges of image so min max x y are clamped to be not coloser then 112 pixels to the edge

import os
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def gen_csv(name, padding=112, use_mask=True):
    """
    Generate csv file from data in data/{name} folder
    name: name of the folder in data
    padding: minimum distance from the edge of the image
    use_mask: use mask to determine the sample range
    """
    data_path = Path('data')
    name_path = data_path / name
    images_path = name_path / 'images'
    masks_path = name_path / 'masks'
    csv_path = name_path / (name + '.csv')

    images = os.listdir(images_path)

    df = pd.DataFrame(columns=['file_name', 'width', 'height', 'x_min', 'y_min', 'x_max', 'y_max'])

    for img in tqdm(images):
        img_path = images_path / img
        mask_path = masks_path / img.replace('.png', '_mask.png')
        img = Image.open(img_path)
        width, height = img.size
        # maybe we should check if mask exists
        if not use_mask:
            df = df._append({'file_name': img_path, 
                             'width': width, 
                             'height': height, 
                             'x_min': padding, 
                             'y_min': padding, 
                             'x_max': width - padding, 
                             'y_max': height - padding}, ignore_index=True)
            continue
        # use mask
        try:
            mask = Image.open(mask_path)
        except FileNotFoundError:
            print(f'No mask for {img_path}')
            continue
        img = np.array(img)
        mask = np.array(mask)

        x_min_args = np.argmax(mask, axis=1) 
        x_max_args = np.argmax(np.flip(mask, axis=1), axis=1)
        y_min_args = np.argmax(mask, axis=0)
        y_max_args = np.argmax(np.flip(mask, axis=0), axis=0)

        x_min_mask = np.where(x_min_args > 0, x_min_args, 100000).min()
        y_min_mask = np.where(y_min_args > 0, y_min_args, 100000).min()
        x_max_mask = width - np.where(x_max_args > 0, x_max_args, 100000).min()
        y_max_mask = height - np.where(y_max_args > 0, y_max_args, 100000).min()
        
        # clamp the values to be at least 112 pixels from the edge
        x_min = max(padding, x_min_mask)
        y_min = max(padding, y_min_mask)
        x_max = min(width - padding, x_max_mask)
        y_max = min(height - padding, y_max_mask)

        # 'DataFrame' object has no attribute 'append'.
        df = df._append({'file_name': img_path, 
                         'width': width, 
                         'height': height, 
                         'x_min': x_min, 
                         'y_min': y_min, 
                         'x_max': x_max, 
                         'y_max': y_max}, ignore_index=True)
        #show_img_mask_rect(img, mask, x_min, y_min, x_max, y_max)
        #break
    df.to_csv(csv_path, index=False)

def show_img_mask(img_path, mask_path):
    """
    Show image and mask
    """
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    plt.imshow(img / 2 + mask / 2)
    plt.show()

def show_img_mask_rect(img, mask, x_min, y_min, x_max, y_max):
    """
    Show image and mask with rectangle
    """
    plt.imshow(img / 2 + mask / 2)
    plt.plot([x_min, x_min], [y_min, y_max])
    plt.plot([x_max, x_max], [y_min, y_max])
    plt.plot([x_min, x_max], [y_min, y_min])
    plt.plot([x_min, x_max], [y_max, y_max])
    plt.show()

if __name__ == '__main__':
    gen_csv('dataraw')