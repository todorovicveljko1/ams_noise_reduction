import pandas as pd
import os
import random
from PIL import Image
from tqdm import tqdm
from tqdm import tqdm

# TODO: Should we add noise here and then crop them and save noise images?
# Simplest 
class DatasetGen:
    def __init__(self, csv_file, output_dir, n=10, img_size=224):
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.n = n
        self.isof = img_size//2 # image_size_offset
        self.df = pd.read_csv(csv_file)
    
    def gen(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            img = Image.open(row['file_name'])
            img_name = row['file_name'].split('\\')[-1].split('.')[0]
            for i in range(self.n):
                x = random.randint(row['x_min'], row['x_max'])
                y = random.randint(row['y_min'], row['y_max'])
                x_min = x-self.isof
                y_min = y-self.isof
                x_max = x+self.isof
                y_max = y+self.isof
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                cropped_img.save(os.path.join(self.output_dir, img_name + '_' + str(x_min//32) + '_' + str(y_min//32) + '.png'))
    def split_train_val_test(self, train_ratio=0.8, val_ratio=0.1):
        # read files inside output_dir and move them into train, val, test folders
        files = os.listdir(self.output_dir)
        random.shuffle(files)
        train_files = files[:int(len(files)*train_ratio)]
        val_files = files[int(len(files)*train_ratio):int(len(files)*(train_ratio+val_ratio))]
        test_files = files[int(len(files)*(train_ratio+val_ratio)):]
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        test_dir = os.path.join(self.output_dir, 'test')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        for file in tqdm(train_files):
            os.rename(os.path.join(self.output_dir, file), os.path.join(train_dir, file))
        for file in tqdm(val_files):
            os.rename(os.path.join(self.output_dir, file), os.path.join(val_dir, file))
        for file in tqdm(test_files):
            os.rename(os.path.join(self.output_dir, file), os.path.join(test_dir, file))



if __name__ == '__main__':
    dataset_gen = DatasetGen('data/dataraw/dataraw.csv', 'data\\dataset', 50)
    dataset_gen.gen()
    dataset_gen.split_train_val_test()