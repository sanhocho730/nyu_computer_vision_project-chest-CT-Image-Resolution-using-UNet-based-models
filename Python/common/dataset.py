from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import os
import numpy as np

class ChestCTScanDataset(Dataset):
    def __init__(self, data_root, transformations, is_training_set=True):
        # Define the subdirectory based on whether it's training or testing data
        directory = 'train/' if is_training_set else 'test/'
        self.path = os.path.join(data_root, directory)

        # Define the classes (or types of images) in your dataset
        self.classes = ['Adenocarcinoma', 'Large cell carcinoma', 'Squamous cell carcinoma', 'Normal']
        self.input_paths = []

        # Load all images from each class
        for cls in self.classes:
            class_path = os.path.join(self.path, cls)
            print(class_path)
            self.input_paths.extend(glob(os.path.join(class_path, '*')))
        print("Number of images found:", len(self.input_paths))
        self.transformations = transformations

    def __getitem__(self, index):
        # Get the path of the specific image
        img_path = self.input_paths[index]

        # Open the image
        img = Image.open(img_path).convert('RGB')


        # Apply transformations if any
        if self.transformations:
            img = self.transformations(img)

        # In this case, the image is both the input and the label
        sample = {'image': np.array(img), 'label': np.array(img)}

        return sample

    def __len__(self):
        # Return the total number of images in the dataset
        return len(self.input_paths)