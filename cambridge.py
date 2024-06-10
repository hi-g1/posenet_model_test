import os
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils import data

# data loader for cambridge dataset
class CambridgeDataset(data.Dataset):

    def __init__(self, dataset_root, mode='train', transform=None):
        super(CambridgeDataset, self).__init__()

        self.dataset_root = dataset_root
        self.mode = mode
        self.transform = transform

        self.image_filenames = []
        self.trs = []
        self.rots = []

        file_path = os.path.join(self.dataset_root, 'dataset_' + self.mode + '.txt')

        # read txt file
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # get image path and label
            # skip first there lines
            for line in lines[3:]:
                line_split = line.split(' ')

                image_path = os.path.join(self.dataset_root, line_split[0])
                x = float(line_split[1])
                y = float(line_split[2])
                z = float(line_split[3])
                w = float(line_split[4])
                p = float(line_split[5])
                q = float(line_split[6])
                r = float(line_split[7])

                tr = [x, y, z]
                rot = [w, p, q, r]

                self.image_filenames.append(image_path)
                self.trs.append(tr)
                self.rots.append(rot)

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index])
        tr = torch.tensor(self.trs[index], dtype=torch.float32)
        rot = torch.tensor(self.rots[index], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, tr, rot
        

            



