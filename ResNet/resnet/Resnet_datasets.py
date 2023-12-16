import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import cv2 as cv

class vehicledata(Dataset):
    def __init__(self,positive_dir,negative_dir):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128,128)),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        self.image_path=[]
        self.type = []
        positive_file_names = os.listdir(positive_dir)
        negative_file_names = os.listdir(negative_dir)
        for file_name in positive_file_names:
            self.type.append(np.int32(1))
            self.image_path.append(os.path.join(positive_dir,file_name))
        for file_name in negative_file_names:
            self.type.append(np.int32(0))
            self.image_path.append(os.path.join(negative_dir,file_name))

    def __len__(self):
        return len(self.image_path)

    def output_nums(self):
        return len(self.image_path)

    def num_of_samples(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_pa = self.image_path[idx]
        else:
            image_pa = self.image_path[idx]
        img = cv.imread(image_pa)  # BGR order
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        sample = {'image': self.transform(img), 'type': self.type[idx]}
        return sample


if __name__ == "__main__":
    ds = vehicledata("../concerete_crack_images/training/Positive","../concerete_crack_images/training/Negative")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['type'])
        if i == 3:
            break

    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    # data loader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['type'])
        break