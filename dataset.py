import torch
from torch import nn
import os
from torch.utils.data import Dataset
import numpy as np
import h5py
import torchvision.transforms as transforms

class ImagesDataset(Dataset):


  def __init__(self, kind):
    self.kind = kind
    self.img_path = 'FFHQ/thumbnails128x128'
    self.emb_path = 'FFHQ.hdf5'

    self.clip_enc = np.array(h5py.File(self.emb_path, "r")['image'])
    self.toTensor = transforms.ToTensor()
    self.transform = nn.Sequential(
        transforms.Resize((128, 128), antialias=True),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    )
    self.set_dataset()

  def set_dataset(self):
    # N = 70000
    # dataset = np.empty((N, 3, 128, 128), dtype=np.uint8)

    # for idx in range(N):
    #   img_path = self.get_image_path(idx)
    #   img = Image.open(img_path)
    #   img = img.convert('RGB')
    #   img = self.toTensor(img)
    #   img = self.transform(img)
    #   img = img[:, :, :]
    #   dataset[idx, ...] = img

    # self.dataset = torch.tensor(dataset)
    self.dataset = torch.load('/FFHQ/ffhq.pt', map_location=device)

  def __len__(self):
    if self.kind == 'train':
      return 60000
    
    # self.kind == test
    return 10000

  def get_image_path(self, idx):
    return os.path.join(self.img_path, format(idx, '05d') + '.png')


  def __getitem__(self, idx):
    if self.kind == 'test':
      idx += 60000

    img = self.dataset[idx]
    img_emb = self.clip_enc[idx]
    
    return img, torch.tensor(img_emb)
