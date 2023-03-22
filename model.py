from torch import nn
import torch


class encoder(nn.Module):


  def __init__(self):
    super().__init__()
    convnet = []
    c_hid = [3, 32, 64, 128, 256, 512]

    for i in range(5):
      convnet.extend([
        nn.Conv2d(c_hid[i], c_hid[i+1], 3, stride=2, padding=1),
        nn.BatchNorm2d(c_hid[i+1]),
        nn.LeakyReLU()
      ])

    convnet.append(nn.Flatten())
    convnet.append(nn.Linear(8192, 512))


    