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

    mlp = []
    c_hid = [1024, 512, 512, 512, 512, 512]

    for i in range(4):
      mlp.extend([
          nn.Linear(c_hid[i], c_hid[i+1]),
          nn.ReLU(),
      ])

    self.convnet = nn.Sequential(*convnet)
    self.mlp = nn.Sequential(*mlp)
    self.to_mu = nn.Linear(512, 256)
    self.to_var = nn.Linear(512, 256)

  def forward(self, x, clip_enc):
    x = self.convnet(x)
    x = torch.cat([x, clip_enc], dim=-1)
    x = self.mlp(x)

    return self.to_mu(x), self.to_var(x)

    