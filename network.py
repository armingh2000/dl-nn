from model import *
from loss import *
from dataset import *
import torch
from torch import nn
import clip
import pickle
import functools
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.distributions.dirichlet import Dirichlet
import matplotlib.pyplot as plt
import random
import sys
import lpips

sg2_path = 'stylegan2-ada-pytorch'
sys.path.append(sg2_path )

ckpt = torch.load('iteration_500000.pt', map_location=device)
device = "cuda" if torch.cuda.is_available() else "cpu"

class network():


  def __init__(self):
    self.set_network()
    self.set_optimizer()
    self.set_dataset()
    self.set_loss_fns()
    self.latent_avg = ckpt['latent_avg']
    self.set_transforms()
    self.set_save_paths()

  def set_save_paths(self):
    self.encoder_path = 'encoder4.pt'
    self.decoder_path = 'decoder4.pt'
    
  def set_network(self):
    self.encoder = encoder().to(device)
    
    clip_model, preprocess = clip.load("ViT-B/16", device)
    clip_model = clip_model.eval()
    self.clip = clip_model
    self.preprocess = preprocess

    self.decoder = decoder().to(device)

    with open('ffhq.pkl', 'rb') as f:
      G = pickle.load(f)['G_ema'].eval()

    G.forward = functools.partial(G.forward, force_fp32=True)
    self.sg2 = G.to(device)

    self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

  def set_optimizer(self):
    params = list(
        filter(lambda p: p.requires_grad, self.encoder.parameters()))
    params.extend(
        list(filter(lambda p: p.requires_grad, self.decoder.parameters())))

    optimizer = torch.optim.Adam(params, lr=6e-4)
    
    self.optimizer = optimizer

  def set_dataset(self):
    self.train_dataset = ImagesDataset('train')
    self.test_dataset = ImagesDataset('test')
    self.train_data_loader = DataLoader(self.train_dataset,
                                        batch_size=4,
                                        shuffle=True,
                                        drop_last=True)
    self.test_data_loader = DataLoader(self.test_dataset,
                                       batch_size=4,
                                       shuffle=False,
                                       drop_last=True)

  def set_loss_fns(self):
    self.p_var = torch.ones(256, device=device)
    self.p_mu = torch.zeros(256, device=device)

    self.latent_avg = ckpt['latent_avg']

    self.kl_loss = KLDivLoss().eval().to(device)
    self.clip_loss = CLIPLoss().eval().to(device)
    self.w_norm_loss = WNormLoss().eval().to(device)
    self.lpips_loss = lpips.LPIPS(net='alex').eval().to(device)

  def loss_fn(self, x, x_hat128, x_hat256, latent, mu, log_var, clip_enc):
    loss_dict = {}
    loss = 0.0

    lpips_lambda, w_norm_lambda, clip_lambda, KL_lambda = 1.0, 2e-4, 1.0, 0.2

    loss_lpips = self.lpips_loss(x_hat128, x)
    loss_lpips = loss_lpips.squeeze()
    loss_lpips = loss_lpips.mean()
    loss_dict['loss_lpips'] = float(loss_lpips)
    loss += loss_lpips * lpips_lambda
    
    loss_w_norm = self.w_norm_loss(latent, self.latent_avg)
    loss_dict['loss_w_norm'] = float(loss_w_norm)
    loss += loss_w_norm * w_norm_lambda
    
    loss_clip = self.clip_loss(x_hat256, clip_enc)
    loss_dict['loss_clip'] = float(loss_clip)
    loss += loss_clip * clip_lambda
    
    assert mu is not None and log_var is not None

    loss_kl = self.kl_loss(mu, log_var, self.p_mu, self.p_var)
    loss_dict['loss_kl'] = float(loss_kl)
    loss += loss_kl * KL_lambda

    loss_dict['loss'] = float(loss)

    return loss, loss_dict

  def set_transforms(self):
    self.transform256 = nn.Sequential(
        transforms.Resize((256, 256), antialias=True),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    )

    self.transform128 = nn.Sequential(
        transforms.Resize((128, 128), antialias=True),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    )

  def forward(self, x, clip_enc):
    mu, log_var = self.encoder(x, clip_enc)
    delta = self.decoder(mu, log_var, clip_enc)
    w = delta
    image = self.sg2(w, c=None, truncation_psi=0.5, noise_mode='const')
    image128 = self.transform128(image)
    image256 = self.transform256(image)

    return image128, image256, mu, log_var, w

  def train_mode(self):
    self.encoder.train()
    self.decoder.train()

  def train(self):
    self.train_mode()
    step = 0
    
    for i in range(5):
      for x, clip_enc in tqdm(self.train_data_loader, position=0, leave=True):
        x, clip_enc = x.to(device).float(), clip_enc.to(device).float()
        x_hat128, x_hat256, mu, log_var, latent = self.forward(x, clip_enc)

        self.optimizer.zero_grad()
        loss, loss_dict = self.loss_fn(x, x_hat128, x_hat256, latent, mu, log_var, clip_enc)
        loss.backward()
        self.optimizer.step()

        step += 1

        if (step + 1) % 2500 == 0:
          self.validate(step + 1)


    self.save()
      
  def save(self):
    torch.save(self.encoder.state_dict(), self.encoder_path)
    torch.save(self.decoder.state_dict(), self.decoder_path)

  def eval_mode(self):
    self.encoder.eval()
    self.decoder.eval()

  def validate(self, step):
    self.eval_mode()
    losses = []
    
    with torch.no_grad():
      for x, clip_enc in tqdm(self.test_data_loader, position=0, leave=True):
        x, clip_enc = x.to(device).float(), clip_enc.to(device).float()
        x_hat128, x_hat256, mu, log_var, latent = self.forward(x, clip_enc)

        loss, loss_dict = self.loss_fn(x, x_hat128, x_hat256, latent, mu, log_var, clip_enc)

        losses.append(loss)

      print(f'validate loss at step {step}: {sum(losses) / len(losses)}')


  def load(self):
    self.encoder.load_state_dict(torch.load(self.encoder_path, map_location=device))
    self.decoder.load_state_dict(torch.load(self.decoder_path, map_location=device))

  def generate(self, text):
    text = clip.tokenize(text).to(device)
    K, M = 10, 3
    COS = nn.CosineSimilarity(dim=1, eps=1e-6)

    with torch.no_grad():
        img_enc = torch.tensor(self.train_dataset.clip_enc).to(device)
        text_enc = self.clip.encode_text(text)
        sim = COS(text_enc, img_enc)
        K_best = torch.topk(sim, K).indices

        weights = torch.ones_like(K_best, dtype=torch.float).to(device)
        samples = torch.multinomial(weights, M)
        M_best = K_best[samples]

        weights = Dirichlet(torch.tensor([1/M] * M)).sample().to(device)
        c = torch.matmul(weights, img_enc[M_best])
        c = c[None, :]

        
        z = torch.randn(1, 256).to(device)
        z = torch.cat([z, c], dim=-1)
        delta = self.decoder.mlp(z)

        law = random.choice([-.2, -.1, .1, .3])
        w = self.latent_avg + law
        image = self.sg2(w, c=None, truncation_psi=0.4, noise_mode='const')
        image = image.squeeze()
        image = image.moveaxis(0, -1)
        plt.imshow(image.to('cpu'))
        plt.show()


      