import clip
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

class WNormLoss(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, latent, latent_avg=None):
		w = latent
		w_norm = torch.norm(w, p=2, dim=1)
		w_norm_loss = torch.mean((w_norm - 1) ** 2)

		return w_norm_loss


class CLIPLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.model.eval()
        self.upsample = nn.Upsample(scale_factor=7)
        self.kk = int(256/8)
        self.avg_pool = nn.AvgPool2d(kernel_size=256 // self.kk)

    def forward(self, recon, orig_features):
        recon = self.avg_pool(self.upsample(recon))
        orig_features = orig_features/(orig_features.norm(dim=1,keepdim=True)+1e-8)
        recon_features = self.model.encode_image(recon)
        recon_features = recon_features/(recon_features.norm(dim=1,keepdim=True)+1e-8)
        similarity = (orig_features*recon_features).sum(dim=1)

        return (1-similarity).mean()
    

class KLDivLoss(nn.Module):
	def __init__(self):
		super(KLDivLoss, self).__init__()

	def forward(self, mu_q, logvar_q, mu_p, logvar_p):
		kld = -0.5 * torch.sum(1 + logvar_q - logvar_p - (mu_q - mu_p)**2 / logvar_p.exp() - logvar_q.exp() / logvar_p.exp(), dim=1)

		return torch.mean(kld)

  
