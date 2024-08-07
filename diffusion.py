import torch

import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import math


class VarianceScheduler:
    """
    This class is used to keep track of statistical variables used in the diffusion model
    and also adding noise to the data
    """
    def __init__(self, beta_start: float=0.0004, beta_end: float=0.01, num_steps :int=1000):
        self.num_steps = num_steps

        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x: torch.Tensor, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method receives the input data and the timestep, generates a noise according to the
        timestep, perturbs the data with the noise, and returns the noisy version of the data and
        the noise itself

        Args:
            x (torch.Tensor): input image [B, 1, 28, 28]
            timestep (torch.Tensor): timesteps [B]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: noisy_x [B, 1, 28, 28], noise [B, 1, 28, 28]
        """
        # generate noise according to the timestep
        noise = torch.randn_like(x, device=x.device)

        # calculate mean and standard deviation
        sqrt_a_bar = torch.sqrt(self.alpha_bars.to(x.device)[timestep])
        mean = sqrt_a_bar[:, None, None, None].to(x.device) * x

        sqrt_1_a_bar = torch.sqrt(1.0 - self.alpha_bars.to(x.device)[timestep])
        std = sqrt_1_a_bar[:, None, None, None].to(x.device) * noise

        # perturb the data with the noise
        noisy_x = mean + std

        return noisy_x, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings

class TimeEmbeddings(nn.Module):
    def __init__(self, time_emb_dim, out):
        super().__init__()
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU(),
                nn.Linear(time_emb_dim, out))
    def forward(self, x, t):
        return x + self.time_mlp(t).unsqueeze(-1).unsqueeze(-1)


class ClassEmbeddings(nn.Module):
    def __init__(self, class_emb_dim, out, num_classes=10):
        super().__init__()
        self.class_mlp = nn.Sequential(
                nn.Embedding(num_classes, class_emb_dim),
                nn.Linear(class_emb_dim, out))

    def forward(self, x, y):
        return x + self.class_mlp(y).unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.net = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        x = self.conv(x)

        # apply residual block
        return self.net(x) + x


class NoiseEstimatingNet(nn.Module):
    """
    The implementation of the noise estimating network for the diffusion model.
    A manual UNet
    """
    def __init__(self, time_emb_dim: int, class_emb_dim: int, num_classes: int=10):
        super().__init__()
        t = time_emb_dim
        c = class_emb_dim
        hid_ch = [64, 128]  # hidden channels

        # Down Sampling
        self.time1 = TimeEmbeddings(t, 1)
        self.block1 = ResBlock(1, hid_ch[0])
        self.class1 = ClassEmbeddings(c, hid_ch[0], num_classes)
        self.time2 = TimeEmbeddings(t, hid_ch[0])

        self.down1 = nn.Conv2d(hid_ch[0], hid_ch[0], 4, stride=2, padding=1)
        self.block2 = ResBlock(hid_ch[0], hid_ch[1])
        self.class2 = ClassEmbeddings(c, hid_ch[1], num_classes)
        self.time3 = TimeEmbeddings(t, hid_ch[1])

        # Middle
        self.down2 = nn.Conv2d(hid_ch[1], hid_ch[1], 4, stride=2, padding=1)
        self.block3 = ResBlock(hid_ch[1], hid_ch[1])
        self.class3 = ClassEmbeddings(c, hid_ch[1], num_classes)
        self.attention = nn.MultiheadAttention(hid_ch[1], 4)

        # Up Sampling
        self.up1 = nn.ConvTranspose2d(hid_ch[1], hid_ch[1], 4, stride=2, padding=1)
        self.time4 = TimeEmbeddings(t, hid_ch[1]*2)
        self.block4 = ResBlock(hid_ch[1]*2, hid_ch[0])
        self.class4 = ClassEmbeddings(c, hid_ch[0], num_classes)

        self.up2 = nn.ConvTranspose2d(hid_ch[0], hid_ch[0], 4, stride=2, padding=1)
        self.class5 = ClassEmbeddings(c, hid_ch[0]*2, num_classes)
        self.time5 = TimeEmbeddings(t, hid_ch[0]*2)
        self.block5 = ResBlock(hid_ch[0]*2, 1)


    def forward(self, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate the noise given the input image, timestep, and the label

        Args:
            x (torch.Tensor): the input (noisy) image [B, 1, 28, 28]
            timestep (torch.Tensor): timestep [B]
            y (torch.Tensor): the corresponding labels for the images [B]

        Returns:
            torch.Tensor: out (the estimated noise) [B, 1, 28, 28]
        """
        # Down Sampling
        x1 = self.time1(x, timestep)
        x1 = self.block1(x1)
        x1 = self.class1(x1, y)
        x1 = self.time2(x1, timestep)

        x2 = self.down1(x1)  # [B, ]
        x2 = self.block2(x2)
        x2 = self.class2(x2, y)
        x2 = self.time3(x2, timestep)

        # Middle
        m = self.down2(x2)
        m = self.block3(m)
        m = self.class3(m, y)
        m = self.attention(m)

        # Up Sampling
        y2 = self.up1(m)
        y2 = torch.cat([x2, y2], dim=1)  # apply res
        y2 = self.time4(y2, timestep)
        y2 = self.block4(y2)
        y2 = self.class4(y2, y)

        y1 = self.up2(y2)
        y1 = torch.cat([y1, x1], dim=1)  # apply res
        y1 = self.class5(y1, y)
        y1 = self.time5(y1, timestep)
        return self.block5(y1)


class DiffusionModel(nn.Module):
    """
    The whole diffusion model put together
    """
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler):
        """

        Args:
            network (nn.Module): your noise estimating network
            var_scheduler (VarianceScheduler): variance scheduler for getting
                                the statistical variables and the noisy images
        """

        super().__init__()

        self.network = network
        self.var_scheduler = var_scheduler

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.float32:
        """
        The forward method for the diffusion model gets the input images and
        their corresponding labels

        Args:
            x (torch.Tensor): the input image [B, 1, 28, 28]
            y (torch.Tensor): labels [B]

        Returns:
            torch.float32: the loss between the actual noises and the estimated noise
        """

        # sample timesteps
        timesteps = torch.randint(0, self.var_scheduler.num_steps, (x.size(0),), device=x.device)

        # compute the noisy versions of the input image according to your timesteps
        noisy_x, noise = self.var_scheduler.add_noise(x, timesteps)

        # estimate the noises using your noise estimating network
        estimate = self.network(noisy_x, timesteps, y)

        # compute the loss between the estimated noises and the true noises
        loss = F.mse_loss(noise, estimate)

        return loss

    @torch.no_grad()
    def generate_sample(self, num_images: int, y, device) -> torch.Tensor:
        """
        This method generates as many samples as specified according to the given labels

        Args:
            num_images (int): number of images to generate
            y (_type_): the corresponding expected labels of each image
            device (_type_): computation device (e.g. torch.device('cuda'))

        Returns:
            torch.Tensor: the generated images [num_images, 1, 28, 28]
        """
        # get the variables from the variance scheduler
        a_ = self.var_scheduler.alpha_bars.to(device)

        x = torch.randn((num_images, 1, 28, 28), device=device)

        for t in range(self.var_scheduler.num_steps-1, -1, -1):  # [999, 998 ... 0]
            timesteps = t * torch.ones(num_images, device=device)  # [1] -> [B]
            et = self.network(x, timesteps, y)

            sqrt_a_ = torch.sqrt(a_[t])
            sqrt_1_a_ = torch.sqrt(1 - a_[t])
            theta = (x - sqrt_1_a_ * et) / sqrt_a_

            if t != 0:
                std = torch.sqrt(a_[t-1]) * theta
                mean = torch.sqrt(1 - a_[t-1]) / sqrt_1_a_ * \
                            (x - sqrt_a_ * theta)
                x = std + mean
            else:
                x = theta

        return x


def load_diffusion_and_generate():
    device = torch.device('cuda')
    var_scheduler = VarianceScheduler()
    network = NoiseEstimatingNet(time_emb_dim=32, class_emb_dim=10)
    diffusion = DiffusionModel(network=network, var_scheduler=var_scheduler)

    # loading the weights of Diffusion
    diffusion.load_state_dict(torch.load('diffusion.pt'))
    diffusion = diffusion.to(device)

    desired_labels = []
    for i in range(10):
        for _ in range(5):
            desired_labels.append(i)

    desired_labels = torch.tensor(desired_labels).to(device)
    generated_samples = diffusion.generate_sample(50, desired_labels, device)

    return generated_samples
