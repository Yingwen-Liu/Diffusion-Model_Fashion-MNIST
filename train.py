import torch

import matplotlib.pyplot as plt

from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets.mnist import FashionMNIST

from diffusion import DiffusionModel


# loading dataset
kwargs = {'root':'datasets/FashionMNIST',
          'train':True,
          'transform':transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: 2 * x - 1.)]),
          'download':True}

train_dataset = FashionMNIST(**kwargs)

train_dataset, val_dataset = random_split(train_dataset, [len(train_dataset) - 12000, 12000])


def train(diffusion_model: DiffusionModel,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: optim,
          epochs: int,
          device=torch.device('cuda'),
          display_interval: int=5):


  itrs = tqdm(range(epochs))
  itrs.set_description('Train Loss: ? - Val Loss: ?')

  best_loss = float('inf')

  for epoch in itrs:
    avg_train_loss = 0.
    for sample in train_loader:
      x = sample[0].to(device)
      label = sample[1].type(torch.long).to(device)

      optimizer.zero_grad()

      loss = diffusion_model(x, label)

      loss.backward()
      optimizer.step()

      avg_train_loss += loss.item()

    avg_train_loss /= len(train_loader)

    # validating and saving the model
    with torch.no_grad():
      avg_val_loss = 0.
      for sample in val_loader:
        x = sample[0].to(device)
        label = sample[1].type(torch.long).to(device)

        loss = diffusion_model(x, label)

        avg_val_loss += loss.item()

      avg_val_loss /= len(val_loader)

    itrs.set_description(f'Train Loss: {avg_train_loss:.3f} - Val Loss: {avg_val_loss:.3f}')

    # save the model on the best validation loss
    if best_loss > avg_val_loss:
      best_loss = avg_val_loss
      torch.save(diffusion_model.state_dict(), 'diffusion.pt')

    if display_interval is not None:
      if (epoch % display_interval == 0 or epoch == epochs - 1) and epoch != 0:
        # generate some sample to see the quality of the generative model
        samples = diffusion_model.generate_sample(10, torch.arange(10).cuda(), torch.device('cuda'))
        fig, ax = plt.subplots(1, 10)
        fig.set_size_inches(15, 10)
        for i in range(10):
          ax[i].set_xticks([])
          ax[i].set_yticks([])
          ax[i].imshow(samples[i].cpu().permute(1, 2, 0).numpy(), cmap='gray')
        plt.show()


# training the diffusion model
device = torch.device('cuda')
num_steps = 1000 # define the number of steps (>500)
batch_size = 100 # define your batch size
lr = 0.0005
epochs = 100
num_classes = 10
display_interval = 20

# defining the diffusion model component nets
var_scheduler = VarianceScheduler()
noise_net = NoiseEstimatingNet(time_emb_dim=32, class_emb_dim=10)

# loading a train and validation data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# defining the diffusion model
diffusion = DiffusionModel(noise_net, var_scheduler).to(device)
#diffusion.load_state_dict(torch.load('diffusion.pt'))

optimizer = optim.Adam(diffusion.parameters(), lr)

# training the diffusion model
train(diffusion_model=diffusion,
      train_loader=train_loader,
      val_loader=val_loader,
      optimizer=optimizer,
      epochs=epochs,
      device=device,
      display_interval=display_interval)
