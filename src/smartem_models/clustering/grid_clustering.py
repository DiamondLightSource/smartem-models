"""
Created on Mon Jul  3 09:46:05 2023

@author: jaehoon cha
@email: jaehoon.cha@stfc.ac.uk
"""

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from smartem_models.clustering.models import EIAE


def bvae_loss(x, logits):
    recon_x, c, z, loc = logits

    MSE = torch.sum((recon_x - x) ** 2, axis=(1, 2, 3))

    return MSE.mean()


def train(dataloader: DataLoader, inmodel: EIAE, inoptimizer: torch.optim.Adam, indevice: str):
    loss_record = []
    epoch_loss = 0.0

    inmodel.train()
    for samples in dataloader:
        data, _ = Variable(samples["x1"]).to(indevice), Variable(samples["x1"]).to(indevice)

        inoptimizer.zero_grad()

        outputs = inmodel(data)
        loss = bvae_loss(data.detach().clone(), outputs)

        loss_record.append(loss.detach().cpu().numpy())

        epoch_loss += loss.detach().cpu().numpy()
        loss.backward()
        inoptimizer.step()

    epoch_loss = epoch_loss / len(dataloader)

    return loss_record, epoch_loss


def get_recon_loss(dataloader: DataLoader, model: EIAE, device: str):
    loss_record = 0.0
    model.eval()
    for samples in dataloader:
        data, target = Variable(samples["x1"]).to(device), Variable(samples["x1"]).to(device)
        output = model(data)
        recons_loss = torch.sum((output - target) ** 2, axis=(1, 2, 3)).mean()
        loss_record += recons_loss.detach().cpu().numpy()
    return loss_record / len(dataloader)
