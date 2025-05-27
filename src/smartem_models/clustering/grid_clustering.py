"""
Created on Mon Jul  3 09:46:05 2023

@author: jaehoon cha
@email: jaehoon.cha@stfc.ac.uk
"""

import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from smartem_models.clustering.calldata import smartem
from smartem_models.clustering.models import EIAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="eiae")
    parser.add_argument("--path_dir", type=str, default=os.getcwd())
    parser.add_argument("--dataset", type=str, default="grid")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--input_dim", type=tuple, default=(1, 64, 64))
    parser.add_argument("--hidden_dims", type=tuple, default=(1, 16, 32, 64, 128))
    parser.add_argument("--lat_dim", type=int, default=2)
    parser.add_argument("--rnd", type=int, default=14)
    parser.add_argument("--grid_id", type=int, default=33)

    args = parser.parse_args()

    config = OrderedDict(
        [
            ("model_name", args.model_name),
            ("path_dir", args.path_dir),
            ("dataset", args.dataset),
            ("epochs", args.epochs),
            ("batch_size", args.batch_size),
            ("lr", args.lr),
            ("lr_decay", args.lr_decay),
            ("input_dim", args.input_dim),
            ("hidden_dims", args.hidden_dims),
            ("lat_dim", args.lat_dim),
            ("rnd", args.rnd),
            ("grid_id", args.grid_id),
        ]
    )

    return config


config = parse_args()

data_path = os.path.join(config["path_dir"], "datasets")

### call data ###
config["datasets"] = "grid"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_x = smartem(data_path, config["datasets"], grid_id=config["grid_id"])

train_dataloader = DataLoader(train_x, batch_size=config["batch_size"], shuffle=True, pin_memory=True)

try:
    os.mkdir("results")
except OSError:
    pass

np.random.seed(config["rnd"])
torch.manual_seed(config["rnd"])


save_name = "pretrained/model.pth"
model = EIAE(
    alpha=torch.Tensor([[1.0, 1.0]]).to(device),
    input_dims=config["input_dim"],
    hidden_dims=config["hidden_dims"],
    lat_dim=config["lat_dim"],
)
model.load_state_dict(torch.load(save_name, map_location=torch.device("cpu")))
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999))


def bvae_loss(x, logits):
    recon_x, c, z, loc = logits

    MSE = torch.sum((recon_x - x) ** 2, axis=(1, 2, 3))

    return MSE.mean()


def train(epoch: int, dataloader: DataLoader, inmodel: EIAE, inoptimizer: torch.optim.Adam, indevice: str):
    loss_record = []
    epoch_loss = 0.0

    inmodel.train()
    for samples in tqdm(dataloader):
        data, _ = Variable(samples["x1"]).to(indevice), Variable(samples["x1"]).to(indevice)

        inoptimizer.zero_grad()

        outputs = inmodel(data)
        loss = bvae_loss(data.detach().clone(), outputs)

        loss_record.append(loss.detach().cpu().numpy())

        epoch_loss += loss.detach().cpu().numpy()
        loss.backward()
        inoptimizer.step()

    epoch_loss = epoch_loss / len(dataloader)

    print("Train Epoch: {}/{} Loss: {:.4f}".format(epoch, config["epochs"], loss.data))
    return loss_record, epoch_loss


def get_recon_loss():
    loss_record = 0.0
    model.eval()
    for samples in train_dataloader:
        data, target = Variable(samples["x1"]).to(device), Variable(samples["x1"]).to(device)
        output = model(data)
        recons_loss = torch.sum((output - target) ** 2, axis=(1, 2, 3)).mean()
        loss_record += recons_loss.detach().cpu().numpy()
    return loss_record / len(train_dataloader)


def run():
    losses = []
    epoch_losses = [10**15]

    for epoch in range(config["epochs"]):
        loss_record, epoch_loss = train(epoch, train_dataloader, model, optimizer, device)
        losses += loss_record
        epoch_losses.append(epoch_loss)

    return losses


stime = time.time()
losses = run()
etime = time.time()
training_time = [etime - stime]

save_folder = os.path.join("results", f"{config['grid_id']}")

try:
    os.mkdir(save_folder)
except OSError:
    pass

save_name = os.path.join(save_folder, "model.pth")
torch.save(model.state_dict(), save_name)
