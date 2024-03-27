# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO


class PriorNet(nn.Module):
    def __init__(self, x_dim, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, hidden_1) 
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        hidden = self.relu(self.fc1(x.float()))
        hidden = self.relu(self.fc2(hidden))
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return z_loc, z_scale


class GenerationNet(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, z):
        y = self.relu(self.fc1(z))
        y = self.relu(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        y = y.view(-1, 1, 28, 28)
        return y


class RecognitionNet(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(x_dim + y_dim, hidden_1) 
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        y_flat = y.view(y.size(0), -1)
        combined_tensor = torch.cat((x, y_flat), dim=1)
        hidden = self.relu(self.fc1(combined_tensor.float()))
        hidden = self.relu(self.fc2(hidden))
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return z_loc, z_scale


class CVAE(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, hidden_1, hidden_2):
        super().__init__()
        # The CVAE is composed of multiple MLPs, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z).
        self.prior_net = PriorNet(x_dim, z_dim, hidden_1, hidden_2)
        self.generation_net = GenerationNet(z_dim, hidden_1, hidden_2)
        self.recognition_net = RecognitionNet(x_dim, y_dim, z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("generation_net", self)
        with pyro.plate("data"):
            # Create latent variable z based on the one-hot encoded digit
            # returns parameters for the Gaussian distribution to sample from.
            prior_loc, prior_scale = self.prior_net(xs)

            # Latent variable is a sample from the Gaussian distribution.
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))
            
            # the output y (image) is generated from the distribution pθ(y|x, z)
            loc = self.generation_net(zs)
            if ys is not None:
                loc_flat = loc.view(loc.shape[0], -1) 
                pyro.sample(
                "y",
                dist.Bernoulli(loc_flat, validate_args=False).to_event(1),
                obs=ys.view(ys.shape[0], -1)  # Flatten ys similarly
                ) 
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                # at inference time, ys is not provided. In that case,
                # the model uses the prior network
                loc, scale = self.prior_net(xs)
            else:
                # at training time, uses the variational distribution
                # q(z|x,y) = normal(loc(x,y),scale(x,y))
                loc, scale = self.recognition_net(xs, ys)

            pyro.sample("z", dist.Normal(loc, scale).to_event(1))


def train(
    device,
    dataloaders,
    dataset_sizes,
    learning_rate,
    num_epochs,
    early_stop_patience,
    model_path,
):
    # clear param store
    pyro.clear_param_store()

    cvae_net = CVAE(10, 784, 100, 500, 500) 
    cvae_net.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())


    best_loss = np.inf
    early_stop_count = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            running_loss = 0.0
            num_preds = 0

            # Iterate over data.
            bar = tqdm(
                dataloaders[phase],
                desc="CVAE Epoch {} {}".format(epoch, phase).ljust(20),
            )
            for i, batch in enumerate(bar):
                inputs = batch["digit"].to(device) 
                outputs = batch["original"].to(device) 

                if phase == "train":
                    loss = svi.step(inputs, outputs)
                else:
                    loss = svi.evaluate_loss(inputs, outputs)

                # statistics
                running_loss += loss / inputs.size(0)
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(
                        loss="{:.3f}".format(running_loss / num_preds),
                        early_stop_count=early_stop_count,
                    )

            epoch_loss = running_loss / dataset_sizes[phase]
            # deep copy the model
            if phase == "val":
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(cvae_net.state_dict(), model_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    # Save model weights
    cvae_net.load_state_dict(torch.load(model_path))
    cvae_net.eval()
    return cvae_net
