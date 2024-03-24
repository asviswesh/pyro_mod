# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from baseline import BCELoss
from mnist import get_data
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from pyro.infer import Predictive, Trace_ELBO

# Modified to not include num_quadrant_inputs
def generate_table(
    device,
    pre_trained_baseline,
    pre_trained_cvae,
    num_particles,
    col_name,
):
    # Load sample random data
    _, dataloaders, dataset_sizes = get_data(
            batch_size=32
    )

    # Load sample data
    criterion = BCELoss()
    loss_fn = Trace_ELBO(num_particles=num_particles).differentiable_loss

    baseline_cll = 0.0
    cvae_mc_cll = 0.0
    num_preds = 0

    df = pd.DataFrame(index=["NN (baseline)", "CVAE (Monte Carlo)"], columns=[col_name])

    # Iterate over data.
    bar = tqdm(dataloaders["val"], desc="Generating predictions".ljust(20))
    for batch in bar:
        inputs = batch["digit"].to(device)
        outputs = batch["original"].to(device)
        num_preds += 1

        # Compute negative log likelihood for the baseline NN
        with torch.no_grad():
            preds = pre_trained_baseline(inputs)
        baseline_cll += criterion(preds, outputs).item() / inputs.size(0)

        # Compute the negative conditional log likelihood for the CVAE
        cvae_mc_cll += loss_fn(
            pre_trained_cvae.model, pre_trained_cvae.guide, inputs, outputs
        ).detach().item() / inputs.size(0)

    df.iloc[0, 0] = baseline_cll / num_preds
    df.iloc[1, 0] = cvae_mc_cll / num_preds
    return df
