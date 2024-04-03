# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import cvae
import pandas as pd
import torch
from torchvision.utils import save_image
from util import generate_table, get_data, one_hot_encode

import pyro


def main(args):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    _, dataloaders, dataset_sizes = get_data(
                    batch_size=128
            )
    
    cvae_net = cvae.train(
                device=device,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                early_stop_patience=args.early_stop_patience,
                model_path="cvae_net.pth",
            )
    for i in range(1, 6):
        for theta_input in args.theta_inputs:
            label = theta_input
            print("Reconstructing digit = {}".format(label))
            
            one_hot_label = one_hot_encode(label)
            
            reconstructed_image = cvae_net.model(one_hot_label).detach().cpu()
            reconstructed_image = reconstructed_image.view(1, 28, 28)
            save_image(reconstructed_image, f"../image_results/digit_{label}_run_{i}_no_bern.png")


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.9.0")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-theta",
        "--theta-inputs",
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="theta: the digit to reproduce in the model output",
    )
    parser.add_argument(
        "-n", "--num-epochs", default=20, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-esp", "--early-stop-patience", default=3, type=int, help="early stop patience"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument(
        "-vi",
        "--num-images",
        default=10,
        type=int,
        help="number of images to visualize",
    )
    parser.add_argument(
        "-vs",
        "--num-samples",
        default=10,
        type=int,
        help="number of samples to visualize per image",
    )
    parser.add_argument(
        "-p",
        "--num-particles",
        default=10,
        type=int,
        help="n of particles to estimate logpθ(y|x,z) in ELBO",
    )
    args = parser.parse_args()

    main(args)

