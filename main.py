# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import baseline
import cvae
import pandas as pd
import torch
from torchvision.utils import save_image
from util import generate_table, get_data, visualize

import pyro


def main(args):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    results = []
    columns = []

    print("Running for theta = {}".format(args.theta_input))
    label = torch.tensor(args.theta_input)
    print(label)
    _, dataloaders, dataset_sizes = get_data(
                batch_size=128
        )
    
    # Ran to compare log likelihoods.
    baseline_net = baseline.train(
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stop_patience=args.early_stop_patience,
            model_path="baseline_net_theta{}.pth".format(args.theta_input),
        )
    
    cvae_net = cvae.train(
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stop_patience=args.early_stop_patience,
            model_path="cvae_net_theta{}.pth".format(args.theta_input),
        )
    

    
    print("Finished running CVAE!")

    reconstructed_image = cvae_net.model(label).detach().cpu()
    save_image(reconstructed_image, f"/Users/aviswesh/Downloads/reconstructed_digit_{args.theta_input}.png")
    # Retrive conditional log-likelihood
    df = generate_table(
            device=device,
            pre_trained_baseline=baseline_net,
            pre_trained_cvae=cvae_net,
            num_particles=args.num_particles,
            col_name="Log Likelihoods",
        )
    
    results.append(df)
    print(f"The log likelihoods are {results}")


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.9.0")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-theta",
        "--theta-input",
        type=int,
        default=1,
        help="theta: the digit to reproduce in the model output",
    )
    parser.add_argument(
        "-n", "--num-epochs", default=5, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-esp", "--early-stop-patience", default=3, type=int, help="early stop patience"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-3, type=float, help="learning rate"
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

