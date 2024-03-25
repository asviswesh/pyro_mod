This repository contains the code for a CVAE which predicts an MNIST image of a digit based on the corresponding digit passed in.

To run, download this repository and create a custom conda environment as follows

```bash
conda create --name <my-env>
```

Install the following libraries: numpy, Pytorch, and matplotlib using the standard conda installations in this custom environment.

To install the Pyro module within the same environment, run the following

```bash
conda install conda-forge::pyro-ppl
```

Once the conda environment has been configured, create a folder called 'data' inside the same root directory as the directory of the downloaded code. Then, run the following

```bash
<location of custom conda environment> <filepath to main.py>
```
