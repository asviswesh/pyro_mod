This repository contains the code for a CVAE which predicts an MNIST image of a digit based on the corresponding digit passed in.

To run, download this repository and create the custom conda environment as follows from the .yml file provided.

```bash
conda env create -f model.yml
```

Activate the environment in the manner below

```bash
conda activate myenv
```

Once the conda environment has been created, create a folder called 'data' inside the same root directory as the directory of the downloaded code. We would recommend grouping all files contained into this repository into one folder (calling it 'cvae' for example), and consequently creating the 'data' folder in the same directory as the 'cvae' folder. Then, run the following

```bash
<location of custom conda environment> <filepath to main.py>
```
