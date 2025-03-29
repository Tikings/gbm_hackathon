<div align="center">
<h1>GBM Hackathon: Data Loading</h1>
Data loading API and data visualization of GBM Hackathon data.
</div>

# Table of Contents
- [Installation](#installation)
- [Starters](#starters)
- [Credit](#credit)


# Installation
1. Start by initialising Conda.
```bash
conda init bash && bash
```

2. Created `.env` file to get the access keys of the data storage (DON'T ACTIVATE THE CONDA ENV !!)
```bash
abstra mkenv --eval > .env
export $(abstra mkenv --eval)
```

3. Create the environment.
```bash
conda create -n gbmhackathon -y python=3.10
conda activate gbmhackathon
```

4. Optional, fork the repository
If you aim to share your code with your team members, consider doing
a fork or create a copy of the actual repository with a new name.

5. Clone the repository and install the dependencies.
```bash
git clone https://github.com/owkin/gbm_hackathon.git
cd gbm_hackathon
make install-poetry
make install-all
pip install -r requirements_foundation.txt
```

Note that this also makes your environment visible in Jupyter.


# Starter Notebooks
To help you start working quickly on the data, we provided you with the following starter Notebooks.
- `notebooks/data_loader_demo.ipynb` Will help you understand the data loading and which data sources (modalities) are available.
- `notebooks/visium_starter_mosaic.ipynb` Will help you load and explore the Visium data. This includes performing data normalization, unsupervised clustering, and quantifying cell population and transcription factor activities.

We hope these Notebooks will help you with your data analyses!


# Credit
We would like to thank the following contributors to this project: Lucas Fidon, Quentin Bayard, Alex Cornish and Valérie Ducret.
