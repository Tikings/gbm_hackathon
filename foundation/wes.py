# Do functions to retrieve embedding for WES (they will be of shape n x genes x channels)
# Check why Jhon did not use oncogenic and why he did not apply dimensionality reduction
# Check for NAs


# Load packages and classes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tiffslide
import seaborn as sns
import gget
import tifffile
import zarr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# MosaicDataset and BruceDataset classes allow loading and visualisation of the different data sources
from gbmhackathon import MosaicDataset

source_dict_mosaic = MosaicDataset.load_tabular()

onc = source_dict_mosaic["wes"]["WES CNV oncogenic"]
dele = source_dict_mosaic["wes"]["WES CNV deletion"]
amp = source_dict_mosaic["wes"]["WES CNV amplification"]
mut = source_dict_mosaic["wes"]["WES mutations"]

amp = amp[mut.columns]
dele = dele[mut.columns]

len(set(amp.columns))
column_counts = pd.Series(amp.columns).value_counts()
duplicated_columns = column_counts[column_counts > 1].index.tolist()


## Droping the duplicated column
column_counts = pd.Series(amp.columns).value_counts()
duplicated_columns = column_counts[column_counts > 1].index.tolist()

amp = amp.loc[:, ~amp.columns.duplicated()]
dele = dele.loc[:, ~dele.columns.duplicated()]

print(amp.shape)
print(dele.shape)
print(mut.shape)
