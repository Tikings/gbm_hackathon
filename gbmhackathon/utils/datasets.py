import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PatientWiseDataset(Dataset):
    """Dataset for Patient wise representation learning"""

    pass


class PredictiveDataset(Dataset):
    """Dataset for clinically relevant predictve tasks learning"""

    pass
