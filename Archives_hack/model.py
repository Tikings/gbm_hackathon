import numpy as np
import pandas as pd
import pickle as pkl

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import novae

# WES Encoder
class DNAAutoencoderCNN(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, dropout_rate=0.2):
        super(DNAAutoencoderCNN, self).__init__()

        # 🔹 Encoder
        self.conv1d = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        conv_out_dim = (input_dim // 2) * 8  # reduced size by convolution
        self.encoder = nn.Sequential(
            nn.Linear(conv_out_dim, middle_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(middle_dim, output_dim) 
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, conv_out_dim),
            nn.ReLU()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="linear")  # To restore the dimension
        self.deconv1d = nn.ConvTranspose1d(in_channels=8, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        # 🔹 Encoder
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  
        encoded = self.encoder(x)  
        
        # Decoder
        x = self.decoder(encoded)
        x = x.view(x.shape[0], 8, -1)  # Restauration to the format of deconv1
        x = self.upsample(x)  # restauration of original size
        x = self.deconv1d(x)  # Return [batch, 3, input_dim]

        return encoded, x  # Embedding + reconstruction

# Spatial Transcriptomics Encoder
class ST_Encoder(nn.Module):
    def __init__(self, embedding_dico, emb_size=64):
        self.
        self.graph_encoder = novae.module.AttentionAggregation(emb_size)
        

# HnE and clinical Encoder
class Small_MLP_Encoder(nn.Module):
    def __init__(self, embedding_dico, h1, h2, out_size, input_type):
        super().__init__()
        self.input_embedding = embedding_dico

        if input_type == "torch":
            self.in_size = embedding_dico[list(embedding_dico.keys())[0]].size()[-1]
        elif input_type == "dataframe":
            self.in_size = embedding_dico[list(embedding_dico.keys())[0]].shape[-1]
        self.encoder = nn.Sequential(nn.Linear(self.in_size, h1),
                                 nn.Mish(),
                                 nn.Dropout(0.1),
                                 nn.Linear(h1, h2),
                                 nn.Mish(),
                                 nn.Dropout(0.1),
                                 nn.Linear(h2, out_size))

        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, sample_id):
        # One sample ID
        emb = self.input_embedding[sample_id]
        # Encoder
        new_emb = self.encoder(emb)
        return new_emb
        

        
                                 
                                 
# bulkRNAseq Encoder

# scRNAseq Encoder

# clinical Encoder