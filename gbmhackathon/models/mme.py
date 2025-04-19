import torch
import torch.nn as nn


class HnEEncoder(nn.Module):
    """Encoder for HnE data"""

    pass


class SpatialEncoder(nn.Module):
    """Encoder for Spatial transcriptmic data"""

    pass


class BulkEncoder(nn.Module):
    """Encoder for BulkRNAseq data"""

    pass


class SingleCellEncoder(nn.Module):
    """Encoder for scRNAseq data"""

    pass


class WESEncoder(nn.Module):
    """Encoder for Whole Exome Sequencing data"""

    pass


class ClinicalEncoder(nn.Module):
    """Encoder for Clinical data"""

    pass


class MultiModalEncoder(nn.Module):
    """Global encoder that encompasses all 6 modalities"""

    pass
